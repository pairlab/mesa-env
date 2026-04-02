import json
import os
import random
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import imageio
import numpy as np
import tyro

# natural sort for variant ordering
from natsort import natsorted

import mesa
import mesa.mimicgen.utils.file_utils as MG_FileUtils
import mesa.sim.envs.bddl_utils as BDDLUtils
from mesa import MESA_ROOT
from mesa.mimicgen.configs import MG_TaskSpec, finish_task_spec
from mesa.mimicgen.configs.config import MGConfig
from mesa.mimicgen.data_generator import DataGenerator
from mesa.mimicgen.env_interfaces.base import make_interface
from mesa.mimicgen.stitching_data_generator import StitchingDataGenerator
from mesa.mimicgen.utils.misc_utils import get_important_stats
from mesa.sim.envs.bddl_base_domain import TASK_MAPPING
from mesa.utils.replay import replay_dataset

# Disable numpy scientific notation
np.set_printoptions(suppress=True)


@dataclass
class GenerateDatasetArgs:
    """Arguments for the generate_dataset script.
    
    This script generates datasets using MimicGen for vision-language-action models.
    """
    config: str
    """Path to MimicGen config json (required)."""
    
    debug: bool = False
    """Set this flag to run a quick generation run for debugging purposes."""
    
    auto_remove_exp: bool = False
    """Force delete the experiment folder if it exists."""
    
    auto_keep_exp: bool = False
    """Automatically keep the experiment folder if it exists, adding new files to it instead of overwriting them."""

    render: bool = False
    """Render each data generation attempt on-screen."""

    robot: str = "Panda"
    """Robot to use for data generation."""
    
    video_path: Optional[str] = None
    """If provided, render the data generation attempts to the provided video path."""
    
    video_skip: int = 5
    """Skip every nth frame when writing video."""
    
    render_image_names: Optional[List[str]] = None
    """(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. 
    Default is None, which corresponds to a predefined camera for each env type."""
    
    pause_subtask: bool = False
    """Pause after every subtask during generation for debugging - only useful with render flag."""
    
    source: Optional[str] = None
    """Path to source dataset, to override the one in the config."""
    
    folder: Optional[str] = None
    """Folder that will be created with new data, to override the one in the config."""
    
    num_demos: Optional[int] = None
    """Number of demos to generate, or attempt to generate, per variant, to override the one in the config."""
    
    seed: Optional[int] = None
    """Seed, to override the one in the config."""
    
    variant_start: Optional[int] = None
    """0-based start index (inclusive) of variants to process."""
    
    variant_end: Optional[int] = None
    """0-based end index (inclusive) of variants to process."""
    
    disable_merge: bool = False
    """Disable merging per-episode HDF5s; write a manifest to merge later."""

    catch_exceptions: bool = False
    """Catch exceptions and continue generating data."""


def generate_dataset(mg_config: MGConfig, args: GenerateDatasetArgs):
    """
    Main function to collect a new dataset with MimicGen.

    Args:
        mg_config (MGConfig): MimicGen config object
        args (GenerateDatasetArgs): Command line arguments containing all configuration
    """

    # time this run
    start_time = time.time()

    # check some args
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both
    if args.pause_subtask:
        assert args.render, "should enable on-screen rendering for pausing to be useful"

    if write_video:
        # debug video - use same cameras as observations
        if len(mg_config.obs.camera_names) > 0:
            assert args.render_image_names is None
            render_image_names = list(mg_config.obs.camera_names)

    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

    # set seed for generation
    random.seed(mg_config.experiment.seed)
    np.random.seed(mg_config.experiment.seed)

    # create new folder for this data generation run
    base_folder = os.path.expandvars(os.path.expanduser(mg_config.experiment.generation.path))
    new_dataset_folder_name = mg_config.experiment.name
    new_dataset_folder_path = os.path.join(
        base_folder,
        new_dataset_folder_name,
    )
    print("\nData will be generated at: {}".format(new_dataset_folder_path))

    # ensure dataset folder does not exist, and make new folder
    exist_ok = False
    if os.path.exists(new_dataset_folder_path):
        assert not (args.auto_remove_exp and args.auto_keep_exp), "cannot both auto-remove and auto-keep experiment folder"
        if args.auto_keep_exp:
            print("Keeping old dataset folder. Note that individual files may still be overwritten.")
            exist_ok = True
        else:
            if not args.auto_remove_exp:
                ans = input("\nWARNING: dataset folder ({}) already exists! \noverwrite? (y/n)\n".format(new_dataset_folder_path))
            else:
                ans = "y"
                
            if ans == "y":
                print("Removed old results folder at {}".format(new_dataset_folder_path))
                shutil.rmtree(new_dataset_folder_path)
            else:
                print("Keeping old dataset folder. Note that individual files may still be overwritten.")
                exist_ok = True
    os.makedirs(new_dataset_folder_path, exist_ok=exist_ok)

    # save config to disk
    mg_config.dump(os.path.join(new_dataset_folder_path, "mg_config.json"))

    print("\n============= Config =============")
    print(mg_config)
    print("")

    # some paths that we will create inside our new dataset folder

    # new dataset that will be generated
    new_dataset_path = os.path.join(new_dataset_folder_path, "demo.hdf5")

    # tmp folder that will contain per-episode hdf5s that were successful (they will be merged later)
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    os.makedirs(tmp_dataset_folder_path, exist_ok=exist_ok)

    # folder containing logs
    json_log_path = os.path.join(new_dataset_folder_path, "logs")
    os.makedirs(json_log_path, exist_ok=exist_ok)

    if mg_config.experiment.generation.keep_failed:
        # new dataset for failed trajectories, and tmp folder for per-episode hdf5s that failed
        new_failed_dataset_path = os.path.join(new_dataset_folder_path, "demo_failed.hdf5")
        tmp_dataset_failed_folder_path = os.path.join(new_dataset_folder_path, "tmp_failed")
        os.makedirs(tmp_dataset_failed_folder_path, exist_ok=exist_ok)

    if not mg_config.experiment.generation.stitching:
        # get list of source demonstration keys from source hdf5
        all_demos = MG_FileUtils.get_all_demos_from_dataset(
            dataset_path=source_dataset_path,
            filter_key=mg_config.experiment.source.filter_key,
            start=mg_config.experiment.source.start,
            n=mg_config.experiment.source.n,
        )

    # prepare args for creating simulation environment

    # auto-fill camera rendering info if not specified
    if args.render_image_names is not None:
        render_image_names = args.render_image_names
    else:
        render_image_names = ["leftshoulder"]
    if args.render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    # env args: cameras to use come from debug camera video to write, or from observation collection
    camera_names = (mg_config.obs.camera_names if not write_video else render_image_names)

    task_folder = os.path.join(
        MESA_ROOT,
        "task_suites",
        "bddl_files",
        mg_config.task_suite_name,
        mg_config.name,
        "train"
    )

    # list and deterministically order variants; optionally restrict to a start/end range
    variants = natsorted(os.listdir(task_folder))
    total_variants = len(variants)
    sel_start = 0 if (args.variant_start is None) else max(0, int(args.variant_start))
    sel_end = (total_variants - 1) if (args.variant_end is None) else min(total_variants - 1, int(args.variant_end))
    if sel_start > sel_end:
        raise ValueError(f"variant_start ({sel_start}) > variant_end ({sel_end}) after bounding; nothing to process")
    print(f"\nVariant selection: total={total_variants}, processing indices [{sel_start}, {sel_end}] inclusive")

    total_num_success = 0
    total_num_failures = 0
    total_num_attempts = 0
    total_num_problematic = 0
    total_ep_lengths = []
    
    existing_variants = set([int(name.split("_")[0]) for name in os.listdir(tmp_dataset_folder_path)])

    for i, variant in enumerate(variants):
        
        if int(variant.split(".")[0]) in existing_variants:
            print(f"Skipping variant {variant} because it already exists")
            continue

        try:

            # skip variants before start; break after end for efficiency
            if i < sel_start:
                continue
            if i > sel_end:
                break
            variant_path = os.path.join(task_folder, variant)

            parsed_problem = BDDLUtils.load_problem(variant_path)

            # simulation environment
            env = mesa.make_env(
                parsed_problem=parsed_problem,
                robots=[args.robot],
                camera_names=camera_names,
                has_renderer=args.render,
                control_delta=True,
            )


            if i == 0:
                print("\n==== Using MESA environment with the following BDDL ====")
                print(json.dumps(parsed_problem, indent=4))
                print("")

            # get information necessary to create env interface

            # create environment interface to use during data generation
            # TODO: I don't think hardcoding is an issue here but tbd
            env_interface = make_interface(
                name="MESAInterface",
                interface_type="mesa",
                # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
                env=env,
            )
            if i == 0:
                print("Created environment interface: {}".format(env_interface))
                print("")

            # get task spec object from config
            task_spec_json_string = json.dumps(mg_config.task.task_spec)
            task_spec_json_dict = json.loads(task_spec_json_string)
            task_spec_json_dict = finish_task_spec(task_spec_json_dict, parsed_problem)
            task_spec = MG_TaskSpec.from_json(json_dict=task_spec_json_dict)
            
            if mg_config.experiment.generation.stitching:
                # make stitching data generator object; dataset_path is a folder with demo_lookup.json
                data_generator = StitchingDataGenerator(
                    task_spec=task_spec,
                    dataset_path=source_dataset_path,
                    demo_keys=None,
                )
            else:
                # get list of source demonstration keys from source hdf5
                all_demos = MG_FileUtils.get_all_demos_from_dataset(
                    dataset_path=source_dataset_path,
                    filter_key=mg_config.experiment.source.filter_key,
                    start=mg_config.experiment.source.start,
                    n=mg_config.experiment.source.n,
                )

                # make data generator object
                data_generator = DataGenerator(
                    task_spec=task_spec,
                    dataset_path=source_dataset_path,
                    demo_keys=all_demos,
                )

            if i == 0:
                print("\n==== Created Data Generator ====")
                print(data_generator)
                print("")

            # we might write a video to show the data generation attempts
            video_writer = None
            if write_video:
                video_writer = imageio.get_writer(args.video_path, fps=20)

            # data generation statistics
            num_success = 0
            num_failures = 0
            num_attempts = 0
            num_problematic = 0
            ep_lengths = [] # episode lengths for successfully generated data

            # we will keep generating data until @num_trials successes (if @guarantee_success) else @num_trials attempts
            num_trials = mg_config.experiment.generation.num_trials
            guarantee_success = mg_config.experiment.generation.guarantee
            initial_state = None
            n_reuse_initial_state = 0

            while True:

                # generate trajectory
                generated_traj = data_generator.generate(
                    env=env,
                    env_interface=env_interface,
                    select_src_per_subtask=mg_config.experiment.generation.select_src_per_subtask,
                    transform_first_robot_pose=mg_config.experiment.generation.transform_first_robot_pose,
                    interpolate_from_last_target_pose=mg_config.experiment.generation.interpolate_from_last_target_pose,
                    render=args.render,
                    video_writer=video_writer,
                    video_skip=args.video_skip,
                    camera_names=render_image_names,
                    pause_subtask=args.pause_subtask,
                    init_noops=mg_config.experiment.generation.init_noops,
                    initial_state=initial_state,
                )
                
                # check if generated trajectory was successful
                success = bool(generated_traj["success"])
                if mg_config.experiment.generation.filter_joint_limit:
                    success = success and not generated_traj['joint_limit_hit']

                initial_state = generated_traj["initial_state"]

                if success:
                    num_success += 1
                    total_num_success += 1

                    # store successful demonstration
                    ep_lengths.append(generated_traj["actions"].shape[0])
                    total_ep_lengths.append(generated_traj["actions"].shape[0])
                    MG_FileUtils.write_demo_to_hdf5(
                        folder=tmp_dataset_folder_path,
                        env=env,
                        prefix=variant.split(".")[0],
                        initial_state=initial_state,
                        states=generated_traj["states"],
                        observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                        datagen_info=generated_traj["datagen_infos"],
                        actions=generated_traj["actions"],
                        abs_actions=generated_traj["abs_actions"],
                    )
                    initial_state = None
                else:
                    num_failures += 1
                    total_num_failures += 1

                    if n_reuse_initial_state < mg_config.experiment.generation.n_reuse_initial_state:
                        n_reuse_initial_state += 1
                    else:
                        initial_state = None

                    # check if this failure should be kept
                    if mg_config.experiment.generation.keep_failed and \
                        ((mg_config.experiment.max_num_failures is None) or (num_failures <= mg_config.experiment.max_num_failures)):
                        
                        # save failed trajectory in separate folder
                        MG_FileUtils.write_demo_to_hdf5(
                            folder=tmp_dataset_failed_folder_path,
                            env=env,
                            prefix=variant.split(".")[0],
                            initial_state=generated_traj["initial_state"],
                            states=generated_traj["states"],
                            observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                            datagen_info=generated_traj["datagen_infos"],
                            actions=generated_traj["actions"],
                            abs_actions=generated_traj["abs_actions"],
                        )

                num_attempts += 1
                total_num_attempts += 1
                print("")
                print("*" * 50)
                print("trial {} success: {}".format(total_num_attempts, success))
                print("have {} successes out of {} trials so far".format(total_num_success, total_num_attempts))
                print("have {} failures out of {} trials so far".format(total_num_failures, total_num_attempts))
                print("*" * 50)

                # regularly log progress to disk every so often
                if (num_attempts % mg_config.experiment.log_every_n_attempts) == 0:

                    # get summary stats
                    summary_stats = get_important_stats(
                        new_dataset_folder_path=new_dataset_folder_path,
                        num_success=total_num_success,
                        num_failures=total_num_failures,
                        num_attempts=total_num_attempts,
                        num_problematic=num_problematic,
                        start_time=start_time,
                        ep_length_stats=None,
                    )

                    # write stats to disk
                    max_digits = len(str(num_trials * 1000)) + 1 # assume we will never have lower than 0.1% data generation SR
                    json_file_path = os.path.join(json_log_path, "attempt_{}_succ_{}_rate_{}.json".format(
                        str(num_attempts).zfill(max_digits), # pad with leading zeros for ordered list of jsons in directory
                        num_success,
                        np.round((100. * num_success) / num_attempts, 2),
                    ))
                    MG_FileUtils.write_json(json_dic=summary_stats, json_path=json_file_path)

                # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
                check_val = num_success if guarantee_success else num_attempts
                if check_val >= num_trials:
                    break
            
        except Exception as e:
            if args.catch_exceptions:
                print(f"Error generating data for variant {variant}")
                print(f"Error: {e}")
                continue
            else:
                raise e
    if write_video:
        video_writer.close()

    # merge all new created files, unless disabled
    if args.disable_merge:
        print("\nFinished data generation. Skipping merge due to --disable-merge. Writing merge manifest...\n")
        print("To merge later, run: uv run scripts/mimicgen/merge_generated_dataset.py --config "
              f"{args.config}")
        
        return None
    else:
        print("\nFinished data generation. Merging per-episode hdf5s together...\n")
        MG_FileUtils.merge_all_hdf5(
            folder=tmp_dataset_folder_path,
            new_hdf5_path=new_dataset_path,
            delete_folder=True,
            per_demo_env_args=True,
        )
        if mg_config.experiment.generation.keep_failed:
            MG_FileUtils.merge_all_hdf5(
                folder=tmp_dataset_failed_folder_path,
                new_hdf5_path=new_failed_dataset_path,
                delete_folder=True,
                per_demo_env_args=True,
            )

        # get episode length statistics
        ep_length_stats = None
        if len(total_ep_lengths) > 0:
            ep_lengths_arr = np.array(total_ep_lengths)
            ep_length_mean = float(np.mean(ep_lengths_arr))
            ep_length_std = float(np.std(ep_lengths_arr))
            ep_length_max = int(np.max(ep_lengths_arr))
            ep_length_3std = int(np.ceil(ep_length_mean + 3. * ep_length_std))
            ep_length_stats = dict(
                ep_length_mean=ep_length_mean,
                ep_length_std=ep_length_std,
                ep_length_max=ep_length_max,
                ep_length_3std=ep_length_3std,
            )

        stats = get_important_stats(
            new_dataset_folder_path=new_dataset_folder_path,
            num_success=total_num_success,
            num_failures=total_num_failures,
            num_attempts=total_num_attempts,
            num_problematic=total_num_problematic,
            start_time=start_time,
            ep_length_stats=ep_length_stats,
        )
        print("\nStats Summary")
        print(json.dumps(stats, indent=4))

        # maybe render videos
        if mg_config.experiment.render_video:
            if (num_success > 0):
                playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}.mp4".format(new_dataset_folder_name))
                num_render = mg_config.experiment.num_demo_to_render
                print("Rendering successful trajectories...")
                replay_dataset(
                    dataset=new_dataset_path,
                    n=num_render,
                    output_format="video",
                    video_output_dir=playback_video_path,
                    video_fps=20,
                    speedup=5.0,
                    camera_names=render_image_names,
                )
            else:
                print("\n" + "*" * 80)
                print("\nWARNING: skipping dataset video creation since no successes")
                print("\n" + "*" * 80 + "\n")
            if mg_config.experiment.generation.keep_failed:
                if (num_failures > 0):
                    playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}_failed.mp4".format(new_dataset_folder_name))
                    num_render = mg_config.experiment.num_fail_demo_to_render
                    print("Rendering failure trajectories...")
                    replay_dataset(
                        dataset=new_failed_dataset_path,
                        n=num_render,
                        output_format="video",
                        video_output_dir=playback_video_path,
                        video_fps=20,
                        speedup=5.0,
                        camera_names=render_image_names,
                    )
                else:
                    print("\n" + "*" * 80)
                    print("\nWARNING: skipping dataset video creation since no failures")
                    print("\n" + "*" * 80 + "\n")

        # return some summary info
        final_important_stats = get_important_stats(
            new_dataset_folder_path=new_dataset_folder_path,
            num_success=total_num_success,
            num_failures=total_num_failures,
            num_attempts=total_num_attempts,
            num_problematic=total_num_problematic,
            start_time=start_time,
            ep_length_stats=ep_length_stats,
        )

        # write stats to disk
        json_file_path = os.path.join(new_dataset_folder_path, "important_stats.json")
        MG_FileUtils.write_json(json_dic=final_important_stats, json_path=json_file_path)

        return final_important_stats


def main(args: GenerateDatasetArgs):
    # Load config from args
    config = json.load(open(args.config))

    # load config object
    with open(args.config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by MimicGen
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    
    # Create config from external JSON
    mg_config = MGConfig.from_dict(ext_cfg)

    # We assume that the external config specifies all subtasks, so
    # delete any subtasks not in the external config.
    source_subtasks = set(mg_config.task.task_spec.keys())
    new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
    for subtask in (source_subtasks - new_subtasks):
        print("deleting subtask {} in original config".format(subtask))
        del mg_config.task.task_spec[subtask]

    # Override config with command line arguments
    if args.source is not None:
        mg_config.experiment.source.dataset_path = args.source

    if args.folder is not None:
        mg_config.experiment.generation.path = args.folder

    if args.num_demos is not None:
        mg_config.experiment.generation.num_trials = args.num_demos

    if args.seed is not None:
        mg_config.experiment.seed = args.seed

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of generation to test whether this run is likely to crash
        mg_config.experiment.source.n = 3
        mg_config.experiment.generation.guarantee = False
        mg_config.experiment.generation.num_trials = 2

        # send output to a temporary directory
        mg_config.experiment.generation.path = "/tmp/tmp_mimicgen"

    important_stats = generate_dataset(mg_config, args)
    important_stats = json.dumps(important_stats, indent=4)
    print("\nFinal Data Generation Stats")
    print(important_stats)


if __name__ == "__main__":
    args = tyro.cli(GenerateDatasetArgs)
    main(args)
