import json
from copy import deepcopy
from typing import Literal

import h5py
import numpy as np
import robosuite
from tqdm import tqdm

import mesa.utils.tensor_utils as TensorUtils
from mesa.sim.envs import BDDLBaseDomain
from mesa.utils.data_saver import (
    DummySaver,
    EpisodeSaver,
    HDF5Saver,
    LeRobotSaver,
    VideoSaver,
)


def extract_trajectory(
    env: BDDLBaseDomain,
    initial_state: dict,
    states: np.ndarray,
    actions: np.ndarray,
    abs_actions: np.ndarray | None = None,
    render: bool = False,
    remove_done_frames: bool = False,
) -> dict:
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        render (bool): whether to render the environment
    """
    # assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    obs = env.reset_to(initial_state)
    if render:
        # render to screen
        env.render()

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
    )
    if abs_actions is not None:
        traj["abs_actions"] = np.array(abs_actions)

    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})

        if render:
            # render to screen
            env.render()

        # infer reward signal
        r = env.reward()

        # infer done signal
        done = env._check_success()

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

        if remove_done_frames and done:
            break

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


def replay_dataset(
    dataset: str,
    n: int | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    camera_names: list[str] = ["leftshoulder", "rightshoulder", "robot0_eye_in_hand"],
    camera_height: int = 224,
    camera_width: int = 224,
    return_2d_bboxes: bool = False,
    return_3d_bboxes: bool = False,
    render: bool = False,
    remove_done_frames: bool = False,
    output_format: Literal["hdf5", "lerobot", "video"] | None = None,
    hdf5_output_file: str | None = None,
    hdf5_compress: bool = False,
    lerobot_root_dir: str | None = None,
    lerobot_repo_id: str | None = None,
    lerobot_fps: int = 15,
    lerobot_visual_mode: Literal["image", "video"] = "image",
    video_output_dir: str | None = None,
    video_fps: int = 15,
    speedup: float = 1.0,
) -> None:
    """
    Convert a dataset of simulator states/actions into observations using the environment.
    """
    # open source hdf5 file
    f_src = h5py.File(dataset, "r")
    demos = list(f_src["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if n is not None:
        assert end_idx is None, "n and start_idx/end_idx cannot be provided at the same time"
        end_idx = start_idx + n
    elif end_idx is None:
        end_idx = len(demos)
    demos = demos[start_idx:end_idx]

    # create environment with updated env_meta
    env_meta = json.loads(f_src["data"].attrs["env_args"])
    multi_env = 'env_kwargs' not in env_meta and 'demo_0' in env_meta
    instructions = []
    
    if multi_env:
        for demo in demos:
            demo_env_meta = env_meta[demo]
            demo_env_meta["env_kwargs"]["camera_heights"] = camera_height
            demo_env_meta["env_kwargs"]["camera_widths"] = camera_width
            demo_env_meta["env_kwargs"]["camera_names"] = camera_names
            # Enable 2D and 3D bboxes if requested
            demo_env_meta["env_kwargs"]["return_2d_bboxes"] = bool(return_2d_bboxes)
            demo_env_meta["env_kwargs"]["return_3d_bboxes"] = bool(return_3d_bboxes)
            if render:
                demo_env_meta["env_kwargs"]["has_renderer"] = True

            instructions.append(' '.join(demo_env_meta["env_kwargs"]["parsed_problem"]["language_instruction"]))
        env_meta_to_save = deepcopy(env_meta)
    else:
        env_meta["env_kwargs"]["camera_heights"] = camera_height
        env_meta["env_kwargs"]["camera_widths"] = camera_width
        env_meta["env_kwargs"]["camera_names"] = camera_names
        # Enable 2D and 3D bboxes if requested
        env_meta["env_kwargs"]["return_2d_bboxes"] = bool(return_2d_bboxes)
        env_meta["env_kwargs"]["return_3d_bboxes"] = bool(return_3d_bboxes)
        instructions = [' '.join(env_meta["env_kwargs"]["parsed_problem"]["language_instruction"])] * len(demos)
        env_meta_to_save = deepcopy(env_meta)
        if render:
            env_meta["env_kwargs"]["has_renderer"] = True
        env = robosuite.make(
            env_meta["env_name"],
            **env_meta["env_kwargs"],
        )
    env_attrs_to_save = deepcopy(dict(f_src["data"].attrs))
    env_attrs_to_save["env_args"] = env_meta_to_save

    saver: EpisodeSaver
    if output_format == "hdf5":
        saver = HDF5Saver(
            hdf5_output_file=hdf5_output_file,
            data_attrs=env_attrs_to_save,
            compress=hdf5_compress,
        )
    elif output_format == "lerobot":
        saver = LeRobotSaver(
            root_dir=lerobot_root_dir,
            repo_id=lerobot_repo_id,
            fps=lerobot_fps,
            visual_mode=lerobot_visual_mode,
        )
    elif output_format == "video":
        saver = VideoSaver(
            video_dir=video_output_dir,
            fps=video_fps,
            speedup=speedup,
        )
    elif output_format is None:
        saver = DummySaver()
    else:
        raise ValueError(
            f"Unknown data_format: {output_format!r} (expected 'hdf5', 'lerobot', 'video', or None)"
        )

    with saver:
        dest_desc = saver.destination
        existing_demos = saver.existing_demos
        if len(existing_demos) > 0:
            print(f"Found {len(existing_demos)} existing episodes in {dest_desc}. Resuming...")

        # saving trajectories (skip any already processed demos)
        demos_to_process = [d for d in demos if d not in existing_demos]
        for ind in tqdm(range(len(demos_to_process))):
            if multi_env:
                demo_env_meta = env_meta_to_save[demos_to_process[ind]]
                env = robosuite.make(
                    demo_env_meta["env_name"],
                    **demo_env_meta["env_kwargs"],
                )

            ep = demos_to_process[ind]

            # prepare initial state to reload from
            states = f_src["data/{}/states".format(ep)][()]
            initial_state = dict(
                states=states[0], model=f_src[f"data/{ep}"].attrs["model_file"]
            )

            # get actions from data
            actions = f_src["data/{}/actions".format(ep)][()]
            abs_actions = f_src["data/{}/abs_actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                abs_actions=abs_actions,
                render=render,
                remove_done_frames=remove_done_frames,
            )

            actions_joint_pos = traj['obs']['robot0_joint_pos'][1:]
            actions_joint_pos = np.concatenate([actions_joint_pos, actions_joint_pos[-1:]], axis=0)
            actions_gripper = actions[:len(actions_joint_pos), -1:]
            actions_joint_pos = np.concatenate([actions_joint_pos, actions_gripper], axis=1)
            traj['actions_joint_pos'] = actions_joint_pos

            saver.write_episode(
                episode_key=ep,
                trajectory=traj,
                initial_state=initial_state,
                task=instructions[ind],
            )

        print(f"Wrote {len(demos_to_process)} new episodes to {dest_desc}")

    # close source hdf5
    f_src.close()

