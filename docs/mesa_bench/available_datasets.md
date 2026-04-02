# Available Datasets

We provide several versions of our dataset. First, we provide two versions of the canonical MESA-70 train set:

- [MESA-70-hdf5](https://huggingface.co/datasets/albertwilcox/mesa-70-hdf5): The canonical MESA-70 train set in HDF5 format. This is similar to robomimic-style datasets, and should work as a drop-in replacement for settings which use this format for loading data but not for instantiating environments, due to the exceptions described below.
- [MESA-70-lerobot](https://huggingface.co/datasets/albertwilcox/mesa-70-lerobot): The canonical MESA-70 train set in LeRobot format with 224x224 RGB images for the left and right shoulder cameras and wrist camera.

We also provide two versions of the larger dataset, most of which are not included in the canonical MESA-70 train set or evaluation sets.

- [MESA-all-hdf5](https://huggingface.co/datasets/albertwilcox/mesa-all-hdf5): The full set of tasks for which we generated demonstrations in HDF5 format with *only low-dimensional observations*. This is for those hoping to download the data and process it into some format for their own purposes. We include both the training subset in `train/` and data overlapping with some, but not all, held-out tasks in `test/`. To make use of this data, you'll need to process it to add observations as described in the [data processing section](../using_mesa/processing_data.md).
- [MESA-all-train-lerobot](https://huggingface.co/datasets/albertwilcox/mesa-all-train-lerobot): The full set of tasks for which we generated demonstrations, with the exception of tasks which would overlap with our held out task sets, in LeRobot format with 224x224 RGB images for the left and right shoulder cameras and wrist camera.

We empirically found naive cotraining on this data to not provide much benefit, but hope they can be useful for future researchers.

```{warning}
The MESA-all datasets contain some tasks which are present in the held-out evaluation sets. If you plan to use them, you should refer to `mesa/task_suites/task_sets.py` for the list of tasks in MESA-aux, which does not include held-out tasks.
```

## HDF5 Format

Our HDF5 format is mostly similar to robomimic-style datasets with the following exceptions:

 - We provide several action spaces. `actions` contains delta end-effector poses, `abs_actions` contains absolute end-effector poses, and `actions_joint_pos` contains absolute joint positions.
 - Robomimic-style datasets typically have an `env_args` dictionary stored as an attribute of the `data` group. Since each demonstration corresponds to a unique BDDL file, and therefore a unique set of `env_args`, the `env_args` dictionary is now a dictionary mapping demonstration keys to per-demonstration `env_args`.

An example of the structure of a dataset is shown below:
```
|__data
|   |__demo_0
|   |   |__actions
|   |   |__abs_actions
|   |   |__actions_joint_pos
|   |   |__obs
|   |   |   |__leftshoulder_image
|   |   |   |__rightshoulder_image
|   |   |   |__robot0_eye_in_hand_image
|   |   |   |__robot0_eef_pos
|   |   |   |__<several other low-dimensional observations>
|   |   |__states
```