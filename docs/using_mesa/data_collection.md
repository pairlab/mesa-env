# Collecting data

MESA supports demonstration collection with Meta Quest or SpaceMouse. Please refer to [setup devices](../getting_started/setup_devices.md) for instructions on setting up your devices before starting data collection.

## Collecting expert demonstrations

Before collecting data, you need to generate a task suite and move it into `mesa/task_suites/bddl_files/`. **Do not forget this step!** See [generating a task](generating_a_task.md) for instructions.

To collect data, use `scripts/data/collect_data.py`:

```bash
uv run scripts/data/collect_data.py \
  --task-suite-name <task_suite_name> \
  --task-id <task_id> \
  --num-demos 10 \
  --device quest \
  --save-datagen-info \
  --control-delta
```

Key arguments:

- `--task-suite-name`: suite name under `mesa/task_suites/bddl_files/`
- `--task-id`: task id in that suite
- `--num-demos` (or `--min-num-demos`): number of demos to collect. By default, we'll collect `len(source_bddl_files)` demos.
- `--device`: `quest` or `spacemouse`
- `--save-datagen-info`: whether to save datagen_info. This is enabled by default, but if you disable it for some reason, note that your data will not be able to be used by MimicGen.

Collected demonstrations are saved under:

```text
data/source/<task_suite_name>/<task_id>/
```

Once you've collected the demonstrations, run the following script to merge them into a single HDF5 file:

```bash
uv run scripts/data/merge_collected_demos.py \
  --demo-dir data/source/<task_suite_name>/<task_id>/ \
  --delete-source-files
```

## Device controls

### Meta Quest

- Right controller: move robot
- Left controller: move second robot (if bimanual)
- A button: save demo
- B button: discard demo
- Ctrl+C: discard current demo and exit
- Use one trigger to engage the robot and the other to open and close the gripper

### SpaceMouse

- Joystick: move robot
- Right button: start/discard demo
- Left button: toggle gripper
- Ctrl+C: discard current demo and exit