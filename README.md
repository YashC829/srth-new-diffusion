# srth-new

`srth-new` is a Hydra-based scaffold for surgical robot policy learning. The repository is currently most complete for low-level ACT training. High-level policy code and some inference workflows are still under active development.

## Repository Layout

- [`src/srth_new/low_level_policy/`](src/srth_new/low_level_policy): low-level training code, datasets, ACT model, and runtime inference utilities
- [`src/srth_new/high_level_policy/`](src/srth_new/high_level_policy): early high-level-policy scaffold and dataset code
- [`conf/low_level_policy/`](conf/low_level_policy): Hydra configs for low-level training and evaluation
- [`conf/high_level_policy/`](conf/high_level_policy): Hydra configs for high-level experiments
- [`catkin_ws/`](catkin_ws): catkin workspace for ROS/dVRK dependencies used by runtime inference
- `outputs/`: Hydra logs, resolved configs, and checkpoints written during runs

## Setup

For training-only workflows, the provided Conda environment is usually enough. For ROS/dVRK inference, you will also need the catkin workspace build described below.

### Option 1: Use the provided Conda environment

The quickest setup path is [`environment.yml`](environment.yml). Note that the file currently names the environment `srth-new_`.

```bash
conda env create -f environment.yml
conda activate srth-new_
python -m pip install -e .
```

If you also want the pinned Python dependencies from [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

### Option 2: Create the environment from scratch with mamba

If `conda env create` is too slow, install `mamba` first and use it for the rest of the setup:

```bash
conda install -n base -c conda-forge mamba
```

```bash
mamba create -n srth-new -c conda-forge -c robostack-noetic \
  python=3.11 \
  ros-noetic-desktop

conda activate srth-new
mamba config --env --add channels robostack-noetic
```

```bash
mamba install -c conda-forge \
  ros-dev-tools \
  ros-noetic-actionlib \
  ros-noetic-camera-calibration \
  ros-noetic-camera-calibration-parsers \
  ros-noetic-catkin \
  ros-noetic-controller-manager \
  ros-noetic-controller-manager-msgs \
  ros-noetic-cv-bridge \
  ros-noetic-diagnostic-analysis \
  ros-noetic-diagnostic-common-diagnostics \
  ros-noetic-diagnostic-updater \
  ros-noetic-dynamic-reconfigure \
  ros-noetic-gazebo-plugins \
  ros-noetic-gazebo-ros \
  ros-noetic-gencpp \
  ros-noetic-geneus \
  ros-noetic-genlisp \
  ros-noetic-gennodejs \
  ros-noetic-genmsg \
  ros-noetic-genpy \
  ros-noetic-image-geometry \
  ros-noetic-interactive-markers \
  ros-noetic-joint-state-publisher \
  ros-noetic-joint-state-publisher-gui \
  ros-noetic-laser-geometry \
  ros-noetic-message-filters \
  breezy \
  catkin_tools
```

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Build the catkin workspace

If you plan to use ROS/dVRK runtime inference, build and source the workspace in [`catkin_ws/`](catkin_ws):

```bash
cd catkin_ws
catkin build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
cd ..
```

Source `catkin_ws/devel/setup.bash` again in each new shell before launching ROS-based inference.

## Low-Level ACT Training

The main training entrypoint is:

```bash
python -m srth_new.low_level_policy.train
```

Hydra loads [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml), which composes:

- a dataloader config from [`conf/low_level_policy/dataloader/`](conf/low_level_policy/dataloader)
- a policy config from [`conf/low_level_policy/policy/`](conf/low_level_policy/policy)
- the custom Hydra logging config

### Before you run

1. Update the dataset path in [`conf/low_level_policy/dataloader/example.yaml`](conf/low_level_policy/dataloader/example.yaml):

```yaml
dataset_dir: /path/to/your/dataset
```

2. Review the split and dataloader settings in [`conf/low_level_policy/dataloader/example.yaml`](conf/low_level_policy/dataloader/example.yaml):

- `tissue_sample_ids_train` and `tissue_sample_ids_val`
- `num_episodes_train` and `num_episodes_val`
- `batch_size`, `num_workers`, and `chunk_size`

3. Update the default output directory in [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml) if your checkout is not at this exact path:

```yaml
hydra:
  run:
    dir: /path/to/your/repo/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

4. Set the Weights & Biases fields in [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml) before your first run:

```yaml
wandb:
  project: your_project_name
  entity: your_team_or_username
  name: experiment_name
  resume: false
  id: null
  mode: online
```

When `wandb.resume=true`, `wandb.id` is required and the original Hydra config is restored from W&B before training continues.

### Common commands

```bash
# Default run
python -m srth_new.low_level_policy.train

# Override config values
python -m srth_new.low_level_policy.train \
  dataloader.dataset_dir=/data/chole \
  dataloader.batch_size=4 \
  train.num_train_steps=1000

# Log offline
python -m srth_new.low_level_policy.train wandb.mode=offline

# Disable W&B
python -m srth_new.low_level_policy.train wandb.mode=disabled

# Resume an existing run
python -m srth_new.low_level_policy.train \
  wandb.resume=true \
  wandb.id=<existing_run_id>
```

### Outputs

Each training run writes to a timestamped Hydra directory, typically under `outputs/`, containing:

- `.hydra/`: the resolved Hydra config
- `main.log`: the training log
- `checkpoints/`: saved model checkpoints

By default, checkpoints are written to:

```text
${hydra:runtime.output_dir}/checkpoints
```

## Inference

### ROS/dVRK runtime inference

The live inference implementation lives in [`src/srth_new/low_level_policy/inference/inference.py`](src/srth_new/low_level_policy/inference/inference.py). Before using it:

- activate your Conda environment
- source `catkin_ws/devel/setup.bash`
- make sure the required ROS/dVRK dependencies and topics are available
- review the task-specific hardcoded settings in the `LowLevelPolicy` class

This path is intended for project-specific robot execution and is not yet a plug-and-play CLI workflow.

### Hydra-based checkpoint evaluation

The evaluation config lives in [`conf/low_level_policy/inference.yaml`](conf/low_level_policy/inference.yaml). At a minimum, set:

- `checkpoint_path`
- `stats_path` if you need to load dataset stats separately
- `output_path` if you want to save predictions

The current evaluation path is still a scaffold. [`src/srth_new/low_level_policy/run_inference.py`](src/srth_new/low_level_policy/run_inference.py) and [`conf/low_level_policy/inference.yaml`](conf/low_level_policy/inference.yaml) are good starting points, but expect some project-specific wiring before using them end to end.

## Hydra Quick Reference

- Run with defaults: `python -m srth_new.low_level_policy.train`
- Override one value: `python -m srth_new.low_level_policy.train train.num_train_steps=500`
- Swap a config group member: `python -m srth_new.low_level_policy.train dataloader=example`
- Inspect the resolved config: check `.hydra/config.yaml` inside a run directory

## High-Level Policy Status

The high-level policy config tree lives under [`conf/high_level_policy/`](conf/high_level_policy), but the corresponding training and inference entrypoints are still scaffolds. Treat those configs as templates rather than a fully wired workflow.
