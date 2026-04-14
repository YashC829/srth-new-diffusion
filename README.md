# srth-new

`srth-new` is a Hydra-based scaffold for surgical robot policy learning.
The most complete workflow in this repository is low-level ACT training.
ROS/dVRK inference support exists, but it is still fairly project-specific.
The high-level policy tree is present, but it should still be treated as an active scaffold rather than a polished workflow.

## Current Status

- Low-level ACT training is the main supported path.
- Hydra configs for training and inference live under [`conf/low_level_policy/`](conf/low_level_policy).
- ROS/dVRK runtime code lives under [`src/srth_new/low_level_policy/inference/`](src/srth_new/low_level_policy/inference).
- The ROS workspace in [`catkin_ws/`](catkin_ws) now vendors its CRTK/dVRK dependencies directly in this repo instead of using git submodules.

## Repository Map

- [`src/srth_new/low_level_policy/`](src/srth_new/low_level_policy): low-level datasets, ACT model code, training loop, and inference utilities
- [`conf/low_level_policy/`](conf/low_level_policy): Hydra configs for low-level training and inference
- [`src/srth_new/high_level_policy/`](src/srth_new/high_level_policy): early high-level policy scaffold
- [`conf/high_level_policy/`](conf/high_level_policy): high-level experiment configs and templates
- [`catkin_ws/`](catkin_ws): catkin workspace for ROS/dVRK dependencies used during runtime inference
- [`saved_runs/`](saved_runs): experiment notes and archived run metadata
- `outputs/`: Hydra run directories with resolved configs, logs, and checkpoints

## Quick Start

If you only want to train policies, the shortest path is:

```bash
conda env create -f environment.yml
conda activate srth-new_
python -m pip install -e .
```

If you also want the pinned Python packages from [`requirements.txt`](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

If you plan to use ROS/dVRK runtime inference, also build the catkin workspace:

```bash
cd catkin_ws
catkin build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
cd ..
```

You will need to source `catkin_ws/devel/setup.bash` again in each new shell before running ROS-based code.

## Environment Setup

### Option 1: Provided Conda Environment

[`environment.yml`](environment.yml) is the simplest setup path.
Note that it currently names the environment `srth-new_`.

```bash
conda env create -f environment.yml
conda activate srth-new_
python -m pip install -e .
```

Optional:

```bash
python -m pip install -r requirements.txt
```

### Option 2: Manual Environment with Mamba

If `conda env create` is too slow, you can build the environment manually:

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

## Low-Level ACT Training

The main training entrypoint is:

```bash
python -m srth_new.low_level_policy.train
```

Hydra loads [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml), which composes:

- a dataloader config
- a policy config
- a custom Hydra logging config

### Before Your First Run

1. Pick a real dataloader config.

   [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml) currently defaults to:

   ```yaml
   defaults:
     - dataloader: test_full_actions_scoring
   ```

   but the repo currently only includes [`conf/low_level_policy/dataloader/example.yaml`](conf/low_level_policy/dataloader/example.yaml).
   Either change the default in `train.yaml` or override it on the command line with `dataloader=example`.

2. Update the dataset settings in [`conf/low_level_policy/dataloader/example.yaml`](conf/low_level_policy/dataloader/example.yaml):

   - `dataset_dir`
   - `tissue_sample_ids_train`
   - `tissue_sample_ids_val`
   - `num_episodes_train`
   - `num_episodes_val`
   - `batch_size`
   - `num_workers`
   - `chunk_size`

3. Review the W&B settings in [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml):

   - `wandb.project`
   - `wandb.entity`
   - `wandb.name`
   - `wandb.mode`

4. If needed, change the Hydra output root in [`conf/low_level_policy/train.yaml`](conf/low_level_policy/train.yaml):

   ```yaml
   hydra:
     run:
       dir: outputs/low_level_policy/train/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

5. If you plan to resume training, set both:

   - `wandb.resume=true`
   - `train.resume_checkpoint=/path/to/checkpoint.ckpt`

   `wandb.id` is also required when resuming from an existing W&B run.

### Common Commands

Use the included example dataloader:

```bash
python -m srth_new.low_level_policy.train dataloader=example
```

Override config values:

```bash
python -m srth_new.low_level_policy.train \
  dataloader=example \
  dataloader.dataset_dir=/data/chole \
  dataloader.batch_size=4 \
  train.num_train_steps=1000
```

Log offline:

```bash
python -m srth_new.low_level_policy.train \
  dataloader=example \
  wandb.mode=offline
```

Disable W&B:

```bash
python -m srth_new.low_level_policy.train \
  dataloader=example \
  wandb.mode=disabled
```

Resume a run:

```bash
python -m srth_new.low_level_policy.train \
  dataloader=example \
  wandb.resume=true \
  wandb.id=<existing_run_id> \
  train.resume_checkpoint=/path/to/checkpoint.ckpt
```

### Training Outputs

Each run writes to a timestamped Hydra directory, typically under:

```text
outputs/low_level_policy/train/<date>/<time>/
```

That directory contains:

- `.hydra/`: resolved Hydra config
- `main.log`: training log
- `checkpoints/`: saved checkpoints

By default, checkpoints are written to:

```text
${hydra:runtime.output_dir}/checkpoints
```

## Inference

There are currently two inference-oriented paths in this repo.

### 1. Hydra-Based Runtime Entry Point

The main entrypoint is:

```bash
python -m srth_new.low_level_policy.run_inference
```

Its config lives in [`conf/low_level_policy/inference.yaml`](conf/low_level_policy/inference.yaml).
At a minimum, set:

- `checkpoint_path`
- `device`
- `prediction_frequency_hz`
- `sleep_rate`

Example:

```bash
python -m srth_new.low_level_policy.run_inference \
  checkpoint_path=/path/to/train_step_1000.ckpt
```

The current inference config is intentionally small, so expect some project-specific wiring around ROS topics, robot state, and deployment details.

### 2. ROS/dVRK Live Runtime

The live runtime implementation is centered around [`src/srth_new/low_level_policy/inference/inference.py`](src/srth_new/low_level_policy/inference/inference.py).
Before using it:

- activate your Conda environment
- source `catkin_ws/devel/setup.bash`
- make sure the required ROS/dVRK services and topics are available
- review any task-specific settings in the `LowLevelPolicy` runtime code

For dVRK connectivity, the repo currently documents these environment variables:

```bash
export ROS_MASTER_URI=http://10.162.34.59:11311
export ROS_IP=10.162.34.58
```

Treat those values as machine-specific examples rather than universal defaults.

## Hydra Quick Reference

- Run with defaults: `python -m srth_new.low_level_policy.train`
- Override one value: `python -m srth_new.low_level_policy.train train.num_train_steps=500`
- Use the included dataloader config: `python -m srth_new.low_level_policy.train dataloader=example`
- Inspect the resolved config: open `.hydra/config.yaml` inside a run directory

## High-Level Policy Status

The high-level policy tree lives under [`src/srth_new/high_level_policy/`](src/srth_new/high_level_policy) and [`conf/high_level_policy/`](conf/high_level_policy), but it is still best understood as a scaffold.
Use it as a starting point for experiments rather than expecting a complete end-to-end workflow.
