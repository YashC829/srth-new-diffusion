# srth-new

`srth-new` is a Hydra-driven training scaffold for surgical robot policy learning. The repository currently has the most complete support for low-level ACT policy training, plus early high-level-policy configuration files that are still being built out.

## Repository Layout

- `src/srth_new/low_level_policy/`: low-level policy training, dataset utilities, and ACT model code
- `src/srth_new/high_level_policy/`: high-level-policy scaffold and dataset code
- `conf/low_level_policy/`: Hydra configs for low-level training and inference
- `conf/high_level_policy/`: Hydra configs for high-level experiments
- `outputs/`: Hydra run outputs, logs, and checkpoints

## Setup

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate srth-new
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

## Low-Level Policy Training

The main low-level training entrypoint is:

```bash
python -m srth_new.low_level_policy.train
```

Hydra loads [`conf/low_level_policy/config.yaml`](/home/grayson/surpass/srth-new/conf/low_level_policy/config.yaml), which in turn composes:

- a dataloader config from `conf/low_level_policy/dataloader/`
- a policy config from `conf/low_level_policy/policy/`
- the custom Hydra logging config

### Before You Run

Update the dataset path in [`conf/low_level_policy/dataloader/example.yaml`](/home/grayson/surpass/srth-new/conf/low_level_policy/dataloader/example.yaml):

```yaml
dataset_dir: /path/to/your/dataset
```

Also, update the repo root path for the location of your run `output` folder:

```yaml
hydra:
  run:
    # Each run gets its own timestamped directory with logs, configs, and checkpoints.
    ### UPDATE THE BELOW LINE ###
    dir: <repo_dir_root_path>/outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

You will usually also want to review:

- `tissue_sample_ids_train` / `tissue_sample_ids_val`
- `num_episodes_train` / `num_episodes_val`
- `batch_size`, `num_workers`, and `chunk_size`
- `wandb.*` fields in [`conf/low_level_policy/config.yaml`](/home/grayson/surpass/srth-new/conf/low_level_policy/config.yaml)

### Weights & Biases

Low-level training initializes Weights & Biases from the `wandb` block in [`conf/low_level_policy/config.yaml`](/home/grayson/surpass/srth-new/conf/low_level_policy/config.yaml):

```yaml
wandb:
  project: your_project_name
  entity: your_team_or_username
  name: experiment_name
  resume: false
  id: null
  mode: online
```

What each field does:

- `project`: the W&B project where runs will be logged
- `entity`: your W&B username or team/org name
- `name`: the display name shown in the W&B UI
- `resume`: if `true`, restore the prior Hydra config and resume the W&B run
- `id`: optional for fresh runs, required when `resume=true`
- `mode`: `online`, `offline`, or `disabled`

Common workflows:

- Log normally:

```bash
python -m srth_new.low_level_policy.train \
  wandb.project=my-project \
  wandb.entity=my-team \
  wandb.name=baseline-run
```

- Log offline and sync later:

```bash
python -m srth_new.low_level_policy.train wandb.mode=offline
```

- Disable W&B entirely:

```bash
python -m srth_new.low_level_policy.train wandb.mode=disabled
```

- Resume an existing run:

```bash
python -m srth_new.low_level_policy.train \
  wandb.resume=true \
  wandb.id=<existing_run_id>
```

When resuming, the training code reloads the saved Hydra config from W&B, so the resumed run uses the original experiment configuration instead of any accidental local drift.

### Common Overrides

Hydra lets you override config values from the command line:

```bash
python -m srth_new.low_level_policy.train \
  dataloader.dataset_dir=/data/chole \
  dataloader.batch_size=4 \
  train.num_train_steps=1000 \
  wandb.mode=offline
```

### Outputs

Each training run writes to a timestamped Hydra directory under `outputs/`, including:

- `.hydra/`: the resolved Hydra config for the run
- `main.log`: the training log
- `checkpoints/`: saved model checkpoints

By default, low-level checkpoints are written to:

```text
${hydra:runtime.output_dir}/checkpoints
```

## Low-Level Inference

There is also a low-level inference script:

```bash
python -m srth_new.low_level_policy.inference
```

Before running it, you must set at least:

- `checkpoint_path`
- the task-specific dataset fields expected by [`conf/low_level_policy/inference.yaml`](/home/grayson/surpass/srth-new/conf/low_level_policy/inference.yaml)

The current inference config is still a template and expects additional task configuration to be filled in for your dataset.

## High-Level Policy Status

The high-level-policy config tree is present under [`conf/high_level_policy/`](/home/grayson/surpass/srth-new/conf/high_level_policy), but the corresponding training and inference entrypoints are still scaffolds. Treat those configs as reference templates rather than a fully wired training workflow.

## Hydra Config Guide

Hydra composes configs from small building blocks. In this repo, the most common pattern is:

```yaml
defaults:
  - dataset_or_dataloader: example
  - policy: act
  - _self_
```

That means:

1. Load a base config.
2. Insert one dataset or dataloader sub-config.
3. Insert one policy sub-config.
4. Apply values from the current file last.

Useful Hydra patterns:

- Run with defaults: `python -m srth_new.low_level_policy.train`
- Override one value: `python -m srth_new.low_level_policy.train train.num_train_steps=500`
- Swap a config group member: `python -m srth_new.low_level_policy.train dataloader=example`
- See the fully resolved config: check `.hydra/config.yaml` inside a run directory

## Notes

- `README.md` and the example configs now include inline comments to clarify the most important fields.
- The repository does not currently contain a top-level `requirements.txt`; use `environment.yml` or `pyproject.toml` as the source of truth for dependencies.
