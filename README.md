# srth-new

Minimal ACT training and inference scaffold with Hydra configs.

This subtree keeps only the pieces needed to:

- train an `ACTPolicy` from episodic HDF5 data
- save checkpoints and dataset normalization stats
- load a trained checkpoint and run batched inference

The expected dataset format matches the ACT-style episodes already used in this repository:

- `episode_000.hdf5`, `episode_001.hdf5`, ...
- `/observations/qpos`
- `/observations/images/<camera_name>`
- `/action`
- optional file attribute: `compress`

Run training:

```bash
conda create -n srth-new -c conda-forge -c robostack-noetic \
    python=3.11 \
    ros-noetic-desktop
conda activate srth-new
conda config --env --add channels robostack-noetic
conda install -c conda-forge ros-dev-tools \
    ros-noetic-actionlib \
    ros-noetic-camera-calibration \
    ros-noetic-camera-calibration-parsers \
    ros-noetic-catkin \
    ros-noetic-controller-manager \
    ros-noetic-controller-manager-msgs \
    ros-noetic-cv-bridge \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-genmsg \
    ros-noetic-genpy \
    ros-noetic-gencpp \
    ros-noetic-geneus \
    ros-noetic-genlisp \
    ros-noetic-gennodejs \
    ros-noetic-diagnostic-updater \
    ros-noetic-diagnostic-analysis \
    ros-noetic-diagnostic-common-diagnostics \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-plugins \
    ros-noetic-image-geometry \
    ros-noetic-laser-geometry \
    ros-noetic-message-filters \
    ros-noetic-interactive-markers \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    breezy
```

```bash
cd srth-new
pip install -e .
python -m srth_new.train task.dataset_dir=/path/to/dataset task.camera_names='[cam_high]'
```

Run inference:

```bash
cd srth-new
python -m srth_new.inference \
  task.dataset_dir=/path/to/dataset \
  task.camera_names='[cam_high]' \
  checkpoint_path=/path/to/policy_best.ckpt
```

If you do not want to install the package, use `PYTHONPATH=src` before `python -m ...`.

Useful overrides:

- `task.state_dim=20`
- `task.action_dim=20`
- `task.chunk_size=60`
- `policy.backbone=efficientnet_b3`
- `policy.pretrained_backbone=true`
- `batch_size=8`
- `num_epochs=1000`
- `checkpoint_dir=/path/to/output`

If `stats_path` is omitted for inference, the script looks for `dataset_stats.pkl` next to the checkpoint.
