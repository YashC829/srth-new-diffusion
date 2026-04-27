# Create conda environment (recommend use mamba)

```bash
mamba create -n srth-new-2 -c conda-forge -c robostack-jazzy \
  python=3.11 \
  ros-jazzy-desktop

conda activate srth-new-2
```

```bash
mamba install -c conda-forge -c robostack-jazzy \
  ros-dev-tools \
  ros-jazzy-action-msgs \
  ros-jazzy-camera-calibration \
  ros-jazzy-camera-info-manager \
  ros-jazzy-camera-calibration-parsers \
  ros-jazzy-controller-manager \
  ros-jazzy-controller-manager-msgs \
  ros-jazzy-cv-bridge \
  ros-jazzy-diagnostic-common-diagnostics \
  ros-jazzy-diagnostic-updater \
  ros-jazzy-image-geometry \
  ros-jazzy-interactive-markers \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-laser-geometry \
  ros-jazzy-message-filters \
  breezy \
  pkg-config \
  gtkmm-3.0 \
  gtk3 \
  gstreamer \
  gst-plugins-base \
  glib \
  jsoncpp \
  glibmm-2.4 \
  atkmm \
  pangomm \
  cairomm \
  sigcpp-2.0
pip install "setuptools<70"
```

# Install, build, and source the dvrk ros2 packages in the ros2 workspace

The below code is from the dvrk documentation [here](https://dvrk.readthedocs.io/main/pages/software/compilation/ros2.html).

```bash
cd ros2_ws/src
vcs import --input https://raw.githubusercontent.com/jhu-saw/vcs/main/ros2-dvrk-main.vcs --recursive --retry 10 --workers 1
cd ..
rm -rf build install log

# we will exclude certain packages that are only for the hardware
colcon build --cmake-args   -DCMAKE_BUILD_TYPE=Release \
  --symlink-install \
  --packages-skip  saw_controllers_ros \
    saw_intuitive_research_kit_example_bilateral_teleop \
    saw_intuitive_research_kit_example_psm_derived \
    saw_robot_io_1394_ros \
    sawRobotIO1394Core \
    sawIntuitiveResearchKitCore \
    sawControllersCore \
    dvrk_robot
source install/setup.bash
```

