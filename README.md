# srth-new

This is a repository created by Grayson Byrd and Jacob Delgado Lopez of the ARCADE lab at The Johns Hopkins University. It builds off of the original SRTH work, also conducted at JHU.

# Environment Setup
This environment uses robostack and ros2 for inference. A lot of the ros2 features are not required for inference. We recommend two different environments. One for training and one for inference. The training environment will be more portable and can be run on a cluster somewhere easily. The inference environment will contain all of the ros2 features and is more difficult to install and handle dependencies. This should only be installed on your inference machine.

## Training Environment

Create conda environment and install repository dependencies:

```bash
conda create -n srth-new python=3.10
conda activate srth-new
pip install -r requirements.txt
pip install -e .
```

Install LeRobot. LeRobot can be a bit annoying to install and get the dependencies correct.. 
```bash
pip install lerobot==0.4.4
```

Sometimes pip will give you a warning that the dependencies are not met, but you can ignore it. Other times, it will be an issue. I recommend installing the below pinned versions of some usually tricky dependencies and choosing your agent of choice and use it to debug the environment. Good luck!

```bash
pip install transformers==4.57.6 huggingface_hub==0.35.3
```

## Install submodules

From parent repo root:
```bash
git submodule update --init --recursive
```

Download the [EndoSynth](https://github.com/TouchSurgery/EndoSynth) checkpoint if you wish to use depth images with the model. Run the below from the parent repository root directory.
```bash
mkdir src/srth_new/general/third_party/EndoSynth/checkpoints/
wget -O src/srth_new/general/third_party/EndoSynth/checkpoints/dav2-f.pth https://digitalsurgery-public.s3.eu-west-1.amazonaws.com/EndoSynth/weights/dav2-f.pth
```

## Inference Environment

The major difference with the inference environment is installing ros2. This is done with robostack. Run the following to create your initial conda environment:

```bash
# Create a ros-jazzy desktop environment
conda create -n srth-new-ros2 -c conda-forge -c robostack-jazzy ros-jazzy-desktop
# Activate the environment
conda activate srth-new-ros2
# Add the robostack channel to the environemnt
conda config --env --add channels robostack-jazzy
```

Next, follow all of the steps from the **Training Environment** section to install the rest of the packages.

### Install dVRK ros2 packages
To be able to communciate with the dVRK, you must download, build, and source the appropriate ros2 directory. Run the following from the root dir of the parent repository:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
vcs import --input https://raw.githubusercontent.com/jhu-saw/vcs/main/ros2-dvrk-main.vcs --recursive --retry 10 --workers 1
```

This will create a `ros2_ws` directory and will clone various ros2 packages into the `ros2_ws/src` directory. Build the packages by running the following in the parent root directory: 

```bash
cd ~/ros2_ws
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install
```

Now, any time you wish to communicate with the robot, you must activate the `srth-new-ros2` conda environment and source the `ros2_ws` with:

```bash
source ~/ros2_ws/install/setup.bash
```

## ros2 Networking

To send/recieve ros2 topics to the dVRK, you must be connected to the same internet network as the dVRK. Both computers must set the same `ROS_DOMAIN_ID` environment variable:

```bash
export ROS_DOMAIN_ID="<your_domain_id_here>"
```

Everything should be set up!

# Creating a Dataset

We use [LeRobot](https://github.com/huggingface/lerobot) for our dataset. We collect a dataset in a specific format using code that is saved to the dVRK. Ask Jacob Delgado Lopez for that code.

## LeRobot dataset conversion

Once you have your raw data, you must create a LeRobot dataset from the raw data by running the following:

```bash
python src/srth-new/low_level_policy/dataset/convert_raw_data_to_lerobot_format.py \
  --source-dir <absolute_path_to_raw_dataset_root_dir> \
  --repo-id <what_you_want_your_dataset_name_to_be **must be unique**>
```

You can look at the python file `parse_args` function to see some other arguments that may be helpful.

# Running the Code

We use [Hydra](https://hydra.cc/) to create configuration files and manage runs. Our configuration files can be seen in `conf/`.

## Run Training with Default Configuration

```bash
python src/srth-new/low_level_policy/train.py
```

## Run Inference

```bash
python src/srth-new/low_level_policy/run_inference.py \ 
  checkpoint_path=<path_to_checkpoint_file>
```