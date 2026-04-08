import os
import time
from typing import Tuple

import cv2
from cv_bridge import CvBridge
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import rotations, batch_rotations, transformations, trajectories
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import normalize
import torch

from src.srth_new.general.utils import processing
from srth_new.low_level_policy.models.act_model import ACTPolicy
from src.srth_new.general.utils.lang_encoding import initialize_model_and_tokenizer, encode_text

# From ros packages
import crtk
from dvrk_scripts.dvrk_control import example_application
import rospy
from rostopics import ros_topics
from std_msgs.msg import Bool

import logging
log = logging.getLogger(__name__)


class LowLevelPolicy:

    ## ----------------- initializations ----------------
    def __init__(
            self,
            policy: ACTPolicy,
            dataset_stats: processing.DatasetStats,
        ):
        self.policy = policy
        self.dataset_stats = dataset_stats
        
        self.initialize_hardcoded_parameters()
        self.initialize_ros()
        # TODO: The below distilbert is hardcoded, but should be loaded from the

        self.tokenizer, self.language_model = initialize_model_and_tokenizer("distilbert")

        # TODO: This is hardcoded for now. Later, this should be predicted by the
        # high level policy
        self.command = "1_grasp"
        
    def initialize_hardcoded_parameters(self):

        self.action_mode = "hybrid_relative" # TODO: This is hardcoded but should be a policy member variable

        self.num_inferences = 4000
        self.action_execution_horizon = 30
        
        self.sleep_rate = 0.18
        self.language_encoder = "distilbert"
        self.max_timesteps = 400 
        self.state_dim = 16
        self.iter = 0
        self.sketch_img = None
        self.sketch_inferencing = False
        self.pause = False
        self.fps = 30
        self.use_contour = None
        self.correction = None
        self.user_correction = None
        self.is_correction = False
        self.user_correction_start_t = None
        self.use_preprogrammed_correction = False
        self.cropped = False

        # placeholder variables for callbacks
            
    def initialize_ros(self):

        # TODO: Not sure what the below does, so not going to remove for now. In
        # the future we should prune and organize this...

        self.rt = ros_topics()
        self.ral = crtk.ral('dvrk_arm_test')
        self.bridge = CvBridge()
        self.psm1_app = example_application(self.ral, "PSM1", 1)
        self.psm2_app = example_application(self.ral, "PSM2", 1)
        self.pause_sub = rospy.Subscriber("/pause_robot", Bool, self.pause_robot_callback, queue_size=10)

    ## --------------------- callbacks -----------------------
    
    def pause_robot_callback(self, msg):
        self.pause = msg.data
        
        if self.pause:
            print("Robot paused. Waiting for the robot to be unpaused...")
        else:
            print("Robot unpaused. Resuming the low level policy...")
        
        
    def contour_image_callback(self, msg):
        self.contour_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        
    def use_contour_image_callback(self, msg):
        self.use_contour = msg.data
        print("use contour image: ", self.use_contour)
        
    def robot_direction_callback(self, msg):
        self.correction = msg.data
        if self.user_correction is not None:
            if time.time() - self.user_correction_start_t < 5:
                self.correction = self.user_correction
            else:
                self.user_correction = None
                self.is_correction = False  # Reset is_correction when user correction expires

    def user_correction_callback(self, msg):
        self.user_correction = msg.data
        self.user_correction_start_t = time.time()
        self.correction = self.user_correction
        self.is_correction = True  # Set the correction flag immediately when user issues a correction
        print("User correction issued: ", self.correction)

    def is_correction_callback(self, msg):
        if self.user_correction is not None and time.time() - self.user_correction_start_t < 5:
            self.is_correction = True
        else:
            self.is_correction = msg.data
            if not self.is_correction:
                self.user_correction = None  # Clear user_correction if it's not active

        # print("Is correction active: ", self.is_correction)
    
    def mid_level_sketch_callback(self, msg):
        self.sketch_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        print("mid level sketch received")
        
        
    def use_preprogrammed_correction_callback(self, msg):
        self.use_preprogrammed_correction = msg.data
        print("use preprogrammed correction: ", self.use_preprogrammed_correction)    
    
    
    def get_image_dvrk(self):
        self.iter += 1

        def decode_ros_img(img_data, img_shape: Tuple[int, int]):
            img = np.fromstring(img_data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, img_shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = rearrange(img, 'h w c -> c h w')
        
        self.left_img = decode_ros_img(self.rt.usb_image_left.data, img_shape=(480, 360))
        self.psm2_img = decode_ros_img(self.rt.endo_cam_psm2, img_shape=(480, 360))
        self.psm1_img = decode_ros_img(self.rt.endo_cam_psm1)

        curr_image = np.stack([self.left_img, self.psm2_img, self.psm1_img], axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        
        return curr_image
    
    def plot_actions(self, qpos_psm1, qpos_psm2, actions_psm1, actions_psm2):
        factor = 1000
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(actions_psm1[:, 0] * factor, actions_psm1[:, 1]* factor, actions_psm1[:, 2]* factor, c ='r')
        ax.scatter(actions_psm2[:, 0]*factor, actions_psm2[:, 1]*factor, actions_psm2[:, 2]*factor, c ='r', label = 'Generated trajectory')
        ax.scatter(qpos_psm1[0]* factor, qpos_psm1[1]* factor, qpos_psm1[2]* factor, c = 'g')
        ax.scatter(qpos_psm2[0]*factor, qpos_psm2[1]*factor, qpos_psm2[2]*factor, c = 'b', label = 'Current end-effector position')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        n_bins = 7
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_bins))
        ax.zaxis.set_major_locator(plt.MaxNLocator(n_bins))
        plt.show()
        # input("Press Enter to continue...")
        # assert(False)

    def execute_actions(self, actions_psm1, actions_psm2):

        for jj in range(self.action_execution_horizon):
            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            time.sleep(self.sleep_rate)
                

    ## --------------------- main loop -----------------------

    def run(self):
        log.info("-------------starting low level policy inference------------------\n")
        time.sleep(1)
        with torch.inference_mode():
            t = 0
            
            while t < self.num_inferences:
                try:
                    if rospy.is_shutdown():
                        print("ROS shutdown signal received. Exiting...")
                        break
        
                    command_embedding = torch.tensor(encode_text(
                        self.command, self.language_encoder, self.tokenizer, self.language_model
                    )).cuda()
                    
                    # TODO: Currently, the qpos is just sent in as a zero, may want
                    # to send the actual qpos
                    qpos_zero = torch.zeros(1, 20).float().cuda()
                    
                    # use this if testing with real endoscope image
                    curr_image = self.get_image_dvrk()

                    action = self.policy(qpos_zero, curr_image, command_embedding=command_embedding).cpu().numpy().squeeze()
                    # TODO: The norm scheme is hardcoded but should be a policy member variable
                    
                    if self.policy.norm_scheme == "std":
                        action = processing.unnormalize_positions_only_std(
                            action, self.dataset_stats.mean, self.dataset_stats.std
                        )
                    else:
                        raise NotImplementedError()

                    qpos_psm1 = np.array((self.rt.psm1_pose.position.x, self.rt.psm1_pose.position.y, self.rt.psm1_pose.position.z,
                                        self.rt.psm1_pose.orientation.x, self.rt.psm1_pose.orientation.y, self.rt.psm1_pose.orientation.z, self.rt.psm1_pose.orientation.w,
                                        self.rt.psm1_jaw))

                    qpos_psm2 = np.array((self.rt.psm2_pose.position.x, self.rt.psm2_pose.position.y, self.rt.psm2_pose.position.z,
                                        self.rt.psm2_pose.orientation.x, self.rt.psm2_pose.orientation.y, self.rt.psm2_pose.orientation.z, self.rt.psm2_pose.orientation.w,
                                        self.rt.psm2_jaw))

                    if self.action_mode == 'hybrid_relative':

                        actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                        actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
                        actions_psm1 = processing.convert_delta_6d_to_taskspace_quat(action[:, 0:10], actions_psm1, qpos_psm1)
                        actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                        
                        actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                        actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
                        actions_psm2 = processing.convert_delta_6d_to_taskspace_quat(action[:, 10:], actions_psm2, qpos_psm2)
                        actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  

                    if self.action_mode == 'relative_endoscope':
                        actions_psm1 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                        actions_psm1[:, 0:3] = qpos_psm1[0:3] + action[:, 0:3] # convert to current translation
                        actions_psm1 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(action[:, 0:10], actions_psm1, qpos_psm1)
                        actions_psm1[:, 7] = np.clip(action[:, 9], -0.698, 0.698)  # copy over gripper angles
                        
                        actions_psm2 = np.zeros((self.chunk_size, 8)) # pos, quat, jaw
                        actions_psm2[:, 0:3] = qpos_psm2[0:3] + action[:, 10:13] # convert to current translation
                        actions_psm2 = processing.convert_delta_6d_to_taskspace_quat_relative_endo(action[:, 10:], actions_psm2, qpos_psm2)
                        actions_psm2[:, 7] = np.clip(action[:, 19], -0.698, 0.698)  # copy over gripper angles  
                        
                    if self.action_mode == 'ego':
                        # compute actions for PSM1
                        actions_psm1 = np.zeros((self.chunk_size, 8)) # pos (3), quat (4), jaw (1) 
                        dts_psm1 = action[:, 0:3]
                        dquats_psm1 = self.convert_6d_rot_to_quat(action[:, 3:9]) # [n x 4] xyzw convention
                        actions_psm1 = self.convert_actions_to_SE3_then_final_actions(dts_psm1, dquats_psm1, qpos_psm1, action[:, 9]) # translation and quaternion
                        
                        # compute actions for PSM2
                        actions_psm2 = np.zeros((self.chunk_size, 8)) # pos (3), quat (4), jaw (1) 
                        dts_psm2 = action[:, 10:13]
                        dquats_psm2 = self.convert_6d_rot_to_quat(action[:, 13:19]) # [n x 4] xyzw convention
                        actions_psm2 = self.convert_actions_to_SE3_then_final_actions(dts_psm2, dquats_psm2, qpos_psm2, action[:, 19]) # translation and quaternion   
                    
                    self.execute_actions(actions_psm1, actions_psm2)
                    t += 1
                    
                except KeyboardInterrupt:
                    log.info("low level policy interrupted")
                    break
