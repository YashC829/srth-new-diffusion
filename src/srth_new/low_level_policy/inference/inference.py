import os
import threading
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

from srth_new.general.utils import processing
from srth_new.low_level_policy.models.act_model import ACTPolicy

# From ros packages
import crtk
from srth_new.low_level_policy.inference.dvrk_control import example_application
import rospy
from srth_new.low_level_policy.inference.rostopics import ros_topics
from std_msgs.msg import Bool

import logging
log = logging.getLogger(__name__)


class LowLevelPolicy:

    ## ----------------- initializations ----------------
    def __init__(
            self,
            policy: ACTPolicy,
            prediction_frequency_hz: float,
            sleep_rate: float,
        ):
        self.policy = policy
        self.prediction_frequency_hz = prediction_frequency_hz
        self.prediction_period = 1.0 / self.prediction_frequency_hz
        if prediction_frequency_hz <= 0:
            raise ValueError("prediction_frequency_hz must be > 0")
        self.sleep_rate = sleep_rate

        self.initialize_hardcoded_parameters()
        self.initialize_ros()

        # TODO: This is hardcoded for now. Later, this should be predicted by the
        # high level policy
        self.command = "1_grasp"
        
    def initialize_hardcoded_parameters(self):
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
        self._execution_thread = None
        self._execution_stop_event = None
        self._shutdown_event = threading.Event()

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
    
    def get_image_dvrk(self):
        self.iter += 1

        def decode_ros_img(img_data, img_shape: Tuple[int, int]):
            img = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, img_shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = rearrange(img, 'h w c -> c h w')
            return img
        
        self.left_img = decode_ros_img(self.rt.usb_image_left.data, img_shape=(480, 360))
        self.psm2_img = decode_ros_img(self.rt.endo_cam_psm2, img_shape=(480, 360))
        self.psm1_img = decode_ros_img(self.rt.endo_cam_psm1, img_shape=(480, 360))

        curr_image = np.stack([self.left_img, self.psm2_img, self.psm1_img], axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
        
        return curr_image

    def get_current_pose(self):
        qpos_psm1 = np.array(
            (
                self.rt.psm1_pose.position.x,
                self.rt.psm1_pose.position.y,
                self.rt.psm1_pose.position.z,
                self.rt.psm1_pose.orientation.x,
                self.rt.psm1_pose.orientation.y,
                self.rt.psm1_pose.orientation.z,
                self.rt.psm1_pose.orientation.w,
                self.rt.psm1_jaw,
            )
        )
        qpos_psm2 = np.array(
            (
                self.rt.psm2_pose.position.x,
                self.rt.psm2_pose.position.y,
                self.rt.psm2_pose.position.z,
                self.rt.psm2_pose.orientation.x,
                self.rt.psm2_pose.orientation.y,
                self.rt.psm2_pose.orientation.z,
                self.rt.psm2_pose.orientation.w,
                self.rt.psm2_jaw,
            )
        )
        current_pose = torch.from_numpy(
            np.concatenate((qpos_psm1, qpos_psm2)).astype(np.float32)
        ).unsqueeze(0)
        return current_pose, qpos_psm1, qpos_psm2

    def predict_actions(self):
        curr_image = self.get_image_dvrk()
        current_pose, _, _ = self.get_current_pose()
        action = (
            self.policy(curr_image, current_pose, command_text=self.command)
            .cpu()
            .numpy()
            .squeeze(0)
        )
        return action[:, :8], action[:, 8:16]
    
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

    def execute_actions(self, actions_psm1, actions_psm2, stop_event: threading.Event):
        num_steps = min(self.policy.num_queries, len(actions_psm1), len(actions_psm2))

        for jj in range(num_steps):
            if stop_event.is_set() or self._shutdown_event.is_set() or rospy.is_shutdown():
                return

            while self.pause and not stop_event.is_set() and not self._shutdown_event.is_set():
                time.sleep(0.01)

            if stop_event.is_set() or self._shutdown_event.is_set() or rospy.is_shutdown():
                return

            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            sleep_deadline = time.monotonic() + self.sleep_rate
            while not stop_event.is_set() and not self._shutdown_event.is_set():
                remaining = sleep_deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))

    def stop_action_execution(self):
        if self._execution_stop_event is not None:
            self._execution_stop_event.set()

        if self._execution_thread is not None and self._execution_thread.is_alive():
            self._execution_thread.join()

        self._execution_thread = None
        self._execution_stop_event = None

    def start_action_execution(self, actions_psm1, actions_psm2):
        self.stop_action_execution()
        self._execution_stop_event = threading.Event()
        self._execution_thread = threading.Thread(
            target=self.execute_actions,
            args=(actions_psm1, actions_psm2, self._execution_stop_event),
            daemon=True,
        )
        self._execution_thread.start()

    ## --------------------- main loop -----------------------

    def run(self):
        log.info("-------------starting low level policy inference------------------\n")
        log.info(
            "Low-level policy prediction frequency set to %.3f Hz",
            self.prediction_frequency_hz,
        )
        time.sleep(1)
        with torch.inference_mode():
            try:
                next_prediction_time = time.monotonic()

                while not rospy.is_shutdown():
                    now = time.monotonic()
                    if now < next_prediction_time:
                        time.sleep(min(0.01, next_prediction_time - now))
                        continue

                    if self.pause:
                        self.stop_action_execution()
                        next_prediction_time = time.monotonic() + self.prediction_period
                        continue

                    actions_psm1, actions_psm2 = self.predict_actions()
                    self.start_action_execution(actions_psm1, actions_psm2)
                    next_prediction_time += self.prediction_period

                    if next_prediction_time < time.monotonic():
                        next_prediction_time = time.monotonic()

            except KeyboardInterrupt:
                log.info("low level policy interrupted")
            finally:
                self._shutdown_event.set()
                self.stop_action_execution()
