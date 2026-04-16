import threading
import time

import cv2
from cv_bridge import CvBridge
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch

from srth_new.low_level_policy.inference.gui import run_inference_gui
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
            action_execution_hz: float,
            enable_gui: bool = True,
            start_paused: bool = True,
        ):
        self.policy = policy
        self.enable_gui = enable_gui
        self.start_paused = bool(start_paused)
        
        self.command = self._get_initial_command()
        self.init_threading_params()
        self._control_lock = threading.RLock()
        self.update_runtime_controls(
            prediction_frequency_hz=prediction_frequency_hz,
            action_execution_hz=action_execution_hz,
            command=self.command,
            request_reprediction=False,
        )
        self.init_ros()
        
    def init_threading_params(self):
        self._manual_pause = self.enable_gui and self.start_paused
        self._ros_pause = False
        self._execution_thread = None
        self._execution_stop_event = None
        self._inference_thread = None
        self._shutdown_event = threading.Event()
        self._control_update_event = threading.Event()

    def init_ros(self):

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
        self._ros_pause = msg.data
        
        if self._ros_pause:
            print("Robot paused. Waiting for the robot to be unpaused...")
        else:
            print("Robot unpaused. Resuming the low level policy...")

    def is_paused(self) -> bool:
        return self._manual_pause or self._ros_pause

    def is_manual_pause_enabled(self) -> bool:
        return self._manual_pause

    def set_manual_pause(self, paused: bool) -> None:
        paused = bool(paused)
        if paused == self._manual_pause:
            return

        self._manual_pause = paused
        self._control_update_event.set()

        if paused:
            log.info("Inference paused from GUI")
            self.stop_action_execution()
        else:
            log.info("Inference resumed from GUI")

    def get_pause_status_text(self) -> str:
        if self._manual_pause and self._ros_pause:
            return "Paused (GUI + ROS)"
        if self._manual_pause:
            return "Paused (GUI)"
        if self._ros_pause:
            return "Paused (ROS)"
        return "Running"

    def get_runtime_controls(self):
        with self._control_lock:
            return (
                self.prediction_frequency_hz,
                self.prediction_period,
                self.action_execution_hz,
                self.action_execution_period,
                self.command,
            )

    def get_available_commands(self) -> list[str]:
        saved_commands = list(getattr(self.policy, "training_text_conditionings", []))
        available_commands = []
        seen_commands = set()

        for raw_command in saved_commands:
            try:
                command = self._validate_command(raw_command)
            except ValueError:
                continue

            if command in seen_commands:
                continue

            seen_commands.add(command)
            available_commands.append(command)

        return available_commands

    def _get_initial_command(self) -> str:
        available_commands = self.get_available_commands()
        if available_commands:
            return available_commands[0]
        return "1_grasp"

    @staticmethod
    def _validate_prediction_frequency_hz(value) -> float:
        try:
            prediction_frequency_hz = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction_frequency_hz must be a number") from exc

        if prediction_frequency_hz <= 0:
            raise ValueError("prediction_frequency_hz must be > 0")
        return prediction_frequency_hz

    @staticmethod
    def _validate_action_execution_hz(value) -> float:
        try:
            action_execution_hz = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_execution_hz must be a number") from exc

        if action_execution_hz <= 0:
            raise ValueError("action_execution_hz must be > 0")
        return action_execution_hz

    @staticmethod
    def _validate_command(value) -> str:
        if value is None:
            raise ValueError("command must be a non-empty string")

        command = str(value).strip()
        if not command:
            raise ValueError("command must be a non-empty string")
        return command

    def update_runtime_controls(
        self,
        prediction_frequency_hz=None,
        action_execution_hz=None,
        command=None,
        request_reprediction: bool = True,
    ) -> None:
        current_prediction_frequency_hz = getattr(
            self,
            "prediction_frequency_hz",
            prediction_frequency_hz,
        )
        current_action_execution_hz = getattr(
            self,
            "action_execution_hz",
            action_execution_hz,
        )
        current_command = getattr(self, "command", command)

        new_prediction_frequency_hz = self._validate_prediction_frequency_hz(
            prediction_frequency_hz
            if prediction_frequency_hz is not None
            else current_prediction_frequency_hz
        )
        new_action_execution_hz = self._validate_action_execution_hz(
            action_execution_hz
            if action_execution_hz is not None
            else current_action_execution_hz
        )
        new_command = self._validate_command(
            command if command is not None else current_command
        )

        with self._control_lock:
            self.prediction_frequency_hz = new_prediction_frequency_hz
            self.prediction_period = 1.0 / new_prediction_frequency_hz
            self.action_execution_hz = new_action_execution_hz
            self.action_execution_period = 1.0 / new_action_execution_hz
            self.command = new_command

        if request_reprediction:
            self._control_update_event.set()

        log.info(
            "Updated inference controls: prediction_frequency_hz=%.3f Hz, action_execution_hz=%.3f Hz, command=%s",
            new_prediction_frequency_hz,
            new_action_execution_hz,
            new_command,
        )

    def _format_runtime_controls(self) -> str:
        prediction_frequency_hz, _, action_execution_hz, _, command = self.get_runtime_controls()
        return (
            f"prediction_frequency_hz={prediction_frequency_hz:.3f} Hz | "
            f"action_execution_hz={action_execution_hz:.3f} Hz | command={command}"
        )
    
    def get_image_dvrk(self):

        self.left_img = np.fromstring(self.rt.usb_image_left.data, np.uint8)
        self.left_img = cv2.imdecode(self.left_img, cv2.IMREAD_COLOR)
        self.left_img = cv2.resize(self.left_img, (480, 360))
        self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        self.left_img = rearrange(self.left_img, 'h w c -> c h w')

        self.psm2_img = np.fromstring(self.rt.endo_cam_psm2.data, np.uint8)
        self.psm2_img = cv2.resize(self.psm2_img, (480, 360))
        self.psm2_img = cv2.cvtColor(self.psm2_img, cv2.COLOR_BGR2RGB)
        self.psm2_img = rearrange(self.psm2_img, 'h w c -> c h w')

        self.psm1_img = np.fromstring(self.rt.endo_cam_psm1.data, np.uint8)
        self.psm1_img = cv2.resize(self.psm1_img, (480, 360))
        self.psm1_img = cv2.cvtColor(self.psm1_img, cv2.COLOR_BGR2RGB)
        self.psm1_img = rearrange(self.psm1_img, 'h w c -> c h w')

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
        _, _, _, _, command = self.get_runtime_controls()
        action = (
            self.policy(curr_image, current_pose, command_text=command)
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

            while self.is_paused() and not stop_event.is_set() and not self._shutdown_event.is_set():
                time.sleep(0.01)

            if stop_event.is_set() or self._shutdown_event.is_set() or rospy.is_shutdown():
                return

            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            _, _, _, action_execution_period, _ = self.get_runtime_controls()
            sleep_deadline = time.monotonic() + action_execution_period
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
        # self.stop_action_execution()
        # self._execution_stop_event = threading.Event()
        # self._execution_thread = threading.Thread(
        #     target=self.execute_actions,
        #     args=(actions_psm1, actions_psm2, self._execution_stop_event),
        #     daemon=True,
        # )
        # self._execution_thread.start()

        # the old code's way of doing it. let's test to see if this works...
        for jj in range(30):
            self.ral.spin_and_execute(self.psm1_app.run_full_pose_goal, actions_psm1[jj])
            self.ral.spin_and_execute(self.psm2_app.run_full_pose_goal, actions_psm2[jj])
            time.sleep(0.18)




    ## --------------------- main loop -----------------------

    def wait_for_required_topics(self, log_interval_s: float = 5.0) -> bool:
        next_log_time = 0.0

        try:
            while not self.rt.has_received_all_topics():
                if rospy.is_shutdown() or self._shutdown_event.is_set():
                    return False

                now = time.monotonic()
                if now >= next_log_time:
                    missing_topics = self.rt.get_missing_topics()
                    rospy.logerr(
                        "\033[91mWaiting for subscribed ROS topics to receive their first message. Missing topics:\n%s\033[0m",
                        "\n".join(f"  - {topic}" for topic in missing_topics),
                    )
                    next_log_time = now + log_interval_s

                sleep_duration = max(0.0, next_log_time - time.monotonic())
                time.sleep(min(0.1, sleep_duration))
        except KeyboardInterrupt:
            log.info("low level policy interrupted while waiting for ROS topics")
            self._shutdown_event.set()
            return False

        return True

    def run(self):
        if not self.wait_for_required_topics():
            return

        if self.enable_gui:
            gui_started = run_inference_gui(self)
            if gui_started:
                return

            if self._manual_pause:
                log.warning(
                    "Inference GUI was unavailable; starting unpaused so headless inference does not stall."
                )
                self._manual_pause = False

        self._run_inference_loop()

    def _run_inference_loop(self):
        log.info("-------------starting low level policy inference------------------\n")
        log.info("Initial inference controls: %s", self._format_runtime_controls())
        if self._manual_pause:
            log.info("Inference is starting in a paused state.")
        time.sleep(1)
        with torch.inference_mode():
            try:
                next_prediction_time = time.monotonic()

                while not rospy.is_shutdown() and not self._shutdown_event.is_set():
                    if self._control_update_event.is_set():
                        self._control_update_event.clear()
                        self.stop_action_execution()
                        next_prediction_time = time.monotonic()

                    now = time.monotonic()
                    if now < next_prediction_time:
                        time.sleep(min(0.01, next_prediction_time - now))
                        continue

                    if self.is_paused():
                        self.stop_action_execution()
                        _, prediction_period, _, _, _ = self.get_runtime_controls()
                        next_prediction_time = time.monotonic() + prediction_period
                        continue

                    actions_psm1, actions_psm2 = self.predict_actions()
                    if self._shutdown_event.is_set() or self._control_update_event.is_set():
                        continue

                    self.start_action_execution(actions_psm1, actions_psm2)
                    _, prediction_period, _, _, _ = self.get_runtime_controls()
                    next_prediction_time += prediction_period

                    if next_prediction_time < time.monotonic():
                        next_prediction_time = time.monotonic()

            except KeyboardInterrupt:
                log.info("low level policy interrupted")
            finally:
                self._shutdown_event.set()
                self.stop_action_execution()
