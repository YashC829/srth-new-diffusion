import os
import threading
import time

import cv2
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq

from srth_new.general import constants
from srth_new.low_level_policy.inference.gui import run_inference_gui
from srth_new.low_level_policy.models.act_model import ACTPolicy

# From ros packages
import crtk
from srth_new.low_level_policy.inference.dvrk_control_ros2 import example_application
from srth_new.low_level_policy.inference.rostopics import ros_topics
from std_msgs.msg import Bool

import logging
log = logging.getLogger(__name__)

DVRK_COMPUTER_IP = os.environ.get("DVRK_COMPUTER_IP")
if DVRK_COMPUTER_IP is None:
    raise Exception(f"Must set dvrk computer IP address with: export DVRK_COMPUTER_IP=<ip_addr>")

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
        
        # zmq initialization
        self.image_frames = {
            constants.LEFT_ENDOSCOPE_TOPIC: None,
            constants.RIGHT_ENDOSCOPE_TOPIC: None,
            constants.PSM1_WRIST_CAMERA_TOPIC: None,
            constants.PSM2_WRIST_CAMERA_TOPIC: None,
        }
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.receiver_thread, daemon=True)
        self.thread.start()

        self.windows = {}
        
    def init_threading_params(self):
        self._manual_pause = self.enable_gui and self.start_paused
        self._ros_pause = False
        self._execution_thread = None
        self._execution_stop_event = None
        self._inference_thread = None
        self._shutdown_event = threading.Event()
        self._control_update_event = threading.Event()
        self._ros_cleaned_up = False

    def init_ros(self):
        self.ral = crtk.ral('dvrk_arm_test')
        self.ral.spin()
        self.ral.on_shutdown(self._shutdown_event.set)
        self.rt = ros_topics(self.ral._node)
        self.psm1_app = example_application(self.ral, "PSM1", 5.0)
        self.psm2_app = example_application(self.ral, "PSM2", 5.0)
        self.pause_sub = self.ral._node.create_subscription(
            Bool,
            "/pause_robot",
            self.pause_robot_callback,
            10,
        )

    def visualize_zmq_imgs(self):

        with self.frame_lock:
            frames_snapshot = {k: v.copy() if v is not None else None for k, v in self.image_frames.items()}

        for topic, frame in frames_snapshot.items():
            if frame is not None:
                if topic not in self.windows:
                    cv2.namedWindow(topic, cv2.WINDOW_NORMAL)
                    self.windows[topic] = True
                cv2.imshow(topic, frame)
                
                cv2.waitKey(1)  # <-- REQUIRED

    def receiver_thread(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{DVRK_COMPUTER_IP}:5555")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVHWM, 1)

        for topic in self.image_frames:
            sock.setsockopt_string(zmq.SUBSCRIBE, topic)

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if sock in events:
                topic, data = sock.recv_multipart()
                topic = topic.decode()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.frame_lock:
                        self.image_frames[topic] = frame

        sock.close()
        ctx.term()


    ## --------------------- callbacks -----------------------
    
    def pause_robot_callback(self, msg):
        self._ros_pause = msg.data
        
        if self._ros_pause:
            print("Robot paused. Waiting for the robot to be unpaused...")
        else:
            print("Robot unpaused. Resuming the low level policy...")

    def is_ros_shutdown(self) -> bool:
        return self.ral.is_shutdown()

    def shutdown_ros(self) -> None:
        if self._ros_cleaned_up:
            return

        self._ros_cleaned_up = True
        self._shutdown_event.set()

        try:
            if getattr(self, "rt", None) is not None:
                self.rt.destroy()
        finally:
            if getattr(self, "pause_sub", None) is not None:
                self.ral._node.destroy_subscription(self.pause_sub)
                self.pause_sub = None
            self.ral.shutdown()

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

    @staticmethod
    def _decode_compressed_image(message) -> np.ndarray:
        image_buffer = np.frombuffer(bytes(message.data), dtype=np.uint8)
        image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode compressed image from ROS topic")
        return image
    
    def get_image_dvrk(self):

        # NOTE: This is the code that worked in ros1. this should work in ros2, but
        # there seemed to be some odd network issues. Leaving this here for
        # reference just in case
        # self.left_img = self._decode_compressed_image(self.rt.usb_image_left)
        # self.left_img = cv2.resize(self.left_img, (480, 360))
        # self.left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2RGB)
        # self.left_img = rearrange(self.left_img, 'h w c -> c h w')

        # self.psm2_img = self._decode_compressed_image(self.rt.endo_cam_psm2)
        # self.psm2_img = cv2.resize(self.psm2_img, (480, 360))
        # self.psm2_img = cv2.cvtColor(self.psm2_img, cv2.COLOR_BGR2RGB)
        # self.psm2_img = rearrange(self.psm2_img, 'h w c -> c h w')

        # self.psm1_img = self._decode_compressed_image(self.rt.endo_cam_psm1)
        # self.psm1_img = cv2.resize(self.psm1_img, (480, 360))
        # self.psm1_img = cv2.cvtColor(self.psm1_img, cv2.COLOR_BGR2RGB)
        # self.psm1_img = rearrange(self.psm1_img, 'h w c -> c h w')

        # self.visualize_zmq_imgs()

        with self.frame_lock:
            left_img = self.image_frames[constants.LEFT_ENDOSCOPE_TOPIC]
            psm2_img = self.image_frames[constants.PSM2_WRIST_CAMERA_TOPIC]
            psm1_img = self.image_frames[constants.PSM1_WRIST_CAMERA_TOPIC]

        left_img = torch.from_numpy(left_img.transpose(2, 0, 1)).cuda().unsqueeze(0)
        psm2_img = torch.from_numpy(psm2_img.transpose(2, 0, 1)).cuda().unsqueeze(0)
        psm1_img = torch.from_numpy(psm1_img.transpose(2, 0, 1)).cuda().unsqueeze(0)

        return left_img, psm2_img, psm1_img

        curr_image = np.stack([left_img, psm2_img, psm1_img], axis=0)
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
        endo_img, lw_img, rw_img = self.get_image_dvrk()
        current_pose, _, _ = self.get_current_pose()
        _, _, _, _, command = self.get_runtime_controls()
        action = (
            self.policy(endo_img, lw_img, rw_img, current_pose, command_text=command)
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
            if stop_event.is_set() or self._shutdown_event.is_set() or self.is_ros_shutdown():
                return

            while self.is_paused() and not stop_event.is_set() and not self._shutdown_event.is_set():
                time.sleep(0.01)

            if stop_event.is_set() or self._shutdown_event.is_set() or self.is_ros_shutdown():
                return

            self.psm1_app.run_full_pose_goal(actions_psm1[jj])
            self.psm2_app.run_full_pose_goal(actions_psm2[jj])
            _, _, _, action_execution_period, _ = self.get_runtime_controls()
            sleep_deadline = time.monotonic() + action_execution_period
            while not stop_event.is_set() and not self._shutdown_event.is_set():
                remaining = sleep_deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(0.01, remaining))

    def execute_actions_sequential(self, actions_psm1, actions_psm2):
        num_steps = min(self.policy.num_queries, len(actions_psm1), len(actions_psm2))
        max_steps = 40
        for jj in range(num_steps):
            if jj > max_steps:
                break
            if self._shutdown_event.is_set() or self.is_ros_shutdown():
                return

            while self.is_paused() and not self._shutdown_event.is_set():
                time.sleep(0.01)

            if self._shutdown_event.is_set() or self.is_ros_shutdown():
                return
            self.rt.node.get_logger().info(f"PSM2 Action: {actions_psm2[jj]}")
            # self.psm1_app.run_full_pose_goal(actions_psm1[jj])
            self.psm2_app.run_full_pose_goal(actions_psm2[jj], self.rt.psm2_jaw)
            _, _, _, action_execution_period, _ = self.get_runtime_controls()
            sleep_deadline = time.monotonic() + action_execution_period
            time.sleep(0.01)
            while not self._shutdown_event.is_set():
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
        self.execute_actions_sequential(actions_psm1, actions_psm2)
        # self.stop_action_execution()
        # self._execution_stop_event = threading.Event()
        # self._execution_thread = threading.Thread(
        #     target=self.execute_actions,
        #     args=(actions_psm1, actions_psm2, self._execution_stop_event),
        #     daemon=True,
        # )
        # self._execution_thread.start()


    def wait_for_zmq_imgs(self):

        while any(v is None for v in self.image_frames.values()):
            log.info(f"Waiting for ZMQ image topics: {", ".join([k for k, v in self.image_frames.items() if v is None])}")

    ## --------------------- main loop -----------------------

    def wait_for_required_topics(self, log_interval_s: float = 5.0) -> bool:
        next_log_time = 0.0

        try:
            while not self.rt.has_received_all_topics():
                if self.is_ros_shutdown() or self._shutdown_event.is_set():
                    return False

                now = time.monotonic()
                if now >= next_log_time:
                    missing_topics = self.rt.get_missing_topics()
                    log.error(
                        "Waiting for subscribed ROS topics to receive their first message. Missing topics:\n%s",
                        "\n".join(f"  - {topic}" for topic in missing_topics),
                    )
                    next_log_time = now + log_interval_s

                sleep_duration = max(0.0, next_log_time - time.monotonic())
                time.sleep(min(0.1, sleep_duration))

            self.wait_for_zmq_imgs()
            
        except KeyboardInterrupt:
            log.info("low level policy interrupted while waiting for ROS topics")
            self._shutdown_event.set()
            return False

        return True

    def run(self):
        self.policy.eval()
        if not self.wait_for_required_topics():
            self.shutdown_ros()
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

                while not self.is_ros_shutdown() and not self._shutdown_event.is_set():
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
                self.shutdown_ros()
