import importlib.util
import json
import os
from pathlib import Path

# lerobot dataset constants
FPS = 30
ROBOT_NAME = "dvrk-si"
ENDOSCOPE_IMG_SHAPE = (540, 960, 3)
WRIST_CAM_IMG_SHAPE = (480, 640, 3)
STATES_NAME = [
    "psm1_pose.position.x",
    "psm1_pose.position.y",
    "psm1_pose.position.z",
    "psm1_pose.orientation.x",
    "psm1_pose.orientation.y",
    "psm1_pose.orientation.z",
    "psm1_pose.orientation.w",
    "psm1_jaw",
    "psm2_pose.position.x",
    "psm2_pose.position.y",
    "psm2_pose.position.z",
    "psm2_pose.orientation.x",
    "psm2_pose.orientation.y",
    "psm2_pose.orientation.z",
    "psm2_pose.orientation.w",
    "psm2_jaw",
]
ACTIONS_NAME = [
    "psm1_sp.position.x",
    "psm1_sp.position.y",
    "psm1_sp.position.z",
    "psm1_sp.orientation.x",
    "psm1_sp.orientation.y",
    "psm1_sp.orientation.z",
    "psm1_sp.orientation.w",
    "psm1_jaw_sp",
    "psm2_sp.position.x",
    "psm2_sp.position.y",
    "psm2_sp.position.z",
    "psm2_sp.orientation.x",
    "psm2_sp.orientation.y",
    "psm2_sp.orientation.z",
    "psm2_sp.orientation.w",
    "psm2_jaw_sp",
]
CHOLECYSTECTOMY_FEATURES={
    "images.endoscope.left": {
        "dtype": "video",
        "shape": ENDOSCOPE_IMG_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "images.endoscope.right": {
        "dtype": "video",
        "shape": ENDOSCOPE_IMG_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "images.wrist.left": {
        "dtype": "video",
        "shape": WRIST_CAM_IMG_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "images.wrist.right": {
        "dtype": "video",
        "shape": WRIST_CAM_IMG_SHAPE,
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (len(STATES_NAME),),
        "names": [STATES_NAME],
    },
    "action": {
        "dtype": "float32",
        "shape": (len(ACTIONS_NAME),),
        "names": [ACTIONS_NAME],
    },
    "esu_signal": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "meta.high_level_phase": {
        "dtype": "string",
        "shape": (1,),
        "names": ["value"],
    },
    "meta.low_level_phase": {
        "dtype": "string",
        "shape": (1,),
        "names": ["value"],
    },
    "meta.tool.psm1": {
        "dtype": "string",
        "shape": (1,),
        "names": ["value"],
    },
    "meta.tool.psm2": {
        "dtype": "string",
        "shape": (1,),
        "names": ["value"],
    },
    "meta.data_collector": {
        "dtype": "string",
        "shape": (1, ),
        "names": ["value"]
    },
    "meta.tissue_id": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}

# ros2 topics
LEFT_ENDOSCOPE_TOPIC = "/jhu_daVinci/left/camera/image_raw/compressed"
RIGHT_ENDOSCOPE_TOPIC = "/jhu_daVinci/right/camera/image_raw/compressed"
PSM1_WRIST_CAMERA_TOPIC = "/PSM1/endoscope_img/compressed"
PSM2_WRIST_CAMERA_TOPIC = "/PSM2/endoscope_img/compressed"

# the below maps valid high level phases to sets of corresponding valid low
# level phases
PHASES = {
    "unzipping": {
        "1_grabbing_gallbladder_right",
        "1_grabbing_gallbladder_right_recovery",
        "2_initial_incision",
        "2_initial_incision_recovery",
        "3_hook_to_local_home",
        "4_hook_tissue",
        "4_hook_tissue_recovery",
        "5_cauterize_tissue_right",
        "6_hook_to_global_home",
        "7_grasper_to_home",
        "8_grabbing_gallbladder_left",
        "8_grabbing_gallbladder_left_recovery",
        "9_returning_to_initial_incision",
        "9_returning_to_initial_incision_recovery",
        "10_cauterize_tissue_left",
        "11_regrab",
        "12_hook_to_global_home",
        "13_grasper_to_home"
    },
    "calot_dissection": {
        "1_grabbing_gallbladder_right",
        "1_grabbing_gallbladder_right_recovery",
        "2_move_camera_down",
        "3_forceps_approach",
        "3_forceps_approach_recovery",
        "4_forceps_open"
    },
    "clipping_and_cutting": {
        "1_grabbing_gallbladder",
        "1_grabbing_gallbladder_recovery",
        "2_clipping_first_clip_left_tube",
        "2_clipping_first_clip_left_tube_recovery",
        "3_going_back_first_clip_left_tube",
        "4_clipping_second_clip_left_tube",
        "4_clipping_second_clip_left_tube_recovery",
        "5_going_back_second_clip_left_tube",
        "6_clipping_third_clip_left_tube",
        "6_clipping_third_clip_left_tube_recovery",
        "7_going_back_third_clip_left_tube",
        "8_go_to_the_cutting_position_left_tube",
        "8_go_to_the_cutting_position_left_tube_recovery",
        "9_go_back_from_the_cut_left_tube",
        "10_clipping_first_clip_right_tube",
        "10_clipping_first_clip_right_tube_recovery",
        "11_going_back_first_clip_right_tube",
        "12_clipping_second_clip_right_tube",
        "12_clipping_second_clip_right_tube_recovery",
        "13_going_back_second_clip_right_tube",
        "14_clipping_third_clip_right_tube",
        "14_clipping_third_clip_right_tube_recovery",
        "15_going_back_third_clip_right_tube",
        "16_go_to_the_cutting_position_right_tube",
        "16_go_to_the_cutting_position_right_tube_recovery",
        "17_go_back_from_the_cut_right_tube",
    },
    "gallbladder_removal": {
        "1_grabbing_gallbladder_bottom_up",
        "1_grabbing_gallbladder_bottom_up_recovery",
        "2_hook_tissue",
        "2_hook_tissue_recovery",
        "3_pull_and_burn",
        "4_pull_up",
        "5_zoom_out"
    }
}

EPISODE_CSV_FILENAME = "ee_csv.csv"

LEFT_ENDOSCOPE_CAM_NAME = "left"
LEFT_ENDOSCOPE_CAM_DIR_NAME = "left_img_dir"
LEFT_ENDOSCOPE_CAM_IMG_SUFFIX = "left"
RIGHT_ENDOSCOPE_CAM_NAME = "right"
RIGHT_ENDOSCOPE_CAM_DIR_NAME = "right_img_dir"
RIGHT_ENDOSCOPE_CAM_IMG_SUFFIX = "right"
PSM2_WRIST_CAM_NAME = "psm2"
PSM2_WRIST_CAM_DIR_NAME = "endo_psm2"
PSM2_WRIST_CAM_IMG_SUFFIX = "psm2"
PSM1_WRIST_CAM_NAME = "psm1"
PSM1_WRIST_CAM_DIR_NAME = "endo_psm1"
PSM1_WRIST_CAM_IMG_SUFFIX = "psm1"

LOW_LEVEL_DATASET_CAMERA_NAMES = ["left", "left_wrist", "right_wrist"]
LOW_LEVEL_DATASET_CAMERA_SUFFIXES = ["_left.jpg", "_psm2.jpg", "_psm1.jpg"]

CAMERA_NAMES = [LEFT_ENDOSCOPE_CAM_DIR_NAME, PSM1_WRIST_CAM_DIR_NAME, PSM2_WRIST_CAM_DIR_NAME]
IMG_RESIZE_SIZE = (224, 224)

TISSUE_FOLDER_NAME = lambda i: f"tissue_{i}"

HEADER_NAME_QPOS_PSM1 = ["psm1_pose.position.x", "psm1_pose.position.y", "psm1_pose.position.z",
                        "psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w",
                        "psm1_jaw"]

HEADER_NAME_QPOS_PSM2 = ["psm2_pose.position.x", "psm2_pose.position.y", "psm2_pose.position.z",
                        "psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w",
                        "psm2_jaw"]

HEADER_NAME_ACTIONS_PSM1 = ["psm1_sp.position.x", "psm1_sp.position.y", "psm1_sp.position.z",
                            "psm1_sp.orientation.x", "psm1_sp.orientation.y", "psm1_sp.orientation.z", "psm1_sp.orientation.w",
                            "psm1_jaw_sp"]

HEADER_NAME_ACTIONS_PSM2 = ["psm2_sp.position.x", "psm2_sp.position.y", "psm2_sp.position.z",
                            "psm2_sp.orientation.x", "psm2_sp.orientation.y", "psm2_sp.orientation.z", "psm2_sp.orientation.w",
                            "psm2_jaw_sp"]

HEADER_ECM = ["ecm_pose.position.x", "ecm_pose.position.y", "ecm_pose.position.z",
                    "ecm_pose.orientation.x", "ecm_pose.orientation.y", 
                    "ecm_pose.orientation.z", "ecm_pose.orientation.w"]

HEADER_NAME_ESU_SIGNAL = "ESU_signal"

QUAT_CP_PSM1 = ["psm1_pose.orientation.x", "psm1_pose.orientation.y", "psm1_pose.orientation.z", "psm1_pose.orientation.w"]
QUAT_CP_PSM2 = ["psm2_pose.orientation.x", "psm2_pose.orientation.y", "psm2_pose.orientation.z", "psm2_pose.orientation.w"]

# dvrk constants
JAW_MAX_ANGLE_RAD = 1.3
JAW_MIN_ANGLE_RAD = -.36


# dataset statistics caching
spec = importlib.util.find_spec("srth_new")
PACKAGE_ROOT = Path(spec.origin).resolve().parent.parent.parent
DATASET_STATS_CACHE_DIR = PACKAGE_ROOT.joinpath(".dataset_stats_cache")
DATASET_STATS_CACHE_FILE = DATASET_STATS_CACHE_DIR.joinpath("dataset_stats.json")
os.makedirs(DATASET_STATS_CACHE_DIR, exist_ok=True)
if not os.path.exists(DATASET_STATS_CACHE_FILE):
    with open(DATASET_STATS_CACHE_FILE, "w") as file:
        json.dump({}, file)
