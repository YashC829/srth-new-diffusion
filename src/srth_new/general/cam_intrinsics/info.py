from pathlib import Path

current_dir = Path(__file__).resolve().parent

LEFT_ENDO_INTRINSICS_PATH = current_dir.joinpath("endoscope_left.yaml")
RIGHT_ENDO_INTRINSICS_PATH = current_dir.joinpath("endoscope_right.yaml")
