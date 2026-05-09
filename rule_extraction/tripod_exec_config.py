from pathlib import Path

# 当前文件: rule_extraction/tripod_exec_config.py
# 项目根目录: ../
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XML = PROJECT_ROOT / "urdf" / "dexhand_lh_rl.xml"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CTRL_RANGE = {
    "THJ4": (-1.0472, 1.0472),
    "THJ3": (-1.0472, 1.0472),
    "THJ2": (0.0, 1.5708),
    "THJ1": (0.0, 1.5708),
    "FFJ4": (-0.2618, 0.2618),
    "FFJ3": (0.0, 1.5708),
    "FFJ2": (0.0, 1.5708),
    "FFJ1": (0.0, 1.5708),
    "MFJ4": (-0.2618, 0.2618),
    "MFJ3": (0.0, 1.5708),
    "MFJ2": (0.0, 1.5708),
    "MFJ1": (0.0, 1.5708),
    "RFJ4": (-0.2618, 0.2618),
    "RFJ3": (0.0, 1.5708),
    "RFJ2": (0.0, 1.5708),
    "RFJ1": (0.0, 1.5708),
    "LFJ4": (-0.2618, 0.2618),
    "LFJ3": (0.0, 1.5708),
    "LFJ2": (0.0, 1.5708),
    "LFJ1": (0.0, 1.5708),
    "act_left": (-0.008, 0.008),
    "act_right": (-0.008, 0.008),
}

AUX_FIXED_CTRL = {
    "RFJ4": 0.05, "RFJ3": 0.4, "RFJ2": 0.6, "RFJ1": 0.7,
    "LFJ4": -0.05, "LFJ3": 0.5, "LFJ2": 0.7, "LFJ1": 0.8,
    "act_left": 0.0, "act_right": 0.0,
}

OPEN_HAND_CTRL = {
    "THJ1": 0.05, "THJ2": 0.05, "THJ3": 0.0, "THJ4": 0.0,
    "FFJ1": 0.05, "FFJ2": 0.05, "FFJ3": 0.05, "FFJ4": 0.0,
    "MFJ1": 0.05, "MFJ2": 0.05, "MFJ3": 0.05, "MFJ4": 0.0,
}
