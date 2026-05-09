from dataclasses import dataclass
from typing import Dict, List

@dataclass
class HumanFrameParams:
    A: float
    D: float
    F: float
    f_thumb: float
    f_index: float
    f_middle: float

@dataclass
class HumanKeyframesOutput:
    shape_start: HumanFrameParams
    preshape: HumanFrameParams
    enclosure: HumanFrameParams
    final: HumanFrameParams
    T_start_pre: int = 20
    T_pre_enc: int = 25
    T_enc_final: int = 30

@dataclass
class SynergyU:
    thumb_flex: float
    thumb_oppose: float
    index_flex: float
    middle_flex: float
    aperture: float

@dataclass
class CtrlFrame:
    ctrl: Dict[str, float]

@dataclass
class PlannedTrajectory:
    ctrl_seq: List[CtrlFrame]
    dt: float

