from tripod_exec_types import SynergyU, CtrlFrame
from tripod_exec_config import CTRL_RANGE, AUX_FIXED_CTRL, OPEN_HAND_CTRL

def lerp(a, b, t): return a + (b - a) * t
def clip(x, lo, hi): return max(lo, min(hi, x))

FLIP_THUMB_OPPOSE_SIGN = True   # 你当前用 False，先别改

FLEX_GAIN = 0.65
THUMB_GAIN = 0.60

def decode_u_to_ctrl(u: SynergyU) -> CtrlFrame:
    c = {}
    c["THJ1"] = _scale(THUMB_GAIN * u.thumb_flex, "THJ1")
    c["THJ2"] = _scale(0.9 * THUMB_GAIN * u.thumb_flex, "THJ2")

    sign = -1.0 if FLIP_THUMB_OPPOSE_SIGN else 1.0
    th3 = lerp(-0.6*sign, 0.6*sign, u.thumb_oppose)
    th4 = lerp(-0.7*sign, 0.7*sign, u.thumb_oppose)

    c["THJ3"] = clip(th3, *CTRL_RANGE["THJ3"])
    c["THJ4"] = clip(th4, *CTRL_RANGE["THJ4"])

    c["FFJ3"] = _scale(0.95 * FLEX_GAIN * u.index_flex, "FFJ3")
    # ... 后面 FFJ2/FFJ1/FFJ4、MFJ 部分保持不变 ...
    c["FFJ2"] = _scale(0.85 * FLEX_GAIN * u.index_flex, "FFJ2")
    c["FFJ1"] = _scale(0.75 * FLEX_GAIN * u.index_flex, "FFJ1")
    c["FFJ4"] = clip(lerp(0.15, -0.15, 1.0 - u.aperture), *CTRL_RANGE["FFJ4"])

    c["MFJ3"] = _scale(0.95 * FLEX_GAIN * u.middle_flex, "MFJ3")
    c["MFJ2"] = _scale(0.85 * FLEX_GAIN * u.middle_flex, "MFJ2")
    c["MFJ1"] = _scale(0.75 * FLEX_GAIN * u.middle_flex, "MFJ1")
    c["MFJ4"] = clip(lerp(-0.05, 0.05, u.aperture), *CTRL_RANGE["MFJ4"])

    c.update(AUX_FIXED_CTRL)
    for k, v in OPEN_HAND_CTRL.items():
        c.setdefault(k, v)
    return CtrlFrame(ctrl=c)

def _scale(v01, name):
    lo, hi = CTRL_RANGE[name]
    return clip(lerp(lo, hi, v01), lo, hi)
