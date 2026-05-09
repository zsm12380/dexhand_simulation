# rule_extraction/tripod_target_builder.py
import math
from dataclasses import dataclass

@dataclass
class TripodTargetM:
    d_ti: float
    d_tm: float
    d_im: float
    area_tip: float
    d_th_p: float
    d_ff_p: float
    d_mf_p: float

def heron_area_mm2(a, b, c):
    s = 0.5 * (a + b + c)
    v = max(0.0, s*(s-a)*(s-b)*(s-c))
    return math.sqrt(v)

def build_targets_with_scale(s_xml_over_real: float):
    """
    真实世界目标(mm) -> XML/MuJoCo目标(m)
    你可继续微调这些real值
    """
    # phase targets in REAL mm
    real = {
        "shape_start": dict(d_ti=46, d_tm=48, d_im=38, d_th_p=48, d_ff_p=40, d_mf_p=42),
        "preshape":    dict(d_ti=38, d_tm=40, d_im=32, d_th_p=42, d_ff_p=34, d_mf_p=36),
        "enclosure":   dict(d_ti=33, d_tm=34, d_im=28, d_th_p=37, d_ff_p=30, d_mf_p=31),
        "final":       dict(d_ti=30, d_tm=31, d_im=26, d_th_p=34, d_ff_p=28, d_mf_p=30),
    }

    out = {}
    for k, v in real.items():
        a, b, c = v["d_ti"], v["d_tm"], v["d_im"]
        area_real_mm2 = heron_area_mm2(a, b, c)

        # 距离: mm -> xml-mm(乘s) -> m
        d_ti = v["d_ti"] * s_xml_over_real / 1000.0
        d_tm = v["d_tm"] * s_xml_over_real / 1000.0
        d_im = v["d_im"] * s_xml_over_real / 1000.0
        d_th_p = v["d_th_p"] * s_xml_over_real / 1000.0
        d_ff_p = v["d_ff_p"] * s_xml_over_real / 1000.0
        d_mf_p = v["d_mf_p"] * s_xml_over_real / 1000.0

        # 面积: mm^2 -> xml-mm^2(乘s^2) -> m^2
        area_tip = area_real_mm2 * (s_xml_over_real**2) / 1e6

        out[k] = TripodTargetM(
            d_ti=d_ti, d_tm=d_tm, d_im=d_im, area_tip=area_tip,
            d_th_p=d_th_p, d_ff_p=d_ff_p, d_mf_p=d_mf_p
        )
    return out
