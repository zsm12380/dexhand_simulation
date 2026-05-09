from tripod_exec_types import HumanFrameParams, SynergyU
from dataclasses import dataclass

def clamp01(x): return max(0.0, min(1.0, x))

def map_human_to_synergy(h: HumanFrameParams) -> SynergyU:
    F_n = clamp01(h.F / 1.0)
    D_n = clamp01((h.D - 0.4) / 0.8)
    A_n = clamp01((h.A - 0.1) / 0.4)

    thumb_flex  = clamp01(0.65 * F_n + 0.35 * clamp01(h.f_thumb))
    index_flex  = clamp01(0.70 * F_n + 0.30 * clamp01(h.f_index))
    middle_flex = clamp01(0.70 * F_n + 0.30 * clamp01(h.f_middle))
    thumb_oppose = clamp01(0.6 * (1.0 - D_n) + 0.4 * (1.0 - A_n))
    aperture = clamp01(D_n)

    return SynergyU(thumb_flex, thumb_oppose, index_flex, middle_flex, aperture)

@dataclass
class TripodGeomTarget:
    d_ti: float
    d_tm: float
    d_im: float
    area: float

def map_human_to_geom_target(h: HumanFrameParams) -> TripodGeomTarget:
    """
    把A/D/F映射为几何目标（单位先用米的经验值，后续可标定）
    D大 -> 三边长大；F大 -> 收拢(边长减小)；A直接映射三角面积
    """
    # 基准范围（MVP，可调）
    d_open, d_close = 0.070, 0.028   # 7cm -> 2.8cm
    F_n = clamp01(h.F)
    D_n = clamp01((h.D - 0.4)/0.8)
    A_n = clamp01((h.A - 0.1)/0.4)

    # 边长目标（综合D和F）
    d_base = (0.65 * (d_close + (d_open - d_close) * D_n) +
              0.35 * (d_open - (d_open - d_close) * F_n))

    # 允许轻微非等边，符合tripod
    d_ti = d_base * 1.00
    d_tm = d_base * 1.05
    d_im = d_base * 0.95

    # 面积目标（与A相关）
    area_open, area_close = 0.0012, 0.00018
    area = area_close + (area_open - area_close) * (0.6*D_n + 0.4*A_n)

    return TripodGeomTarget(d_ti=d_ti, d_tm=d_tm, d_im=d_im, area=area)
