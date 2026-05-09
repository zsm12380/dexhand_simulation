import mujoco
import numpy as np
from dataclasses import dataclass

@dataclass
class TripodGeom:
    d_ti: float
    d_tm: float
    d_im: float
    area: float
    centroid: np.ndarray  # (3,)

def get_site_id(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"[ref_site] not found: {name}")
    return sid

def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def read_tripod_geom(model, data, thumb_site, index_site, middle_site) -> TripodGeom:
    tid = get_site_id(model, thumb_site)
    iid = get_site_id(model, index_site)
    mid = get_site_id(model, middle_site)

    p_t = data.site_xpos[tid].copy()
    p_i = data.site_xpos[iid].copy()
    p_m = data.site_xpos[mid].copy()

    d_ti = np.linalg.norm(p_t - p_i)
    d_tm = np.linalg.norm(p_t - p_m)
    d_im = np.linalg.norm(p_i - p_m)
    area = triangle_area(p_t, p_i, p_m)
    centroid = (p_t + p_i + p_m) / 3.0

    return TripodGeom(d_ti=d_ti, d_tm=d_tm, d_im=d_im, area=area, centroid=centroid)
