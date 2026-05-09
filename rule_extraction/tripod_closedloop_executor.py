import time
import numpy as np
import mujoco
import mujoco.viewer

from tripod_mapper import map_human_to_synergy
from tripod_decoder import decode_u_to_ctrl
from tripod_target_builder import build_targets_with_scale


def clamp01(x):
    return max(0.0, min(1.0, x))


def _sid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise ValueError(f"site not found: {name}")
    return i


def _dist(a, b):
    return np.linalg.norm(a - b)


def _area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _palm_frame(pff, pmf, prf):
    """
    掌坐标系（每帧更新，但在掌局部定义固定）
    x: ff->mf
    z: 掌法向
    y: z×x
    """
    x = _norm(pmf - pff)
    z = _norm(np.cross(pmf - pff, prf - pff))
    y = _norm(np.cross(z, x))
    return x, y, z


def _thumb_plane_error(th, ff, mf, pff, pmf, prf):
    """
    你要求的约束：
    - 平面经过 ff/mf 的ref中点
    - 平面法向使用掌固定方向（y_palm）
    - 拇指点到该平面的有符号距离 -> 0
    """
    c = 0.5 * (ff + mf)  # 中点
    _, y_palm, _ = _palm_frame(pff, pmf, prf)
    n_plane = y_palm
    e_plane = np.dot(th - c, n_plane)  # m
    return e_plane, n_plane


def execute_closed_loop(xml_path: str, hk, dt=0.01, realtime=True):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    act_idx = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(model.nu)
    }

    # ---- site ids ----
    id_th = _sid(model, "th_j1_ref_site")
    id_ff = _sid(model, "ff_j1_ref_site")
    id_mf = _sid(model, "mf_j1_ref_site")

    id_pff = _sid(model, "palm_contact_ff")
    id_pmf = _sid(model, "palm_contact_mf")
    id_prf = _sid(model, "palm_contact_rf")

    # ---- 尺度标定: palm ff-mf 对齐真实20mm ----
    d_xml_ffmf_mm = _dist(data.site_xpos[id_pff], data.site_xpos[id_pmf]) * 1000.0
    s = d_xml_ffmf_mm / 20.0
    targets = build_targets_with_scale(s)
    print(f"[scale] palm ff-mf xml={d_xml_ffmf_mm:.2f}mm, real=20mm, s={s:.4f}")

    stages = [
        ("shape_start", hk.shape_start, hk.T_start_pre),
        ("preshape",    hk.preshape,    hk.T_pre_enc),
        ("enclosure",   hk.enclosure,   hk.T_enc_final),
        ("final",       hk.final,       60),
    ]

    # ---- gains (保守) ----
    k_close = 1.0
    k_ap = 0.7
    k_opp = 0.5
    k_plane = 7.0      # 拇指到“固定方向平面”纠偏
    k_norm = 0.018     # 三角法向与掌法向对齐
    deadband_mm = 1.8

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # warmup
        for _ in range(80):
            mujoco.mj_step(model, data)
            viewer.sync()

        for sname, hframe, T in stages:
            t = targets[sname]
            u0 = map_human_to_synergy(hframe)

            # 防闭死上限
            cap = {
                "thumb_flex": 0.62,
                "index_flex": 0.58,
                "middle_flex": 0.58,
                "thumb_oppose": 0.72,
            }

            stable_count = 0

            for step in range(T):
                # current points
                th = data.site_xpos[id_th].copy()
                ff = data.site_xpos[id_ff].copy()
                mf = data.site_xpos[id_mf].copy()

                pff = data.site_xpos[id_pff].copy()
                pmf = data.site_xpos[id_pmf].copy()
                prf = data.site_xpos[id_prf].copy()

                # basic geom
                d_ti = _dist(th, ff)
                d_tm = _dist(th, mf)
                d_im = _dist(ff, mf)
                area = _area(th, ff, mf)

                d_th_p = _dist(th, pmf)
                d_ff_p = _dist(ff, pff)
                d_mf_p = _dist(mf, pmf)

                # primary errors
                e_pair = ((d_ti - t.d_ti) + (d_tm - t.d_tm) + (d_im - t.d_im)) / 3.0
                e_area = area - t.area_tip
                e_palm = ((d_th_p - t.d_th_p) + (d_ff_p - t.d_ff_p) + (d_mf_p - t.d_mf_p)) / 3.0
                e_close = 1.0 * e_pair + 0.25 * e_area + 1.1 * e_palm

                # ---- 约束1：拇指应落在“过ff/mf中点、法向为掌固定方向”的平面附近 ----
                e_plane, n_plane = _thumb_plane_error(th, ff, mf, pff, pmf, prf)

                # ---- 约束2：tip三角法向 ~ 掌法向 ----
                _, _, z_palm = _palm_frame(pff, pmf, prf)
                n_tip = _norm(np.cross(ff - th, mf - th))
                e_norm = 1.0 - abs(np.dot(n_tip, z_palm))  # 0最好

                # deadband
                e_pair_mm = e_pair * 1000.0
                e_palm_mm = e_palm * 1000.0
                if abs(e_pair_mm) < deadband_mm and abs(e_palm_mm) < deadband_mm:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= 20:
                    du_flex = 0.0
                    du_ap = 0.0
                    du_opp = 0.0
                else:
                    du_flex = np.clip(k_close * e_close, -0.015, 0.015)
                    du_ap = np.clip(-k_ap * e_pair, -0.015, 0.015)
                    du_opp = np.clip(k_opp * (d_th_p - t.d_th_p), -0.012, 0.012)

                    # 平面纠偏：默认负号让 e_plane -> 0
                    du_opp_plane = np.clip(+k_plane * e_plane, -0.018, 0.018)
                    du_opp += du_opp_plane

                    # 法向纠偏：法向不对齐时，略微收敛aperture
                    du_ap += np.clip(-k_norm * e_norm, -0.008, 0.008)

                # compose u
                thumb_flex = clamp01(u0.thumb_flex + 0.35 * du_flex)
                index_flex = clamp01(u0.index_flex + 0.45 * du_flex)
                middle_flex = clamp01(u0.middle_flex + 0.45 * du_flex)
                thumb_oppose = clamp01(u0.thumb_oppose + du_opp)
                aperture = clamp01(u0.aperture + du_ap)

                # caps
                thumb_flex = min(cap["thumb_flex"], thumb_flex)
                index_flex = min(cap["index_flex"], index_flex)
                middle_flex = min(cap["middle_flex"], middle_flex)
                thumb_oppose = min(cap["thumb_oppose"], thumb_oppose)
                aperture = max(0.20, aperture)

                u = type(u0)(
                    thumb_flex=thumb_flex,
                    thumb_oppose=thumb_oppose,
                    index_flex=index_flex,
                    middle_flex=middle_flex,
                    aperture=aperture,
                )

                ctrl = decode_u_to_ctrl(u).ctrl
                for name, val in ctrl.items():
                    if name in act_idx:
                        data.ctrl[act_idx[name]] = val

                mujoco.mj_step(model, data)
                viewer.sync()
                if realtime:
                    time.sleep(dt)

                # 可选调试输出（每10步）
                if step % 10 == 0 or step % 10 == 5:
                    print(f"[{sname}] e_plane(mm)={e_plane*1000:.2f}, e_norm={e_norm:.3f}, e_pair(mm)={e_pair_mm:.2f}")
