#!/usr/bin/env python3
import argparse
import time
import numpy as np
import mujoco


def safe_name(model, objtype, idx):
    n = mujoco.mj_id2name(model, objtype, idx)
    return n if n is not None else f"<unnamed:{idx}>"


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def world_to_local(xpos, xquat, p_world):
    return quat_rotate(quat_conj(xquat), p_world - xpos)


def local_to_world(xpos, xquat, p_local):
    return xpos + quat_rotate(xquat, p_local)


def print_body_pose(m, d, body_name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        print(f"[body] {body_name}: MISSING")
        return None, None

    pos = d.xpos[bid].copy()
    quat = d.xquat[bid].copy()
    print(f"[body] {body_name}")
    print(f"       world pos : {pos}")
    print(f"       world quat: {quat}")
    return pos, quat


def analyze_connect(m, d, eq_name):
    eqid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if eqid == -1:
        print(f"\n[connect] {eq_name}: MISSING")
        return

    eq_type = int(m.eq_type[eqid])
    obj1 = int(m.eq_obj1id[eqid])
    obj2 = int(m.eq_obj2id[eqid])
    anchor = m.eq_data[eqid, :3].copy()

    name1 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj1)
    name2 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj2)

    print(f"\n=== Connect analysis: {eq_name} ===")
    print(f"type   = {eq_type}")
    print(f"body1  = {name1}")
    print(f"body2  = {name2}")
    print(f"anchor = {anchor}")

    p1 = d.xpos[obj1].copy()
    q1 = d.xquat[obj1].copy()
    p2 = d.xpos[obj2].copy()
    q2 = d.xquat[obj2].copy()

    a1_local = world_to_local(p1, q1, anchor)
    a2_local = world_to_local(p2, q2, anchor)

    print(f"anchor in {name1} local = {a1_local}")
    print(f"anchor in {name2} local = {a2_local}")

    a1_world_recon = local_to_world(p1, q1, a1_local)
    a2_world_recon = local_to_world(p2, q2, a2_local)

    print(f"reconstructed world from {name1} = {a1_world_recon}")
    print(f"reconstructed world from {name2} = {a2_world_recon}")

    d1 = np.linalg.norm(anchor - p1)
    d2 = np.linalg.norm(anchor - p2)
    print(f"|anchor - {name1}_origin| = {d1:.8f} m")
    print(f"|anchor - {name2}_origin| = {d2:.8f} m")

    e1 = np.linalg.norm(a1_world_recon - anchor)
    e2 = np.linalg.norm(a2_world_recon - anchor)
    e12 = np.linalg.norm(a1_world_recon - a2_world_recon)

    print(f"reconstruction err on body1 = {e1:.12e} m")
    print(f"reconstruction err on body2 = {e2:.12e} m")
    print(f"point mismatch body1/body2   = {e12:.12e} m")


def analyze_sites(m, d, site1_name, site2_name):
    sid1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site1_name)
    sid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site2_name)

    if sid1 == -1 or sid2 == -1:
        print(f"\n[site] missing: {site1_name} or {site2_name}")
        return None

    p1 = d.site_xpos[sid1].copy()
    p2 = d.site_xpos[sid2].copy()

    diff = p1 - p2
    dist = np.linalg.norm(diff)
    axis_names = ["x", "y", "z"]
    major_axis = int(np.argmax(np.abs(diff)))

    print(f"\n=== Site analysis: {site1_name} ↔ {site2_name} ===")
    print(f"{site1_name} world pos = {p1}")
    print(f"{site2_name} world pos = {p2}")
    print(f"diff (site1 - site2) = {diff}")
    print(f"distance             = {dist:.6f} m")
    print(f"largest component    = {axis_names[major_axis]} ({diff[major_axis]:+.6f} m)")

    print("\nAdjustment hint:")
    print(f"  If you are tuning {site1_name}, move it roughly opposite to diff.")
    print(f"  Suggested first try on local axes: prioritize {axis_names[major_axis]}.")

    return {
        "site1": site1_name,
        "site2": site2_name,
        "p1": p1,
        "p2": p2,
        "diff": diff,
        "dist": dist,
        "major_axis": axis_names[major_axis],
    }


def print_site_local_info(m, d, site_name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid == -1:
        print(f"[site-local] {site_name}: MISSING")
        return

    bodyid = int(m.site_bodyid[sid])
    body_name = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, bodyid)
    local_pos = m.site_pos[sid].copy()
    world_pos = d.site_xpos[sid].copy()

    print(f"\n[site-local] {site_name}")
    print(f"  attached body : {body_name}")
    print(f"  local pos     : {local_pos}")
    print(f"  world pos     : {world_pos}")


def set_connect_active(m, eq_name, active):
    eqid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if eqid == -1:
        print(f"[connect-toggle] {eq_name}: MISSING")
        return
    m.eq_active[eqid] = 1 if active else 0
    print(f"[connect-toggle] {eq_name} -> {'ON' if active else 'OFF'}")


def load_and_report(xml_path, disable_connect=False):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    if disable_connect:
        set_connect_active(m, "left_parallel_close", False)
        set_connect_active(m, "right_parallel_close", False)

    mujoco.mj_forward(m, d)

    print("\n=== Load OK ===")
    print(f"model: {xml_path}")
    print(f"nbody={m.nbody}, njnt={m.njnt}, nu={m.nu}, neq={m.neq}")

    required_bodies = [
        "left_virtual_link_A3_link",
        "right_virtual_link_A3_link",
        "left_ear_link",
        "right_ear_link",
        "palm_wrist_connector_link",
    ]
    required_sites = [
        "left_a3_site",
        "left_ear_site",
        "right_a3_site",
        "right_ear_site",
    ]

    for b in required_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"[body] {b}: {'OK' if bid != -1 else 'MISSING'}")

    for s in required_sites:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
        print(f"[site] {s}: {'OK' if sid != -1 else 'MISSING'}")

    print("\n=== Equality constraints ===")
    for i in range(m.neq):
        eq_name = safe_name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        eq_type = int(m.eq_type[i])
        obj1 = int(m.eq_obj1id[i])
        obj2 = int(m.eq_obj2id[i])
        active = int(m.eq_active[i])

        b1 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj1) if obj1 >= 0 else "None"
        b2 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj2) if obj2 >= 0 else "None"
        print(
            f"  [{i}] name={eq_name}, type={eq_type}, active={active}, "
            f"obj1={b1}, obj2={b2}, data0-2={m.eq_data[i,:3]}"
        )

    print("\n=== Actuators ===")
    for i in range(m.nu):
        print(f"  [{i}] {safe_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
    if m.nu == 0:
        print("  (none)")

    print("\n=== Key body poses ===")
    print_body_pose(m, d, "left_ear_link")
    print_body_pose(m, d, "right_ear_link")
    print_body_pose(m, d, "left_virtual_link_A3_link")
    print_body_pose(m, d, "right_virtual_link_A3_link")

    print_site_local_info(m, d, "left_a3_site")
    print_site_local_info(m, d, "left_ear_site")
    print_site_local_info(m, d, "right_a3_site")
    print_site_local_info(m, d, "right_ear_site")

    analyze_connect(m, d, "left_parallel_close")
    analyze_connect(m, d, "right_parallel_close")

    left_result = analyze_sites(m, d, "left_a3_site", "left_ear_site")
    right_result = analyze_sites(m, d, "right_a3_site", "right_ear_site")

    return m, d, left_result, right_result


def pick_drive_actuators(model):
    ids = []
    for i in range(model.nu):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).lower()
        if ("linear" in name) or ("rod" in name):
            ids.append(i)
    if len(ids) == 0 and model.nu > 0:
        ids = list(range(min(2, model.nu)))
    return ids


def periodic_site_report(m, d, every_sec=0.5):
    last = getattr(periodic_site_report, "_last_t", None)
    now = d.time
    if last is None or (now - last) >= every_sec:
        periodic_site_report._last_t = now
        left = analyze_sites(m, d, "left_a3_site", "left_ear_site")
        right = analyze_sites(m, d, "right_a3_site", "right_ear_site")
        return left, right
    return None, None


def run_viewer(m, d, seconds=20.0, report_sites=True):
    import mujoco.viewer

    drive_ids = pick_drive_actuators(m)
    print(f"\nDrive actuators: {drive_ids}")

    t0 = time.time()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and (time.time() - t0 < seconds):
            t = d.time

            if m.nu > 0:
                d.ctrl[:] = 0.0
                for k, i in enumerate(drive_ids):
                    if m.actuator_ctrllimited[i]:
                        lo, hi = m.actuator_ctrlrange[i]
                        mid = 0.5 * (lo + hi)
                        amp = 0.2 * (hi - lo)
                    else:
                        mid, amp = 0.0, 0.1
                    d.ctrl[i] = mid + amp * np.sin(2*np.pi*0.5*t + k*np.pi)

            mujoco.mj_step(m, d)

            if report_sites:
                periodic_site_report(m, d, every_sec=0.8)

            viewer.sync()
            time.sleep(m.opt.timestep)

    print("Viewer finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="path to dexhand_lh_mjcf.xml")
    parser.add_argument("--no-viewer", action="store_true", help="only load/check, no GUI")
    parser.add_argument("--seconds", type=float, default=20.0, help="viewer run time")
    parser.add_argument(
        "--disable-connect",
        action="store_true",
        help="temporarily disable left/right parallel connect for raw site-offset inspection",
    )
    parser.add_argument(
        "--quiet-sites",
        action="store_true",
        help="do not periodically print site error during viewer run",
    )
    args = parser.parse_args()

    m, d, _, _ = load_and_report(args.xml, disable_connect=args.disable_connect)

    if not args.no_viewer:
        run_viewer(m, d, seconds=args.seconds, report_sites=not args.quiet_sites)


if __name__ == "__main__":
    main()
