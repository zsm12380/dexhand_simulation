#!/usr/bin/env python3
import argparse
import time
import numpy as np
import mujoco


def safe_name(model, objtype, idx):
    n = mujoco.mj_id2name(model, objtype, idx)
    return n if n is not None else f"<unnamed:{idx}>"


def quat_conj(q):
    # q = [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    # Hamilton product, q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_rotate(q, v):
    # rotate vector v by unit quaternion q
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

    # 把世界 anchor 反算到两个 body 各自的 local frame
    a1_local = world_to_local(p1, q1, anchor)
    a2_local = world_to_local(p2, q2, anchor)

    print(f"anchor in {name1} local = {a1_local}")
    print(f"anchor in {name2} local = {a2_local}")

    # 再正算回世界，检查数值一致性
    a1_world_recon = local_to_world(p1, q1, a1_local)
    a2_world_recon = local_to_world(p2, q2, a2_local)

    print(f"reconstructed world from {name1} = {a1_world_recon}")
    print(f"reconstructed world from {name2} = {a2_world_recon}")

    # 关键诊断 1：
    # 如果 body1/body2 当前真在同一个闭环点上，那么
    # “body1 上这个 local 点的世界坐标” 和 “body2 上这个 local 点的世界坐标”
    # 应该和 anchor 对齐，而且二者彼此重合。
    #
    # 在当前 qpos 下，这里先计算 body 原点到 anchor 的距离，帮助判断 anchor 是否离 body 很离谱
    d1 = np.linalg.norm(anchor - p1)
    d2 = np.linalg.norm(anchor - p2)
    print(f"|anchor - {name1}_origin| = {d1:.8f} m")
    print(f"|anchor - {name2}_origin| = {d2:.8f} m")

    # 关键诊断 2：
    # 通过 mj_forward 后，MuJoCo 会根据当前 qpos 给出 body 世界姿态。
    # connect 想要的是：两个 body 上各自对应的 anchor 点重合。
    #
    # 我们把“同一个世界 anchor”映射到两个 body 的 local，再映射回世界，
    # 如果初始模型几何/层级一致，这两个世界点应该都正好等于 anchor。
    e1 = np.linalg.norm(a1_world_recon - anchor)
    e2 = np.linalg.norm(a2_world_recon - anchor)
    e12 = np.linalg.norm(a1_world_recon - a2_world_recon)

    print(f"reconstruction err on body1 = {e1:.12e} m")
    print(f"reconstruction err on body2 = {e2:.12e} m")
    print(f"point mismatch body1/body2   = {e12:.12e} m")

    # 再给一个更直观的提示
    print("diagnosis:")
    print("  - 上面三个 reconstruction/mismatch 正常应接近 0（浮点误差量级）")
    print("  - 如果你视觉上仍觉得 connect 没接住，问题通常不是 XML 里没读到 connect，")
    print("    而是初始 qpos 下闭环几何不一致，仿真一开始在被约束力拉回去。")


def load_and_report(xml_path):
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
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

    for b in required_bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"[body] {b}: {'OK' if bid != -1 else 'MISSING'}")

    print("\n=== Equality constraints ===")
    for i in range(m.neq):
        eq_name = safe_name(m, mujoco.mjtObj.mjOBJ_EQUALITY, i)
        eq_type = int(m.eq_type[i])
        obj1 = int(m.eq_obj1id[i])
        obj2 = int(m.eq_obj2id[i])

        b1 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj1) if obj1 >= 0 else "None"
        b2 = safe_name(m, mujoco.mjtObj.mjOBJ_BODY, obj2) if obj2 >= 0 else "None"
        print(f"  [{i}] name={eq_name}, type={eq_type}, obj1={b1}, obj2={b2}, data0-2={m.eq_data[i,:3]}")

    print("\n=== Actuators ===")
    for i in range(m.nu):
        print(f"  [{i}] {safe_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
    if m.nu == 0:
        print("  (none)")

    print("\n=== Key body poses ===")
    print_body_pose(m, d, "left_ear_link")
    print_body_pose(m, d, "right_ear_link")




    analyze_connect(m, d, "left_parallel_close")
    analyze_connect(m, d, "right_parallel_close")

    return m, d


def pick_drive_actuators(model):
    ids = []
    for i in range(model.nu):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i).lower()
        if ("linear" in name) or ("rod" in name):
            ids.append(i)
    if len(ids) == 0 and model.nu > 0:
        ids = list(range(min(2, model.nu)))
    return ids


def run_viewer(m, d, seconds=20.0):
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
            viewer.sync()
            time.sleep(m.opt.timestep)

    print("Viewer finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="path to dexhand_lh_mjcf.xml")
    parser.add_argument("--no-viewer", action="store_true", help="only load/check, no GUI")
    parser.add_argument("--seconds", type=float, default=100, help="viewer run time")
    args = parser.parse_args()

    m, d = load_and_report(args.xml)
    if not args.no_viewer:
        run_viewer(m, d, seconds=args.seconds)


if __name__ == "__main__":
    main()
