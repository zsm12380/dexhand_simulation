import numpy as np
import mujoco
import itertools

MODEL_PATH = "dexhand_lh_rl.xml"
OUT_PATH = "workspace_tripod.npz"

SAMPLES_PER_FINGER = 1200

FINGER_CFG = {
    "th": {
        "act": ["THJ4", "THJ3", "THJ2", "THJ1"],
        "site": "th_j1_ref_site",
        "body_for_normal": "THJ1",
    },
    "ff": {
        "act": ["FFJ4", "FFJ3", "FFJ2", "FFJ1"],
        "site": "ff_j1_ref_site",
        "body_for_normal": "FFJ1",
    },
    "mf": {
        "act": ["MFJ4", "MFJ3", "MFJ2", "MFJ1"],
        "site": "mf_j1_ref_site",
        "body_for_normal": "MFJ1",
    },
}


def actuator_id_map(model):
    mp = {}
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n is not None:
            mp[n] = i
    return mp


def default_ctrl(model):
    c = np.zeros(model.nu, dtype=np.float32)
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        c[i] = 0.5 * (lo + hi)
    return c


def sample_workspace(model, data):
    act_map = actuator_id_map(model)
    base_ctrl = default_ctrl(model)

    out = {}

    for f, cfg in FINGER_CFG.items():
        pts = []
        nrm = []

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["site"])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["body_for_normal"])

        if site_id < 0 or body_id < 0:
            raise ValueError(f"{f}: site/body 不存在，请检查命名")

        for _ in range(SAMPLES_PER_FINGER):
            data.ctrl[:] = base_ctrl

            # 只随机该指关节
            for an in cfg["act"]:
                aid = act_map[an]
                lo, hi = model.actuator_ctrlrange[aid]
                data.ctrl[aid] = np.random.uniform(lo, hi)

            mujoco.mj_forward(model, data)

            p = data.site_xpos[site_id].copy()
            R = data.xmat[body_id].reshape(3, 3).copy()

            # 法向近似：取该body局部y轴（你也可改z轴）
            n = R[:, 1].copy()
            n = n / (np.linalg.norm(n) + 1e-9)

            pts.append(p)
            nrm.append(n)

        out[f"{f}_pos"] = np.asarray(pts, dtype=np.float32)
        out[f"{f}_nrm"] = np.asarray(nrm, dtype=np.float32)

        print(f"[{f}] samples={len(pts)}")

    return out


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ws = sample_workspace(model, data)
    np.savez_compressed(OUT_PATH, **ws)
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
