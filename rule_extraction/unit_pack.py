# rule_extraction/unit_check.py
import mujoco
import numpy as np
from tripod_exec_config import DEFAULT_XML

def sid(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise ValueError(f"site not found: {name}")
    return i

def dist(data, i, j):
    return np.linalg.norm(data.site_xpos[i] - data.site_xpos[j])

def main():
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    names = [
        "th_j1_ref_site", "ff_j1_ref_site", "mf_j1_ref_site",
        "palm_contact_ff", "palm_contact_mf", "palm_contact_rf", "palm_contact_lf",
    ]
    ids = {n: sid(model, n) for n in names}

    pairs = [
        ("palm_contact_ff", "palm_contact_mf"),
        ("th_j1_ref_site", "ff_j1_ref_site"),
        ("th_j1_ref_site", "mf_j1_ref_site"),
        ("ff_j1_ref_site", "mf_j1_ref_site"),
        ("ff_j1_ref_site", "palm_contact_ff"),
        ("mf_j1_ref_site", "palm_contact_mf"),
        ("th_j1_ref_site", "palm_contact_mf"),
    ]

    print("=== Unit check (MuJoCo internal is meter) ===")
    for a, b in pairs:
        d_m = dist(data, ids[a], ids[b])
        d_mm = d_m * 1000.0
        print(f"{a:18s} <-> {b:18s}: {d_m:.6f} m  ({d_mm:.2f} mm)")

    # 你给的真实根部距离: ff-mf ≈ 20mm
    d_xml_ffmf_mm = dist(data, ids["palm_contact_ff"], ids["palm_contact_mf"]) * 1000.0
    s = d_xml_ffmf_mm / 20.0
    print(f"\nscale s = xml/real = {d_xml_ffmf_mm:.2f}/20.00 = {s:.4f}")

if __name__ == "__main__":
    main()
