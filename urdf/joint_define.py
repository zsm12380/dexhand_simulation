import mujoco
m = mujoco.MjModel.from_xml_path("dexhand_lh_mjcf.xml")
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i), m.jnt_type[i], m.jnt_range[i])
