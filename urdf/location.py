import mujoco

m = mujoco.MjModel.from_xml_path("dexhand_lh_mjcf.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

for name in ["left_virtual_link_A3_link", "right_virtual_link_A3_link"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    print(name, d.xpos[bid])
