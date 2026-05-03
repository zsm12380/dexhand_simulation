import mujoco

model = mujoco.MjModel.from_xml_path("dexhand_lh_rl.xml")
for i in range(model.ngeom):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
