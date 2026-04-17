import mujoco

m = mujoco.MjModel.from_xml_path("dexhand_lh_mjcf.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("Loaded OK")
print("nbody:", m.nbody, "njnt:", m.njnt, "neq:", m.neq)
