import os, mujoco

urdf = "/home/zhangsm/ros2_ws/src/dexhand_description_v32/urdf/dexhand_lh.urdf"

# 切换到 urdf 所在目录，保证 ../meshes 能解析
os.chdir(os.path.dirname(urdf))

model = mujoco.MjModel.from_xml_path(urdf)
mujoco.mj_saveLastXML("dexhand_lh_mjcf.xml", model)
print("转换成功")
