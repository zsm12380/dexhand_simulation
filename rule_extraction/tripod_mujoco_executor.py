import time
import mujoco
import mujoco.viewer
from tripod_exec_types import PlannedTrajectory

def _actuator_index(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}

def execute_trajectory(xml_path: str, traj: PlannedTrajectory, realtime=True):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    act_idx = _actuator_index(model)

    for _ in range(100):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for f in traj.ctrl_seq:
            for name, val in f.ctrl.items():
                if name in act_idx:
                    data.ctrl[act_idx[name]] = val
            mujoco.mj_step(model, data)
            viewer.sync()
            if realtime:
                time.sleep(traj.dt)
