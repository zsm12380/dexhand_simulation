import time
import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import SAC
from dexhand_env import DexHandGraspEnv


def main():
    # 1) 创建环境（与训练保持一致）
    env = DexHandGraspEnv(
        model_path="dexhand_lh_rl.xml",
        workspace_path="workspace_tripod.npz",
        object_geom_name="object_geom",
        frame_skip=5,
        max_steps=220,
        action_type="delta",
        delta_scale=0.002,
    )

    # 2) 加载模型（注意文件名要与你训练保存一致）
    # SAC.save("xxx") 生成的是 xxx.zip
    model = SAC.load("dexhand_tripod_workspace_sac_final.zip", env=env)

    # 3) reset
    obs, info = env.reset()

    episode_id = 0
    step_id = 0
    ep_reward = 0.0

    print("启动 viewer，可直接观察抓取过程。")
    print("关闭 viewer 窗口即可退出。")

    # 4) 启动 MuJoCo viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.2
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.15])

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            viewer.sync()
            time.sleep(0.01)

            print(
                f"[ep={episode_id:03d} step={step_id:03d}] "
                f"phase={info.get('phase', -1)} | "
                f"contact_sum={info.get('contact_sum', -1)} | "
                f"geom_err={info.get('geom_err', -1.0):.4f} | "
                f"normal_align={info.get('normal_align', -1.0):.3f} | "
                f"freeze_steps={info.get('freeze_steps', -1)} | "
                f"success={info.get('success', False)}"
            )

            step_id += 1

            if terminated or truncated:
                print(
                    f"===== Episode {episode_id} finished | "
                    f"reward={ep_reward:.4f} | "
                    f"success={info.get('success', False)} ====="
                )
                obs, info = env.reset()
                episode_id += 1
                step_id = 0
                ep_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
