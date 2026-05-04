import time
import numpy as np
import mujoco
import mujoco.viewer

from stable_baselines3 import SAC
from dexhand_env import DexHandGraspEnv


def main():
    # 1) 创建环境
    env = DexHandGraspEnv(
        model_path="dexhand_lh_rl.xml",
        object_geom_name="object_geom",
        thumb_geom_name="th_tip_geom",
        index_geom_name="ff_tip_geom",
        middle_geom_name="mf_tip_geom",
        frame_skip=5,
        max_steps=200,
        action_type="delta",
        delta_scale=0.03,
    )

    # 2) 加载模型
    # 你可以改成 best_model 路径
    model = SAC.load("dexhand_tripod_named_sac_final", env=env)

    # 3) reset
    obs, info = env.reset()

    episode_id = 0
    step_id = 0
    ep_reward = 0.0

    print("启动 viewer，可直接观察抓取过程。")
    print("关闭 viewer 窗口即可退出。")

    # 4) 启动 Mujoco viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 可选：设置相机
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.2
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.1])

        while viewer.is_running():
            # 用训练好的策略预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 环境 step
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # 刷新 viewer
            viewer.sync()

            # 控制播放速度（可选）
            time.sleep(0.01)

            # 打印调试信息
            print(
                f"[ep={episode_id:02d} step={step_id:03d}] "
                f"reward={reward:8.4f} | "
                f"contact_sum={info['contact_sum']} | "
                f"hold_steps={info.get('contact_hold_steps', 0)} | "
                f"tripod_mean_dist={info['tripod_mean_dist']:.4f} | "
                f"shape={info['r_tripod_shape']:.4f} | "
                f"success={info['success']}"
            )

            step_id += 1

            # 如果 episode 结束，就自动 reset 继续播放下一局
            if terminated or truncated:
                print(
                    f"===== Episode {episode_id} finished | "
                    f"ep_reward={ep_reward:.4f} ====="
                )
                obs, info = env.reset()
                episode_id += 1
                step_id = 0
                ep_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
