from stable_baselines3 import SAC
from dexhand_env import DexHandGraspEnv


def main():
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

    model = SAC.load("dexhand_tripod_named_sac_final", env=env)

    for ep in range(5):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        step_id = 0

        print(f"\n===== Episode {ep} =====")

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            print(
                f"step={step_id:03d}, "
                f"reward={reward:.4f}, "
                f"contact_sum={info['contact_sum']}, "
                f"tripod_mean_dist={info['tripod_mean_dist']:.4f}, "
                f"shape={info['r_tripod_shape']:.4f}, "
                f"success={info['success']}"
            )
            step_id += 1

        print(f"Episode reward = {ep_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
