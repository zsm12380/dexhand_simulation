from dexhand_env_named import DexHandGraspEnv


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

    obs, info = env.reset()
    print("obs shape:", obs.shape)

    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"t={t:03d}, reward={reward:.4f}, "
            f"contacts=({info['th_contact']}, {info['ff_contact']}, {info['mf_contact']}), "
            f"tripod_mean_dist={info['tripod_mean_dist']:.4f}, "
            f"shape={info['r_tripod_shape']:.4f}"
        )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
