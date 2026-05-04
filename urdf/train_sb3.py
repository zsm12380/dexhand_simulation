import os

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from dexhand_env import DexHandGraspEnv


def make_env():
    env = DexHandGraspEnv(
        model_path="dexhand_lh_rl.xml",

        # 你需要保证物体也有这个名字
        object_geom_name="object_geom",

        # 这里改成你实际的 geom 名字
        thumb_geom_name="th_tip_geom",
        index_geom_name="ff_tip_geom",
        middle_geom_name="mf_tip_geom",

        # 如果你也给物体 body 命名了，可以填；否则不填也行
        # object_body_name="object",

        frame_skip=5,
        max_steps=200,

        # 推荐先用 delta 控制
        action_type="delta",
        delta_scale=0.03,
    )
    return Monitor(env)


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tb", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    env = make_env()
    check_env(env, warn=True)

    eval_env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="dexhand_tripod_named_sac"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./logs/tb/",
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    model.learn(
        total_timesteps=800000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=False,
    )

    model.save("dexhand_tripod_named_sac_final")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
