import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from dexhand_env import DexHandGraspEnv


def make_env():
    env = DexHandGraspEnv(
        model_path="dexhand_lh_rl.xml",
        workspace_path="workspace_tripod.npz",
        object_geom_name="object_geom",
        frame_skip=5,
        max_steps=220,
        action_type="delta",
        delta_scale=0.002,
    )
    return Monitor(env)


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/tb", exist_ok=True)
    os.makedirs("logs/best_model", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    env = make_env()
    check_env(env, warn=True)

    eval_env = make_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="dexhand_tripod_workspace_sac",
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
        learning_starts=8000,
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
        total_timesteps=300000,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=False,
    )

    model.save("dexhand_tripod_workspace_sac_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
