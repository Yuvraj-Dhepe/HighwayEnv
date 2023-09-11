from gymnasium.wrappers import RecordVideo

from highway_env.envs.nicola_custom_env import *
from highway_env.envs.myrt import *
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# Environment
import gymnasium as gym

if __name__ == '__main__':
    algorthm_names = ["PPO", "SAC", "DQN"]

    n_cpu = 8
    batch_size = 64
    env_ppo = make_vec_env("custom_reward_test_env", n_envs=n_cpu,  seed=7113)

    # Create the model
    model_ppo = PPO("MlpPolicy",
                env_ppo,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log="./tensorboard")

    env_dqn = make_vec_env("custom_reward_test_env", n_envs=n_cpu, seed=7113)
    model_dqn = DQN(
        "MlpPolicy",
        env_dqn,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=batch_size,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="./tensorboard"
    )

    model_ppo.learn(total_timesteps=int(1e6), progress_bar=False)
    model_ppo.save(r"./models_final_tests/model_ppo_custom")

    model_dqn.learn(total_timesteps=int(1e6))
    model_dqn.save("./models_final_tests/model_dqn_custom")
