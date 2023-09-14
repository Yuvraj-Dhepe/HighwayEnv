from gymnasium.wrappers import RecordVideo

from highway_env.envs.nicola_custom_env import *
from highway_env.envs.myrt import *
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# Environment
import gymnasium as gym

TRAIN = False
RECORD = True

if __name__ == '__main__':
    n_cpu = 12
    batch_size = 64
    env_ppo = make_vec_env("custom_reward_test_env", n_envs=n_cpu, seed=7113)

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
                tensorboard_log="./new_tensorboard")


    env_dqn = make_vec_env("custom_reward_test_env", n_envs=n_cpu, seed=7113)
    model_dqn = DQN(
        "MlpPolicy",
        env_dqn,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=300,
        batch_size=64,
        gamma=0.6,
        train_freq=1,
        gradient_steps=-1,
        target_update_interval=30,
        verbose=1,
        tensorboard_log="./new_tensorboard"
    )

    if TRAIN:

        model_ppo.learn(total_timesteps=int(1e6), progress_bar=False)
        model_ppo.save(r"./models_final_tests/model_ppo_custom_v2")

        model_dqn.learn(total_timesteps=int(1e6))
        model_dqn.save("./models_final_tests/model_dqn_custom_v2")

    if RECORD:
        env_dqn = gym.make('rt-y-v0', render_mode="rgb_array")
        #env_ppo = gym.make('rt-y-v0', render_mode="rgb_array")
        model_dqn = DQN.load(r"./models/mlp_dqn6", env=env_dqn)
        #model_ppo = PPO.load(r"./models_final_tests/model_ppo_custom_v2", env=env_ppo)

        env = RecordVideo(env_dqn, video_folder="D:\Documents\GitHub\HighwayEnvGroup/videos/dqn6", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)

        for video in range(10):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model_dqn.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action)
                # Render
                env.render()

        env.close()
