import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env.envs.racetrack_env

import highway_env


TRAIN = False

if __name__ == '__main__':
    n_cpu = 6
    batch_size = 128
    env = gym.make("nicola_custom_env-v0")
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=4e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log=r"/Users/I518095/Documents/GitHub/HighwayEnv/tensorboard")
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2.5e4))
        model.save(r"/Users/I518095/Documents/GitHub/HighwayEnv/Models/model_v4")
        del model

    # Run the algorithm
    model = PPO.load("/Users/I518095/Documents/GitHub/HighwayEnv/Models/model_v1", env=env)

    env = gym.make("nicola_custom_env-v0", render_mode="human")
    env = RecordVideo(env, video_folder="/Users/I518095/Documents/GitHub/HighwayEnv/Videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()

    env.close()
å