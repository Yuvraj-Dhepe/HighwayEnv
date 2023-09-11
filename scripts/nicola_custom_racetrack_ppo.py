import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env.envs.nicola_custom_env


TRAIN = True

if __name__ == '__main__':
    n_cpu = 12
    batch_size = 128
    env = make_vec_env("nicola_custom_env", n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed=7113)
    #env = gym.make("nicola_custom_env")
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=20,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log=r"D:\Documents\GitHub\HighwayEnvGroup\tensorboard")
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e6))
        model.save(r"D:\Documents\GitHub\HighwayEnvGroup/Models/model_v4")
        del model

    # Run the algorithm
    model = PPO.load(r"D:\Documents\GitHub\HighwayEnvGroup/models_final_tests/model_v4", env=env)

    env = gym.make("nicola_custom_env", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="D:\Documents\GitHub\HighwayEnvGroup/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            print(reward)
            # Render
            env.render()

    env.close()