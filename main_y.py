import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from highway_env.envs.simple_env import *
import highway_env
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.env_util import make_vec_env 
import os

n_cpu = 8

TRAIN = True
SAVE = False
if __name__ == '__main__':
    # # Create the environment
    #env = gym.make("rt-y-v0",render_mode = 'human')
    env = make_vec_env('rt-y-v0',n_envs = n_cpu,vec_env_cls = SubprocVecEnv,seed = 7113)
    # make_vec_env("rt-y-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv,seed = 7113)
    env.reset()
    
    # # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="./y_models/logs/mlp_dqn")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2.5e4))
        model.save("./y_models/DQN_models/")
        del model

    if SAVE: 
        # Run the trained model and record video
        env = make_vec_env("rt-y-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv,seed=7113)
        obs, info = env.reset()
        model = DQN.load("./y_models/DQN_models/mlp_dqn.zip", env=env)
        env = RecordVideo(env, video_folder="./y_models/vids", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": 30})  # Higher FPS for rendering
        
        for videos in range(10):
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
