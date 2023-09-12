import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO
from highway_env.envs.simple_env import *
import highway_env
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.env_util import make_vec_env 
import os

n_cpu = 6

TRAIN = False
SAVE = True
if __name__ == '__main__':
    name = 'model_dqn_custom'
    save_path = "./y_models/DQN_models/"+name
    model = DQN if  'dqn' in name else PPO
    steps = int(1e6)
    # Train the model
    if TRAIN:
        if model == DQN:
            env = make_vec_env('rt-y-v0',n_envs = n_cpu,vec_env_cls = SubprocVecEnv,seed = 42)
            env.reset()
            # # Create the model
            model_dqn = model('MlpPolicy', env,
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
                        tensorboard_log="./y_models/logs/mlp_dqn")
            model_dqn.learn(total_timesteps=steps)
            model_dqn.save(save_path)
            del model_dqn
        
        elif model == PPO:
            print(f"Started Training PPO for {steps}")
            env = make_vec_env('rt-y-v0',n_envs = n_cpu,vec_env_cls = SubprocVecEnv,seed = 42)
            env.reset()
            model_ppo = model("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=(64 * 12) // n_cpu,
                batch_size=64,
                n_epochs=10,
                learning_rate=5e-4,
                gamma=0.9,
                verbose=2,
                tensorboard_log="./y_models/logs/mlp_dqn")
            model_ppo.learn(total_timesteps=steps, progress_bar=False)
            model_ppo.save(save_path)
            del model_ppo
            

    if SAVE: 
        env = gym.make('rt-y-v0', render_mode = 'rgb_array')
        obs, info = env.reset()
        loaded_model = model.load(save_path+'.zip', env=env)
        env = RecordVideo(env, video_folder="./y_models/vids/"+name, episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": 30})  # Higher FPS for rendering
        
        for videos in range(5):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = loaded_model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action)
                # Render
                env.render()
        env.close()
