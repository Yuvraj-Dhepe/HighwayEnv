from gymnasium.wrappers import RecordVideo

from highway_env.envs.nicola_custom_env import *
from highway_env.envs.myrt import *
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# Environment
import gymnasium as gym

if __name__ == '__main__':
    n_cpu = 12
    batch_size = 64
    # env = make_vec_env("nicola_racetrack_v2", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    env_dqn = gym.make("nicola_racetrack_v2")

    # Create the model
    model_dqn = DQN('MlpPolicy',
                    env_dqn,
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
                    tensorboard_log=r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\tensorboard/")

    print("Started Training")
    model_dqn.learn(total_timesteps=int(30e4), progress_bar=True)
    model_dqn.save(r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\models/nicola_dqn_300k_steps_model_discrete_actions")

    env_ppo = gym.make("nicola_racetrack_v2")
    model_ppo = PPO("MlpPolicy",
                    env_ppo,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.9,
                    verbose=2,
                    tensorboard_log=r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\tensorboard/")

    model_ppo.learn(total_timesteps=int(30e4), progress_bar=True)
    model_ppo.save(r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\models\nicola_ppo_300k_steps_model_discrete_actions")
    # Run the algorithm

    env_dqn = gym.make('nicola_racetrack_v2', render_mode="rgb_array")
    env_ppo = gym.make('nicola_racetrack_v2', render_mode="rgb_array")
    env_dqn = DQN.load(r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\models\nicola_dqn_300k_steps_model_discrete_actions", env=env_dqn)
    model_ppo = PPO.load(r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\models/nicola_ppo_300k_steps_model_discrete_actions", env=env_ppo)


    env_dqn = RecordVideo(env_dqn, video_folder=r"C:\Users\nic-e\OneDrive\UNI Potsdam\Lectures\Dynamic Programming and Reinforcement Learning\Reinforcement Learning Group Project\videos", episode_trigger=lambda e: True)

    env_dqn.unwrapped.set_record_video_wrapper(env_dqn)

    for video in range(10):
        done = truncated = False
        obs, info = env_dqn.reset()
        while not (done or truncated):
            # Predict
            action, _states = env_dqn.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env_dqn.step(action)
            # Render
            env_dqn.render()
    env_dqn.close()

