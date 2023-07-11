import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from highway_env.envs.simple_env import *
import highway_env


TRAIN = True

if __name__ == '__main__':
    # Create the environment
    env = gym.make("rt-simple-v0")#,render_mode = 'human')
    obs, info = env.reset()

    # Create the model
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
                tensorboard_log="./y_models/logs/cnn_dqn")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("./y_models/cnn_dqn/")
        del model

    # Run the trained model and record video
    # model = DQN.load("./y_models/dqn.zip", env=env)
    # env = RecordVideo(env, video_folder="./y_models/vids", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)
    # env.configure({"simulation_frequency": 30})  # Higher FPS for rendering

    # for videos in range(10):
    #     done = truncated = False
    #     obs, info = env.reset()
    #     while not (done or truncated):
    #         # Predict
    #         action, _states = model.predict(obs, deterministic=True)
    #         # Get reward
    #         obs, reward, done, truncated, info = env.step(action)
    #         # Render
    #         env.render()
    env.close()