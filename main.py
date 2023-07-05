import gymnasium as gym
from stable_baselines3 import DQN, PPO
from highway_env.envs.cust_env_m import *
from gymnasium.wrappers import RecordVideo
import tensorboard
TRAIN = True


if __name__ == '__main__':
    # Create the environment
    env = gym.make("rt-m-v0") #render_mode="human")
    obs, info = env.reset()

    # Create the model
    model = PPO('MlpPolicy',
            env,
            policy_kwargs=dict(net_arch=(dict(pi=[256, 256], vf=[256, 256]))),
            learning_rate=5e-4,
            n_steps=64 * 12 // 12,
            batch_size=64,
            n_epochs=10,
            gamma=0.9,
            verbose=2,
            tensorboard_log=r"C:\Users\asus\Documents\GitHub\HighwayEnvReal\models"
            )
    if TRAIN:
        #tensorboard --logdir ./C:/Users/asus/Documents/GitHub/HighwayEn
        model.learn(total_timesteps=int(1e5))
        model.save(r"C:\Users\asus\Documents\GitHub\HighwayEnvReal\models\model_addv")
        del model

    # Run the algorithm
    model = PPO.load(r"C:\Users\asus\Documents\GitHub\HighwayEnvReal\models\model_addv.zip", env=env)

    env = gym.make("rt-m-v0", render_mode='rgb_array')
    env = RecordVideo(env, video_folder=r"C:\Users\asus\Documents\GitHub\HighwayEnvReal\models\model_addv", episode_trigger=lambda e: True)
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
