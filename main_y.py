import gymnasium as gym
from stable_baselines3 import DQN, PPO
from highway_env.envs.cust_env_m import *
from gymnasium.wrappers import RecordVideo
TRAIN = True

if __name__ == '__main__':
    # # Create the environment
    # env = gym.make("rt-y-v0") #render_mode="human")
    # obs, info = env.reset()

    # # Create the model
    # model = PPO('MlpPolicy',
    #         env,
    #         policy_kwargs=dict(net_arch=(dict(pi=[256, 256], vf=[256, 256]))),
    #         learning_rate=5e-4,
    #         n_steps=64 * 12 // 12,
    #         batch_size=64,
    #         n_epochs=10,
    #         gamma=0.9,
    #         verbose=2,
    #         tensorboard_log="./y_models/logs/"
    #         )
    
    # model.learn(total_timesteps=int(1e5))
    # model.save("./y_models/y_v1_model")
    # env.close()
    
    #Run the algorithm
    env = gym.make("rt-y-v0", render_mode="human")
    obs, info = env.reset()
    model = PPO.load("./y_models/y_v1_model", env=env)
    # env = gym.make("rt-y-v0")
    env = RecordVideo(env, video_folder=r"y_models/vids", episode_trigger=lambda e: True)
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