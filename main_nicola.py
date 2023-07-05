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
    env = make_vec_env("RaceTrackYuvraj", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO('MlpPolicy',
                env,
                policy_kwargs=dict(net_arch=(dict(pi=[256, 256], vf=[256, 256]))),
                learning_rate=5e-4,
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                gamma=0.9,
                verbose=2,
                )

    print("Started Training")
    model.learn(total_timesteps=100000, progress_bar=True)
    #  model.save(r"D:\Documents\GitHub\HighwayEnv\models\model_v1")

    # Run the algorithm
    model = PPO.load(r"D:\Documents\GitHub\HighwayEnv\models\model_v1", env=env)

    env = gym.make('RaceTrackYuvraj', render_mode="human")
    obs, info = env.reset()
    print(env.action_space)

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)