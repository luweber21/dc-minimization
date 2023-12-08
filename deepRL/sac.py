import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG
from src.environment.DiscreteBattery import DiscreteBattery
import numpy as np

N_train = 300
env = DiscreteBattery(discrete_action_space=False, days=list(range(N_train)))

for seed in range(5):
    env.reset()
    model = SAC("MultiInputPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=1200000, log_interval=np.inf, progress_bar=True)
    model.save(f"cleps/sac_{seed}")
