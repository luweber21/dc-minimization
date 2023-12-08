import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C, SAC, DDPG
from src.environment.DiscreteBattery import DiscreteBattery
import numpy as np
import matplotlib.pyplot as plt
import time

N_train = 300
env = DiscreteBattery(discrete_action_space=True, days=list(range(N_train)))

for seed in range(5):
    env.reset()
    model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.0001, exploration_final_eps = 0.1, seed=seed)
    model.learn(total_timesteps=100000 * 5 * 56, log_interval=np.inf, progress_bar=True)
    model.save(f"cleps/dqn_lr_1e-4_eps1e-1_{seed}")
