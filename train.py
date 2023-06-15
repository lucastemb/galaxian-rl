import gymnasium as gym
import os
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from callback import SaveOnBestTrainingRewardCallback

from stable_baselines3 import PPO

models_dir = "models/PPO"
log_dir="logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = gym.make("ALE/Galaxian-v5")
env=Monitor(env,log_dir)
env.reset()

TIMESTEPS = 100000

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=0.0001)
model.learn(total_timesteps=TIMESTEPS)
model.save(f"{models_dir}/galaxian-ai-v5-{TIMESTEPS}")








