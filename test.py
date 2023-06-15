import gymnasium as gym 
from stable_baselines3 import PPO


env = gym.make("ALE/Galaxian-v5",  render_mode = "human")

model = PPO.load("models/PPO/galaxian-ai-v5-100000", env)

obs, info = env.reset()


while True: 
    env.render()
    action, _state = model.predict(obs,deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated: 
        obs, info = env.reset()
