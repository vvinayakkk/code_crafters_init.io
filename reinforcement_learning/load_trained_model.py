import gym
from stable_baselines3 import PPO
import numpy as np

# Define a simple custom environment for demonstration
class ComplianceEnv(gym.Env):
    def __init__(self):
        super(ComplianceEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)  # 0: Not flag, 1: Flag
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.random.rand(4)  # Dummy state (e.g., user_id, log_id, etc.)
        return self.state

    def step(self, action):
        reward = 1 if action == 1 else -1  # Dummy reward for demo
        self.state = np.random.rand(4)  # New state
        done = False
        return self.state, reward, done, {}

# Load the pre-trained model
model = PPO.load("trained_compliance_model.zip")

# Create the environment
env = ComplianceEnv()

# Test the model
obs = env.reset()
for _ in range(5):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {'Flag' if action == 1 else 'Not Flag'}, Reward: {reward}")