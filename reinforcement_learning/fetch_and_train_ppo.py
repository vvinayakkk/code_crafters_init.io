import gym
import numpy as np
from stable_baselines3 import PPO
import pymongo
from datetime import datetime

# Custom environment for compliance flagging
class ComplianceEnv(gym.Env):
    def __init__(self, logs):
        super(ComplianceEnv, self).__init__()
        self.logs = logs
        self.current_idx = 0
        self.action_space = gym.spaces.Discrete(2)  # 0: Not flag, 1: Flag
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        self.current_idx = np.random.randint(0, len(self.logs))
        return self._get_state()

    def _get_state(self):
        log = self.logs[self.current_idx]
        # Extract features (dummy example: normalize values)
        user_id = hash(log["user_id"]) % 1000 / 1000.0
        log_id = hash(log["log_id"]) % 1000 / 1000.0
        timestamp = (datetime.strptime(log["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp() % 86400) / 86400.0
        details_flag = 1.0 if "flagged" in log["details"].lower() else 0.0
        return np.array([user_id, log_id, timestamp, details_flag])

    def step(self, action):
        log = self.logs[self.current_idx]
        # Reward: +1 if action matches the log's action, -1 otherwise
        expected_action = 1 if log["action"] == "SHIPMENT_FLAGGED" else 0
        reward = 1 if action == expected_action else -1
        self.current_idx = (self.current_idx + 1) % len(self.logs)
        done = self.current_idx == 0
        return self._get_state(), reward, done, {}

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["complianceDB"]
collection = db["logs"]

# Fetch logs
logs = list(collection.find())
if not logs:
    print("No logs found in the database.")
    exit()

# Create environment with logs
env = ComplianceEnv(logs)

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Train for 10,000 timesteps
model.save("trained_rl_model")

print("PPO model retrained and saved as 'trained_rl_model.zip'")