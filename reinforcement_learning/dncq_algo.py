import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import pymongo
from datetime import datetime
import logging
import sys
import os
from flask import Flask, jsonify, request
import threading
import time
import random
import json
from typing import Dict, List, Tuple, Any
import torch

# Configure logging for "fancy" output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [CrazyComplianceRL] - %(message)s',
    handlers=[
        logging.FileHandler("compliance_rl_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Flask app for triggering training
app = Flask(__name__)

# Global variables for "crazy" setup
TRAINING_IN_PROGRESS = False
MODEL_PATH = "dqn_compliance_model.zip"
CHECKPOINT_PATH = "checkpoints/dqn_compliance_checkpoint"
TRAINING_LOG = "training_metrics.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Custom environment for compliance flagging
class ComplianceEnv(gym.Env):
    def __init__(self, logs: List[Dict], max_steps: int = 1000):
        super(ComplianceEnv, self).__init__()
        self.logs = logs
        self.current_idx = 0
        self.max_steps = max_steps
        self.step_count = 0
        self.action_space = gym.spaces.Discrete(2)  # 0: Not flag, 1: Flag
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        logger.info("ComplianceEnv initialized with %d logs", len(self.logs))

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed, options=options)
        self.current_idx = np.random.randint(0, len(self.logs))
        self.step_count = 0
        state = self._get_state()
        logger.debug("Environment reset - Current state: %s", state)
        return state, {}

    def _get_state(self) -> np.ndarray:
        log = self.logs[self.current_idx]
        try:
            user_id = hash(log["user_id"]) % 1000 / 1000.0
            log_id = hash(log["log_id"]) % 1000 / 1000.0
            timestamp = (datetime.strptime(log["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp() % 86400) / 86400.0
            details_flag = 1.0 if "flagged" in log["details"].lower() else 0.0
            return np.array([user_id, log_id, timestamp, details_flag], dtype=np.float32)
        except Exception as e:
            logger.error("Error extracting state: %s", str(e))
            return np.zeros(4, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        log = self.logs[self.current_idx]
        expected_action = 1 if log["action"] == "SHIPMENT_FLAGGED" else 0
        reward = 1.0 if action == expected_action else -1.0
        self.current_idx = (self.current_idx + 1) % len(self.logs)
        self.step_count += 1
        terminated = self.current_idx == 0
        truncated = self.step_count >= self.max_steps
        state = self._get_state()
        info = {"action_taken": "Flag" if action == 1 else "Not Flag", "reward": reward}
        logger.debug("Step %d - Action: %d, Reward: %.2f, Terminated: %s", self.step_count, action, reward, terminated)
        return state, reward, terminated, truncated, info

# Database fetcher with "crazy" logging
def fetch_logs_from_db() -> List[Dict]:
    try:
        logger.info("Initiating MongoDB connection for compliance logs...")
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["complianceDB"]
        collection = db["logs"]
        logs = list(collection.find())
        if not logs:
            logger.warning("No logs found in the database! Generating dummy logs...")
            logs = [
                {
                    "_id": f"objectId('fake_{i}')",
                    "log_id": f"objectId('fake_log_{i}')",
                    "user_id": f"objectId('fake_user_{i}')",
                    "action": "SHIPMENT_FLAGGED" if i % 2 == 0 else "SHIPMENT_CREATED",
                    "details": f"Shipment fake_{i} {'flagged for compliance review' if i % 2 == 0 else 'created'}",
                    "timestamp": "2025-08-07T14:25:51.000+00:00"
                } for i in range(100)
            ]
        logger.info("Successfully fetched %d logs from MongoDB", len(logs))
        return logs
    except Exception as e:
        logger.error("Failed to fetch logs from MongoDB: %s", str(e))
        return []
    finally:
        client.close()

# Training function with DQN and checkpoints
def train_dqn_model(logs: List[Dict], total_timesteps: int = 50000) -> None:
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        logger.warning("Training already in progress! Aborting...")
        return

    TRAINING_IN_PROGRESS = True
    logger.info("Starting DQN training with %d timesteps...", total_timesteps)

    try:
        env = ComplianceEnv(logs)
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_PATH, name_prefix="dqn_compliance")
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=10000,
            batch_size=32,
            learning_starts=1000,
            target_update_interval=1000,
            verbose=1,
            device=DEVICE
        )
        logger.info("DQN model initialized with MlpPolicy on %s", DEVICE)

        # Train the model
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, log_interval=100)
        model.save(MODEL_PATH)
        logger.info("DQN training completed! Model saved as %s", MODEL_PATH)

        # Save training metrics (for "fancy" logging)
        metrics = {
            "total_timesteps": total_timesteps,
            "average_reward": np.mean([env.step(random.randint(0, 1))[1] for _ in range(100)]),
            "training_duration": time.time(),
            "status": "completed"
        }
        with open(TRAINING_LOG, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Training metrics saved to %s", TRAINING_LOG)

    except Exception as e:
        logger.error("Error during DQN training: %s", str(e))
    finally:
        TRAINING_IN_PROGRESS = False

# Load and test a pre-trained model with "crazy" logging
def load_and_test_model(logs: List[Dict]) -> Dict[str, Any]:
    logger.info("Loading pre-trained DQN model for testing...")
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning("Model file %s not found! Skipping test...", MODEL_PATH)
            return {"status": "failed", "message": "Model file not found"}

        env = ComplianceEnv(logs, max_steps=10)
        model = DQN.load(MODEL_PATH, device=DEVICE)
        logger.info("Successfully loaded DQN model from %s", MODEL_PATH)

        # Test the model
        obs, _ = env.reset()
        total_reward = 0.0
        for _ in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            logger.debug("Test Step - Action: %s, Reward: %.2f, Info: %s", info["action_taken"], reward, info)
            if terminated or truncated:
                obs, _ = env.reset()

        logger.info("Model testing completed! Total Reward: %.2f", total_reward)
        return {"status": "success", "total_reward": total_reward}

    except Exception as e:
        logger.error("Error during model loading/testing: %s", str(e))
        return {"status": "failed", "message": str(e)}

# Flask route to trigger training
@app.route('/trigger_training', methods=['POST'])
def trigger_training():
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        return jsonify({"status": "error", "message": "Training already in progress!"}), 400

    try:
        data = request.get_json()
        total_timesteps = data.get("total_timesteps", 50000)
        logger.info("Received training request via API - Timesteps: %d", total_timesteps)

        # Fetch logs
        logs = fetch_logs_from_db()
        if not logs:
            return jsonify({"status": "error", "message": "No logs available for training!"}), 500

        # Start training in a separate thread
        training_thread = threading.Thread(target=train_dqn_model, args=(logs, total_timesteps))
        training_thread.start()
        return jsonify({"status": "success", "message": "Training triggered successfully!"})

    except Exception as e:
        logger.error("Error in /trigger_training route: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# Flask route to check training status
@app.route('/training_status', methods=['GET'])
def training_status():
    metrics = {}
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            metrics = json.load(f)
    return jsonify({
        "training_in_progress": TRAINING_IN_PROGRESS,
        "metrics": metrics
    })

# Main execution with "crazy" orchestration
def main():
    logger.info("Starting CrazyComplianceRL Server v1.0 - Something crazy is cooked! 🍳🔥")
    logger.info("Fetching logs, training DQN, loading models, and starting Flask server...")

    # Step 1: Fetch logs
    logs = fetch_logs_from_db()
    if not logs:
        logger.error("No logs available! Exiting...")
        sys.exit(1)

    # Step 2: Train DQN model (initial training)
    if not os.path.exists(MODEL_PATH):
        logger.info("No pre-trained model found. Starting initial DQN training...")
        train_dqn_model(logs, total_timesteps=20000)  # Initial training with fewer timesteps
    else:
        logger.info("Pre-trained model found at %s. Skipping initial training.", MODEL_PATH)

    # Step 3: Load and test the model
    test_result = load_and_test_model(logs)
    logger.info("Model test result: %s", test_result)

    # Step 4: Start Flask server
    logger.info("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    main()