import torch
import random
import numpy as np
from pathlib import Path
from collections import deque
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CHECKPOINT_FILE = Path("checkpoint.pth")
MODEL_FILE = Path("model.pth")

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LinearQNet(11, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_action_with_debug(self, state):
        self.epsilon = 80 - self.n_games
        state0 = torch.tensor(state, dtype=torch.float)
        q_values = self.model(state0).detach().cpu().numpy()
        q_values_np = np.asarray(q_values, dtype=np.float64)
        q_shifted = q_values_np - np.max(q_values_np)
        exp_q = np.exp(np.clip(q_shifted, -50.0, 50.0))
        softmax_den = float(np.sum(exp_q))
        if softmax_den <= 0:
            policy_probs = np.full(3, 1.0 / 3.0, dtype=np.float64)
        else:
            policy_probs = exp_q / softmax_den

        explore_rate = float(np.clip(self.epsilon / 200.0, 0.0, 1.0))
        best_action = int(np.argmax(q_values_np))
        effective_probs = np.full(3, explore_rate / 3.0, dtype=np.float64)
        effective_probs[best_action] += (1.0 - explore_rate)

        roll = random.randint(0, 200)

        if roll < self.epsilon:
            action = random.randint(0, 2)
            mode = "explore"
        else:
            action = int(np.argmax(q_values))
            mode = "policy"

        debug = {
            "epsilon": float(self.epsilon),
            "roll": int(roll),
            "mode": mode,
            "action": int(action),
            "q_values": [float(v) for v in q_values.tolist()],
            "policy_probs": [float(v) for v in policy_probs.tolist()],
            "effective_probs": [float(v) for v in effective_probs.tolist()],
            "explore_rate": explore_rate,
        }
        return action, debug

    def get_action(self, state):
        action, _ = self.get_action_with_debug(state)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) == 0:
            return

        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def save(self, extra_state=None):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "n_games": self.n_games,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "extra_state": extra_state or {},
        }
        torch.save(checkpoint, CHECKPOINT_FILE)

        # Backward-compatible lightweight model export.
        torch.save(self.model.state_dict(), MODEL_FILE)

    def load(self):
        loaded_from = "none"
        extra_state = {}
        checkpoint_loaded = False

        if CHECKPOINT_FILE.exists():
            try:
                checkpoint = torch.load(CHECKPOINT_FILE, map_location=torch.device("cpu"))
                self.model.load_state_dict(checkpoint["model_state_dict"])

                optimizer_state = checkpoint.get("optimizer_state_dict")
                if optimizer_state:
                    self.trainer.optimizer.load_state_dict(optimizer_state)

                self.n_games = int(checkpoint.get("n_games", 0))
                self.epsilon = float(checkpoint.get("epsilon", 0))
                self.gamma = float(checkpoint.get("gamma", self.gamma))
                extra_state = checkpoint.get("extra_state", {}) or {}
                loaded_from = "checkpoint"
                checkpoint_loaded = True
            except Exception:
                loaded_from = "checkpoint_error"

        if not checkpoint_loaded and MODEL_FILE.exists():
            self.model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
            loaded_from = "model_only"

        return {"loaded_from": loaded_from, "extra_state": extra_state}
