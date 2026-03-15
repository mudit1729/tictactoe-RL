from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.97
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 512
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 15000
    device: str = "cpu"


class TinyQNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[tuple[np.ndarray, int, float, Optional[np.ndarray], bool]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
    ) -> None:
        self.buffer.append((state.copy(), action, reward, None if next_state is None else next_state.copy(), done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        next_state_array = np.stack([
            np.zeros(9, dtype=np.float32) if next_state is None else next_state
            for next_state in next_states
        ])
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            next_state_array.astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        self.policy_net = TinyQNetwork(self.config.hidden_dim).to(self.device)
        self.target_net = TinyQNetwork(self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.update_steps = 0

    def epsilon_for_episode(self, episode: int) -> float:
        if episode >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_end
        span = self.config.epsilon_start - self.config.epsilon_end
        decay = episode / max(1, self.config.epsilon_decay_episodes)
        return self.config.epsilon_start - span * decay

    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = self.policy_net(tensor).squeeze(0).cpu().numpy()
        return values

    def greedy_action(self, state: np.ndarray, valid_actions: list[int]) -> int:
        q_values = self.q_values(state)
        masked = np.full_like(q_values, -1e9)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))

    def act(self, state: np.ndarray, valid_actions: list[int], epsilon: float) -> int:
        if random.random() < epsilon:
            return valid_actions[random.randrange(len(valid_actions))]
        return self.greedy_action(state, valid_actions)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            legal_mask = next_states_t == 0.0
            next_q_values = next_q_values.masked_fill(~legal_mask, -1e9)
            next_best = next_q_values.max(dim=1).values
            next_best = torch.where(dones_t > 0, torch.zeros_like(next_best), next_best)
            targets = rewards_t + self.config.gamma * next_best * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str | Path, extra: Optional[dict] = None) -> None:
        payload = {
            "state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "update_steps": self.update_steps,
            "extra": extra or {},
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "DQNAgent":
        payload = torch.load(str(path), map_location=device or "cpu")
        config = DQNConfig(**payload["config"])
        if device is not None:
            config.device = device
        agent = cls(config)
        agent.policy_net.load_state_dict(payload["state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.update_steps = int(payload.get("update_steps", 0))
        return agent

    def export_json(self, path: str | Path, metadata: Optional[dict] = None) -> None:
        state = self.policy_net.state_dict()
        layers = []
        for weight_key, bias_key in (("layers.0.weight", "layers.0.bias"), ("layers.2.weight", "layers.2.bias"), ("layers.4.weight", "layers.4.bias")):
            layers.append(
                {
                    "weight": state[weight_key].cpu().tolist(),
                    "bias": state[bias_key].cpu().tolist(),
                }
            )
        payload = {
            "hidden_dim": self.config.hidden_dim,
            "layers": layers,
            "metadata": metadata or {},
        }
        Path(path).write_text(json.dumps(payload))
