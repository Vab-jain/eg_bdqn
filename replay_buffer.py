"""
Simple circular replay buffer for DQN training.
Stores (obs_image, obs_mission, action, reward, next_obs_image, next_obs_mission, done).
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list = []
        self.position = 0

    def push(
        self,
        obs_image: np.ndarray,
        obs_mission: list[int],
        action: int,
        reward: float,
        next_obs_image: np.ndarray,
        next_obs_mission: list[int],
        done: bool,
    ):
        data = (obs_image, obs_mission, action, reward, next_obs_image, next_obs_mission, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        obs_images       = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.uint8,   device=device)
        obs_missions     = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long,    device=device)
        actions          = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.long,    device=device)
        rewards          = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
        next_obs_images  = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.uint8,   device=device)
        next_obs_missions= torch.tensor(np.array([b[5] for b in batch]), dtype=torch.long,    device=device)
        dones            = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32, device=device)

        return {
            "obs_image":        obs_images,
            "obs_mission":      obs_missions,
            "action":           actions,
            "reward":           rewards,
            "next_obs_image":   next_obs_images,
            "next_obs_mission": next_obs_missions,
            "done":             dones,
        }

    def __len__(self) -> int:
        return len(self.buffer)


class DualReplayBuffer:
    def __init__(self, agent_capacity: int, demo_capacity: int):
        self.agent_capacity = agent_capacity
        self.demo_capacity = demo_capacity
        
        self.agent_buffer: list = []
        self.agent_position = 0
        
        self.demo_buffer: list = []
        self.demo_position = 0

    def push(
        self,
        obs_image: np.ndarray,
        obs_mission: list[int],
        action: int,
        reward: float,
        next_obs_image: np.ndarray,
        next_obs_mission: list[int],
        done: bool,
        is_demo: bool = False,
    ):
        data = (obs_image, obs_mission, action, reward, next_obs_image, next_obs_mission, done)
        if is_demo:
            if len(self.demo_buffer) < self.demo_capacity:
                self.demo_buffer.append(data)
            # Permanent Oracle Buffer: never overwrites once full
        else:
            if len(self.agent_buffer) < self.agent_capacity:
                self.agent_buffer.append(data)
            else:
                self.agent_buffer[self.agent_position] = data
            self.agent_position = (self.agent_position + 1) % self.agent_capacity

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        total_size = len(self.agent_buffer) + len(self.demo_buffer)
        indices = np.random.randint(0, total_size, size=batch_size)
        
        batch = []
        for i in indices:
            if i < len(self.agent_buffer):
                batch.append(self.agent_buffer[i])
            else:
                batch.append(self.demo_buffer[i - len(self.agent_buffer)])

        obs_images       = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.uint8,   device=device)
        obs_missions     = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.long,    device=device)
        actions          = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.long,    device=device)
        rewards          = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
        next_obs_images  = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.uint8,   device=device)
        next_obs_missions= torch.tensor(np.array([b[5] for b in batch]), dtype=torch.long,    device=device)
        dones            = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32, device=device)

        return {
            "obs_image":        obs_images,
            "obs_mission":      obs_missions,
            "action":           actions,
            "reward":           rewards,
            "next_obs_image":   next_obs_images,
            "next_obs_mission": next_obs_missions,
            "done":             dones,
        }

    def __len__(self) -> int:
        return len(self.agent_buffer) + len(self.demo_buffer)
