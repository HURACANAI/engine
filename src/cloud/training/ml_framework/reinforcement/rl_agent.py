"""
Reinforcement Learning Agent - Adaptive Strategy Optimization

For adaptive trading strategy optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

logger = structlog.get_logger(__name__)


class DQNAgent(nn.Module):
    """
    Deep Q-Network (DQN) agent for reinforcement learning.
    
    Use cases:
    - Adaptive trading strategy optimization
    - Position sizing
    - Risk management
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Build network
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.q_network = nn.Sequential(*layers)
        self.target_network = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
        logger.info("dqn_agent_initialized", state_dim=state_dim, action_dim=action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        return self.q_network(state)
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.forward(state_tensor)
        return q_values.argmax().item()
    
    def update_target_network(self) -> None:
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """
        Train on a batch of experiences.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            Loss value
        """
        # Current Q values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0].unsqueeze(1)
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class RLAgent:
    """
    Reinforcement Learning agent for trading strategy optimization.
    
    Purpose: Adaptive strategy optimization
    Ideal dataset shape: (num_episodes, episode_length, state_dim)
    Feature requirements: State features (price, volume, indicators, position)
    Output schema: Action (buy, hold, sell) and Q-values
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,  # Buy, Hold, Sell
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Agent configuration
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Create DQN agent
        self.agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.get("hidden_dims", [128, 64]),
            learning_rate=self.config.get("learning_rate", 0.001),
            gamma=self.config.get("gamma", 0.99),
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=self.config.get("buffer_capacity", 10000))
        
        logger.info("rl_agent_initialized", state_dim=state_dim, action_dim=action_dim)
    
    def train(
        self,
        episodes: int = 1000,
        batch_size: int = 32,
        update_target_freq: int = 100,
    ) -> List[float]:
        """
        Train RL agent.
        
        Args:
            episodes: Number of training episodes
            batch_size: Batch size for training
            update_target_freq: Frequency of target network updates
            
        Returns:
            List of episode rewards
        """
        logger.info("training_rl_agent", episodes=episodes)
        
        episode_rewards = []
        
        for episode in range(episodes):
            # Simulate episode (would be replaced with actual environment)
            episode_reward = 0.0
            state = np.random.randn(self.state_dim)  # Placeholder
            
            for step in range(100):  # Placeholder episode length
                # Choose action
                action = self.agent.act(state, training=True)
                
                # Get reward and next state (would come from environment)
                reward = np.random.randn()  # Placeholder
                next_state = np.random.randn(self.state_dim)  # Placeholder
                done = False  # Placeholder
                
                # Store experience
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Train if buffer is large enough
                if len(self.replay_buffer) >= batch_size:
                    batch = self.replay_buffer.sample(batch_size)
                    loss = self.agent.train_step(*batch)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # Update target network
            if episode % update_target_freq == 0:
                self.agent.update_target_network()
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            if episode % 100 == 0:
                logger.info("rl_training_episode", episode=episode, reward=episode_reward)
        
        logger.info("rl_training_complete")
        return episode_rewards
    
    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict action for given state.
        
        Args:
            state: Current state
            
        Returns:
            Tuple of (action, q_values)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent(state_tensor)
        action = q_values.argmax().item()
        return action, q_values.detach().numpy()[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for Mechanic."""
        return {
            "purpose": "Adaptive strategy optimization",
            "ideal_dataset_shape": "(num_episodes, episode_length, state_dim)",
            "feature_requirements": "State features (price, volume, indicators, position)",
            "output_schema": "Action (buy, hold, sell) and Q-values",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }

