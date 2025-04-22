import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym
import cv2

model_path = "best_mario_dueling_dqn.pth"

# Preprocessing wrappers for Mario environment
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else shape
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def __len__(self):
        return len(self.buffer)

# Agent class
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = 0.99
        self.lr = 1e-4
        self.batch_size = 32
        self.buffer_size = 100000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 10000
        self.target_update = 1000
        self.frame_stack = 4
        
        # Initialize networks
        self.policy_net = DuelingDQN((self.frame_stack, 84, 84), self.action_space.n).to(self.device)
        self.target_net = DuelingDQN((self.frame_stack, 84, 84), self.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size)
        
        self.steps_done = 0
        self.episode_rewards = []
        self.frame_buffer = deque(maxlen=self.frame_stack)

        if model_path is not None:
            self.load(model_path)
            print(f"Loaded trained model from {model_path}")
            # Set epsilon to final value for inference
            self.epsilon_start = self.epsilon_final
            self.steps_done = self.epsilon_decay * 10  # Force epsilon to be at minimum
        
    def _get_epsilon(self):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        return epsilon
    
    def _preprocess_observation(self, observation):
        # Initialize frame buffer if it's empty
        if len(self.frame_buffer) == 0:
            for _ in range(self.frame_stack):
                self.frame_buffer.append(observation)
        else:
            self.frame_buffer.append(observation)
            
        # Stack frames
        state = np.array(self.frame_buffer)
        return state
    
    # ... existing code ...

    def act(self, observation):
        self.steps_done += 1
        
        # Handle different observation formats
        # If observation is from FrameStack, it might be LazyFrames or have shape (4, 84, 84)
        if hasattr(observation, '_frames'):  # LazyFrames from FrameStack
            observation = np.array(observation)
        
        # Check the shape and reshape if needed
        if len(observation.shape) == 3 and observation.shape[0] == 4:
            # Already stacked frames with shape (4, 84, 84)
            state = observation
        elif len(observation.shape) == 4 and observation.shape[0] == 4:
            # Stacked frames with extra dimension (4, 1, 84, 84)
            state = observation.reshape(4, 84, 84)
        else:
            # Single frame or other format, use our frame buffer
            if len(self.frame_buffer) == 0:
                for _ in range(self.frame_stack):
                    self.frame_buffer.append(observation)
            else:
                self.frame_buffer.append(observation)
            state = np.array(self.frame_buffer)
        
        # Ensure state has shape [channels, height, width] for the network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        epsilon = self._get_epsilon()
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            action = self.action_space.sample()
            
        return action

    # Also update the update method to handle different observation formats
    def update(self, state, action, reward, next_state, done):
        # Convert LazyFrames to numpy arrays if needed
        if hasattr(state, '_frames'):
            state = np.array(state)
        if hasattr(next_state, '_frames'):
            next_state = np.array(next_state)
        
        # Ensure states have the correct shape
        if len(state.shape) == 4:
            state = state.reshape(4, 84, 84)
        if len(next_state.shape) == 4:
            next_state = next_state.reshape(4, 84, 84)
        
        # Store transition in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        # Rest of the update method remains the same
        # ... existing code ...
            
        # Start training when we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update the target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']