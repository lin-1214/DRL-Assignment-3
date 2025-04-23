import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym
import cv2
from gym.wrappers import FrameStack

model_path = 'mario_dueling_dqn.pth'


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

    def act(self, observation):
        # Check if observation needs preprocessing
        if len(observation.shape) == 3 and observation.shape[2] == 3:  # RGB format
            # Convert RGB to grayscale and resize to 84x84
            gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0  # Normalize to 0-1
            
            # Update frame buffer
            if len(self.frame_buffer) == 0:  # Initialize buffer if empty
                for _ in range(self.frame_stack):
                    self.frame_buffer.append(normalized)
            else:
                self.frame_buffer.append(normalized)
                
            # Stack frames
            stacked_frames = np.stack(self.frame_buffer, axis=0)
            state = stacked_frames
        else:
            # Assume observation is already preprocessed
            state = observation
        
        # print(f"Current state: {state}")
        # Convert to tensor and get action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
            
        return action

    
    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
