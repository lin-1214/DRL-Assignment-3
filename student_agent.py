import gym
import torch
import numpy as np
from torchvision import transforms as T
from collections import deque
from train_super_mario import RainbowDQN, RainbowDQNConfig

class Agent:
    def __init__(self, model_path='mario_rainbow_dqn_latest.pth', frame_stack=RainbowDQNConfig.FRAME_STACK):
        self.device = torch.device("cpu")
        self.skip_frame = RainbowDQNConfig.FRAME_SKIP
        self.skip_count = 0
        self.action = None
        # Initialize the action space based on COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(12)
        
        self.policy_net = RainbowDQN(input_channels=RainbowDQNConfig.FRAME_STACK, n_actions=self.action_space.n)
        self.policy_net.load_state_dict(torch.load(model_path, weights_only = True, map_location=self.device))
        self.policy_net.eval()
        
        # Frame processing
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.start = True
        # Image transformation pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((RainbowDQNConfig.WIDTH, RainbowDQNConfig.HEIGHT)),
            T.ToTensor()
        ])
        
        

    def preprocess_frame(self, frame):
        """Convert raw frame to grayscale, resize, and normalize."""
        processed = self.transform(frame)
        normalized = processed / 255
        return normalized
    
    def act(self, observation):
        """Determine action based on current observation."""
        # Preprocess the observation
        processed_frame = self.preprocess_frame(observation)
        
        # Initialize frame stack if this is the first call
        if self.start:
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(processed_frame)
            self.start = False
            self.skip_count = 1  # Start the skip counter
        else:
            # Update skip counter
            self.skip_count = (self.skip_count + 1) % self.skip_frame
        
        # Only update action when skip counter is 0 (first frame of a sequence)
        if self.skip_count == 1:
            # Update frame stack with new observation
            self.frames.append(processed_frame)
            
            # Get action from model
            with torch.no_grad():
                # Stack frames and prepare for model input
                stacked_frames = torch.stack(list(self.frames), dim=0).unsqueeze(0).to(self.device)
                # Model expects shape [batch, channels, height, width]
                # But our stack is [stack_size, channels, height, width]
                stacked_frames = stacked_frames.permute(0, 2, 1, 3, 4).squeeze(0)
                
                # Get Q-values and select best action
                q_values = self.policy_net(stacked_frames)
                self.action = q_values.max(1)[1].item()
        
        return self.action