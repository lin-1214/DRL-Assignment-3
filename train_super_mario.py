import random, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import gym_super_mario_bros
import math

from tqdm import trange
from collections import deque
from gym.wrappers import TimeLimit
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from torchvision import transforms as T


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: 
                break
        return obs, total_reward, done, info
    
class GrayScaleResize(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        super().__init__(env)
        self.width = width
        self.height = height
    
        self.transform = T.Compose([
            T.ToPILImage(), 
            T.Grayscale(), 
            T.Resize((self.width, self.height)), 
            T.ToTensor()
        ])
        self.observation_space = gym.spaces.Box(0, 1, shape=(1, self.width, self.height), dtype=np.float32)
        
    def observation(self, obs):
        return self.transform(obs)
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, stack):
        super().__init__(env)
        self.stack = stack
        self.frames = deque(maxlen=stack)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, 
            shape=(shape[0]*stack, shape[1], shape[2]), 
            dtype=np.float32
        )
        
    def reset(self):
        obs = self.env.reset()
        
        [self.frames.append(obs) for _ in range(self.stack)]
        return self._get_observation()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        observation = self._get_observation()

        return observation, reward, done, info
        
    def _get_observation(self):
        # Helper method to create the stacked observation
        return np.concatenate(list(self.frames), axis=0)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        # Initialize mu weights with Kaiming uniform initialization
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        
        # Initialize sigma weights with constant scaled by fan-in
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(fan_in))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        # Generate standard normal noise
        noise = torch.randn(size, device=self.weight_mu.device)
        # Transform using the factorized Gaussian noise technique
        # This is equivalent to: sign(x) * sqrt(|x|)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return torch.matmul(x, weight.t()) + bias
    
class RainbowDQNConfig:

    VERSION = 'SuperMarioBros-v0'
    WIDTH = 84
    HEIGHT = 84
    # Environment settings
    FRAME_SKIP = 4          # Number of frames to skip for each action
    FRAME_STACK = 4         # Number of frames to stack together for temporal information
    MAX_EPISODE_STEPS = 3000 # Maximum steps before episode termination
    DEATH_PENALTY = -100    # Penalty when Mario loses a life
    
    # Training parameters
    BATCH_SIZE = 32         # Number of samples per training batch
    GAMMA = 0.9             # Discount factor for future rewards
    REPLAY_SIZE = 80000     # Size of experience replay buffer
    LEARNING_RATE = 0.00025 # Learning rate for optimizer
    
    # Training schedule
    TOTAL_EPISODES = 10000  # Total number of episodes to train
    MAX_TRAINING_FRAMES = 10000000  # Maximum frames for training
    REPLAY_START_SIZE = 10000    # Collect this many experiences before training
    
    # Update frequencies
    TARGET_UPDATE_FREQ = 10000     # Update target network every N frames
    CHECKPOINT_FREQ = 100000     # Save model checkpoint frequency
    LOGGING_FREQ = 100         # Logging frequency
    
    # Prioritized Experience Replay parameters
    ALPHA = 0.6             # Controls how much prioritization is used
    BETA_START = 0.4        # Initial importance-sampling correction value
    BETA_INCREMENT = 0.001  # Gradually increase beta to 1.0 during training
    PER_EPSILON = 1e-6      # Small constant to prevent zero priority
    
    # Stuck detection parameters
    STUCK_STEPS = 100       # Number of steps to detect being stuck
    STUCK_WINDOW = 30       # Window size to check for action repetition
    STUCK_TEMP_INCREASE = 2.0  # Temperature increase factor when stuck
    
    # Paths
    MODEL_PATH = "models"   # Directory to save model checkpoints
    
    # Debug settings
    DEBUG = False           # Enable debug mode for shape checking

class RainbowDQN(nn.Module):
    def __init__(self, input_channels, n_actions, debug=False):
        super(RainbowDQN, self).__init__()
        
        self.debug = debug
        
        # Improved feature extraction CNN with residual connections
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        
        # Calculate feature size after convolutions
        self.feature_size = self._get_conv_output_size(input_channels)
        
        # Value stream with dropout for regularization
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512), nn.ReLU(),
            NoisyLinear(512, 1)
        )
        
        # Advantage stream with dropout for regularization
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512), nn.ReLU(),
            NoisyLinear(512, n_actions)
        )
    
    def _get_conv_output_size(self, input_channels):
        """Calculate the size of the flattened features after convolution layers"""
        # Create a dummy input tensor
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        
        # Forward pass through convolutional layers
        x = self.conv(dummy_input)
        
        # Flatten and return size
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        if self.debug:
            print(f"Input shape: {x.shape}")
        
        # Extract features through CNN layers
        x = self.conv(x)
        if self.debug:
            print(f"After conv shape: {x.shape}")
        
        # Flatten features
        features = x.view(x.size(0), -1)
        if self.debug:
            print(f"Flattened features shape: {features.shape}")
        
        # Compute value and advantage streams
        value = self.value_stream(features)
        if self.debug:
            print(f"Value stream shape: {value.shape}")
            
        advantage = self.advantage_stream(features)
        if self.debug:
            print(f"Advantage stream shape: {advantage.shape}")
        
        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class SumTree:
    def __init__(self, capacity):
        # Complete binary tree with capacity leaves requires 2*capacity-1 nodes
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        self._max_priority = 1.0  # Track max priority for new samples
    
    def _propagate(self, idx, change):
        """Update parent nodes efficiently with iteration"""
        while idx != 0:  # Continue until we reach the root
            idx = (idx - 1) // 2  # Parent index
            self.tree[idx] += change
    
    def _retrieve(self, idx, s):
        """Find the leaf node that contains the sum s using iteration"""
        left = 2 * idx + 1
        right = left + 1
        
        # If we reach a leaf node, return it
        while left < len(self.tree):
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
            
            left = 2 * idx + 1
            right = left + 1
            
        return idx
    
    def total(self):
        """Return the sum of all priorities"""
        return self.tree[0]
    
    def max_priority(self):
        """Return the maximum priority"""
        return self._max_priority
    
    def add(self, priority, data):
        """Add a new sample with its priority"""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx, priority):
        """Update the priority of a sample"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        self._max_priority = max(self._max_priority, priority)
    
    def get(self, s):
        """Get a sample based on a value s"""
        if self.total() == 0:
            return 0, 0, None
            
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[data_idx]
    
    def __len__(self):
        """Return the number of samples in the tree"""
        return self.n_entries

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance-sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def push(self, state, action, reward, next_state, done, error=None):
        # Use max priority for new experiences
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self) < batch_size:
            raise ValueError(f"Not enough samples in buffer ({len(self)}) to sample batch of {batch_size}")
            
        # Update beta parameter for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Initialize arrays to store results
        batch_data = []
        tree_indices = []
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Stratified sampling approach
        total_priority = self.tree.total()
        segment_size = total_priority / batch_size
        
        # Sample from each segment
        for i in range(batch_size):
            # Calculate segment boundaries
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            
            # Sample uniformly from segment
            value = np.random.uniform(segment_start, segment_end)
            
            # Retrieve sample from tree
            tree_idx, priority, experience = self.tree.get(value)
            
            # Store results
            tree_indices.append(tree_idx)
            batch_data.append(experience)
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probabilities = priorities / total_priority
        is_weights = np.power(len(self) * sampling_probabilities, -self.beta)
        is_weights = is_weights / np.max(is_weights)  # Normalize weights
        
        # Unpack the batch of experiences
        states, actions, rewards, next_states, dones = map(list, zip(*batch_data))
        
        # Convert to tensors more robustly
        try:
            # For tensor states
            if torch.is_tensor(states[0]):
                states = torch.cat([s for s in states]).to(self.device)
            else:
                # For numpy states
                states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        except (TypeError, RuntimeError):
            # Fallback for other state types
            device = next(iter(states[0].parameters())).device if hasattr(states[0], 'parameters') else self.device
            states = torch.cat(states).to(device)
            
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        try:
            # For tensor next_states
            if torch.is_tensor(next_states[0]):
                next_states = torch.cat([s for s in next_states]).to(self.device)
            else:
                # For numpy next_states
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        except (TypeError, RuntimeError):
            # Fallback for other next_state types
            next_states = torch.cat(next_states).to(self.device)
            
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones, tree_indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities for sampled transitions"""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def __len__(self):
        """Return the number of samples in the buffer"""
        return self.tree.n_entries
        
    def clear(self):
        """Clear the replay buffer"""
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0

def make_env(config):
    env = gym_super_mario_bros.make(config.VERSION)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, config.FRAME_SKIP)
    env = GrayScaleResize(env, config.WIDTH, config.HEIGHT)
    env = FrameStack(env, config.FRAME_STACK)
    env = TimeLimit(env, max_episode_steps=config.MAX_EPISODE_STEPS)
    return env  

def obs_to_state(obs, debug=False):
    """Convert observation to state for neural network input."""
    if debug:
        print(f"Original observation shape: {np.array(obs).shape}")
    
    state = np.array(obs)  
    state = torch.from_numpy(state).float() / 255.0 
    state = state.unsqueeze(0)
    
    if debug:
        print(f"Processed state shape: {state.shape}")
    
    return state

class RainbowAgent:
    def __init__(self, config, n_actions, device):
        self.config = config
        self.device = device
        self.n_actions = n_actions
        self.debug = config.DEBUG
        self.episode_count = 0
        self.frame_count = 0
        
        # Networks
        self.policy_net = RainbowDQN(config.FRAME_STACK, n_actions, debug=self.debug).to(device)
        self.target_net = RainbowDQN(config.FRAME_STACK, n_actions, debug=self.debug).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(
            config.REPLAY_SIZE, 
            alpha=config.ALPHA, 
            beta=config.BETA_START,
            beta_increment=config.BETA_INCREMENT,
            epsilon=config.PER_EPSILON
        )
        
    def select_action(self, state, temperature):
        with torch.no_grad():
            if self.debug:
                print(f"Action selection - state shape: {state.shape}")
            
            q_values = self.policy_net(state.to(self.device))
            
            if self.debug:
                print(f"Q-values shape: {q_values.shape}, values: {q_values}")
                
            action = q_values.max(1)[1].item()
            
            if self.debug:
                print(f"Selected action: {action}")
                
        return action
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def optimize_model(self):
        if len(self.memory) < self.config.REPLAY_START_SIZE:
            return None
            
        # Sample batch with priorities
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
        
        # Compute current Q values
        q_values = self.policy_net(batch_state)
        state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(batch_next_state).max(1)[1].unsqueeze(1)
            next_q_target = self.target_net(batch_next_state)
            next_state_values = next_q_target.gather(1, next_actions).squeeze(1)
            next_state_values[batch_done] = 0.0
            expected_values = batch_reward + self.config.GAMMA * next_state_values
        
        td_error = (state_action_values - expected_values).detach()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_error.abs().cpu().numpy())
        
        # Compute weighted MSE loss
        loss = (weights * F.smooth_l1_loss(state_action_values, expected_values, reduction='none')).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Reset noise in noisy layers
        self.policy_net.reset_noise()
        
        return loss.item()
    
    def save_model(self, path, tag=""):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy_net.state_dict(), f"{path}/mario_rainbow_dqn_{tag}.pth")
        
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Loaded model from {path}")

def train(config, weight=None):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env(config)
    n_actions = env.action_space.n
    
    # Create agent
    agent = RainbowAgent(config, n_actions, device)
    
    # Load pre-trained model if specified
    if weight is not None:
        agent.load_model(weight)
    
    # Initialize tracking variables
    all_rewards = deque(maxlen=100)  # Store last 100 episode rewards
    moving_avg_reward = 0
    best_avg_reward = float('-inf')
    
    # Main training loop
    progress = trange(agent.episode_count + 1, agent.episode_count + config.TOTAL_EPISODES + 1, 
                     dynamic_ncols=True, desc="Training", unit="Episodes")
    
    for episode in progress:
        # Reset environment
        state = env.reset()
        state = obs_to_state(state, debug=config.DEBUG)
        done = False
        episode_reward = 0
        prev_life = None
        temperature = 1.0  # Default temperature for action selection
        
        # Reset noise for this episode
        agent.policy_net.reset_noise()
        
        while not done:
            # Select action with current temperature
            action = agent.select_action(state, temperature)
            
            # # Store action for stuck detection
            # recent_actions.append(action)
            # recent_states.append(state.cpu().numpy().flatten().sum())
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            truncated = info.get("TimeLimit.truncated", False)
            
            # Process observation
            next_state = obs_to_state(next_obs, debug=config.DEBUG)
            
            # Get Mario's x position
            x_pos = info.get('x_pos', 0)
            
            # # Check for progress
            # if x_pos > max_x_pos + 5:  # Significant progress threshold
            #     max_x_pos = x_pos
            #     no_progress_counter = 0
            #     temperature = 1.0  # Reset temperature when making progress
            # else:
            #     no_progress_counter += 1
            
            # # Detect if Mario is stuck using multiple indicators
            # is_stuck = False
            # if no_progress_counter >= config.STUCK_STEPS:
            #     # Check if actions are repetitive
            #     if len(recent_actions) == config.STUCK_WINDOW:
            #         # Count unique actions in the window
            #         unique_actions = len(set(recent_actions))
            #         # Check state variation (to detect repetitive jumping)
            #         state_variation = np.std([s for s in recent_states])
                    
            #         # If few unique actions or low state variation, consider stuck
            #         if unique_actions <= 3 or state_variation < 0.1:
            #             is_stuck = True
            
            # # Apply exploration boost if stuck
            # if is_stuck:
            #     # Increase temperature to encourage exploration
            #     temperature = config.STUCK_TEMP_INCREASE
            #     # Reset noise in the policy network to try different patterns
            #     agent.policy_net.reset_noise()
            #     # Reset counter after applying exploration boost
            #     no_progress_counter = 0
            #     # print(f"\nStuck detected at position {x_pos}, boosting exploration")
            
            # Apply custom penalties
            done = done and not truncated
            shaped_reward = reward
            life = info.get('life')
            
            if prev_life is None: 
                prev_life = life
            elif life < prev_life: 
                shaped_reward += config.DEATH_PENALTY
                prev_life = life
            
            # Add to episode reward
            episode_reward += reward
            
            # Store transition in replay memory
            agent.store_transition(state, action, shaped_reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Increment frame counter
            agent.frame_count += 1
            
            # Train the model
            loss = agent.optimize_model()
            
            if config.DEBUG and loss is not None:
                print(f"Frame {agent.frame_count}, Loss: {loss:.4f}")
            
            # Target network synchronization
            if agent.frame_count % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
            
            # Periodic saving
            if agent.frame_count % config.CHECKPOINT_FREQ == 0:
                agent.save_model(config.MODEL_PATH, tag=f"_{agent.frame_count}")
                # Also save a copy as the latest model
                agent.save_model(config.MODEL_PATH, tag="latest")
            
            # Periodic logging
            if agent.frame_count % config.LOGGING_FREQ == 0:
                progress.set_postfix(
                    frames=agent.frame_count,
                    episode=episode,
                    avg_reward=f"{moving_avg_reward:.2f}",
                    best_avg=f"{best_avg_reward:.2f}"
                )
        
        all_rewards.append(episode_reward)
        moving_avg_reward = np.mean(all_rewards)
        
        if config.DEBUG:
            print(f"Episode {episode} finished with reward {episode_reward}")
            print(f"Moving average reward: {moving_avg_reward:.2f}")
        
        # Update best average reward
        if moving_avg_reward > best_avg_reward:
            best_avg_reward = moving_avg_reward
            # Save best model
            agent.save_model(config.MODEL_PATH, tag="best")
            
            if config.DEBUG:
                print(f"New best average reward: {best_avg_reward:.2f}")
        
        # Increment episode counter
        agent.episode_count += 1
    
    env.close()

if __name__ == "__main__":

    config = RainbowDQNConfig()

    # Create model directory if it doesn't exist
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)

    # Uncomment to enable debug mode
    # config.DEBUG = True

    # Start training
    train(config, weight=f'{config.MODEL_PATH}/mario_rainbow_dqn_latest.pth')