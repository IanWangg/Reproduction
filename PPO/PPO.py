from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from gym import spaces

from model_body import RainbowBody, MlpBody
from model_head import DiagGaussianHead, SoftmaxHead, DeterministicHead

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer


class PPO(object):
    def __init__(
        self,
        env : VecEnv,
        gamma=0.99,
        lr=1e-4,
        gae_lambda=0.95,
        n_step=128,
        max_grad_norm=0.5,
        feature_dim=512,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # keep track of hyperparameters
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.n_step = n_step
        self.max_grad_norm = max_grad_norm
        self.feature_dim = feature_dim
        
        self.num_env = self.env.num_envs
        self.state_space = env.observation_space
        self.action_space = env.action_space
        
        
        # get model body
        self.state_shape = self.state_space.shape
        if len(self.state_shape) > 1:
            self.feature_extractor = RainbowBody(self.state_shape, self.feature_dim).to(self.device)
        else:
            self.feature_extractor = MlpBody(self.state_shape, self.feature_dim).to(self.device)
            
            
        # get model policy head
        if isinstance(self.action_space, spaces.Box):
            self.action_dim = self.action_space.shape[0]
            self.max_action = self.action_space.high
            self.min_action = self.action_space.low
            self.policy_head = DiagGaussianHead(self.feature_dim, self.action_dim).to(self.device)
        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dim = self.action_space.n
            self.policy_head = SoftmaxHead(self.feature_dim, self.action_dim).to(self.device)
        else:
            raise NotImplemented
        
        
        # get model value head
        self.value_head = DeterministicHead(self.feature_dim, 1).to(self.device)
        
        
        # get optimizer
        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters()) + \
                list(self.policy_head.parameters()) + \
                list(self.value_head.parameters()),
            lr=self.lr
        )
        
        
        # get replaybuffer (from sb3)
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_step,
            observation_space=self.state_space,
            action_space=self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.num_env
        )
        
        
        # get stats buffer
        self.ep_info_buffer = deque(maxlen=100)  
        self.ep_success_buffer = deque(maxlen=100)  
        
        
        # reset env
        self._last_state = self.env.reset()
        self._last_episode_start = np.ones((self.num_env,), dtype=bool)
        self.total_step = 0
        
    def train(self):
        self.feature_extractor.train()
        self.policy_head.train()
        self.value_head.train()
        
    def eval(self):
        self.feature_extractor.eval()
        self.policy_head.eval()
        self.value_head.eval()
        
    def _update_info_buffer(self, info, done=None):
        if done is None:
            done = np.array([False] * len(info))
        for idx, single_info in enumerate(info):
            maybe_ep_info = single_info.get("episode")
            maybe_is_success = single_info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and done[idx]:
                self.ep_success_buffer.append(maybe_is_success)
    
    def collect_rollout(self):
        self.train()
        
        rollout_step = 0
        
        # start rollout
        while rollout_step < self.n_step:
            
            # get action
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self._last_state).float().to(self.device)
                # print(obs_tensor.shape)
                # print(self.feature_extractor, self.feature_extractor.input_shape)
                feature = self.feature_extractor(obs_tensor)
                value = self.value_head(feature)
                action = self.policy_head(feature, reparam=False, deterministic=False)
                log_prob = self.policy_head.log_prob(action)
            
            action = action.cpu().numpy()
            if isinstance(self.action_space, spaces.Box):
                action = np.clip(action, self.min_action, self.max_action)
                
                
            # env step
            next_state, reward, done, info = env.step(action)
            self._update_info_buffer(info, done)
            self.total_step += self.num_env
            rollout_step += 1
            
            
            # put reward to buffer
            for idx, single_done in enumerate(done):
                if (single_done and 
                    info[idx].get('terminal_observation') is not None and
                    info[idx].get('TimeLimit.truncated', False)
                ):
                    terminal_state = torch.as_tensor(info[idx]).float().to(self.device)
                    with torch.no_grad():
                        feature = self.feature_extractor(terminal_state)
                        terminal_value = self.value_head(feature)
                    
                    reward[idx] += self.gamma * terminal_value
            
            self.rollout_buffer.add(self._last_state, action, reward, self._last_episode_start, value, log_prob)
            self._last_state = next_state
            self._last_episode_start = done
        
        with torch.no_grad():
            next_state = torch.as_tensor(next_state).float().to(self.device)
            feature = self.feature_extractor(next_state)
            value = self.value_head(feature)
        
        self.rollout_buffer.compute_returns_and_advantage(last_values=value, dones=done)
    
    def train(self):
        pass
    
    
if __name__ == '__main__':
    from stable_baselines3.common.env_util import make_vec_env
    
    env = make_vec_env(
        'Hopper-v3',
        n_envs=10,
    )
    
    model = PPO(
        env,
    )
    
    model.collect_rollout()