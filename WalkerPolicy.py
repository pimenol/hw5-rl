import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class WalkerPolicy(nn.Module):
    def __init__(self, state_dim=29, action_dim=8, init_log_std=-1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)
        
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # action head
        self.mu_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states):            
        return self.shared_layers(states)

    def determine_actions(self, states):
        # if not isinstance(states_np, torch.Tensor):
        #     states = torch.tensor(states_np, dtype=torch.float32)
        # else:
        #     states = states_np.float()
        # mu, _ = self.forward(states)
        # return mu.detach().cpu().numpy
        features = self(states)
        mu = self.mu_head(features)
        return mu

    def sample_actions(self, states):
        features = self(states)
        mu = self.mu_head(features)                 # (N, action_dim)
        std = torch.exp(self.log_std).expand_as(mu)  # broadcast to same shape as mu
        dist = Normal(mu, std)

        actions = dist.sample()       # (N, action_dim)
        # log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)  # sum over action_dim

        return actions

    def log_prob(self, states, actions):
        # mu, std = self.forward(states)
        # dist = Normal(mu, std)
        # logp = dist.log_prob(actions)
        # if logp.ndim == 2:
        #     logp = logp.sum(axis=1, keepdim=True)  
        # return logp.squeeze(-1) 
        features = self(states)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        return log_probs
    
    def value_estimates(self, states):
        features = self(states)
        values = self.value_head(features)  # (N, 1)
        return values.squeeze(-1)
    
    
