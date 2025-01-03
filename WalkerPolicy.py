import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
    
class WalkerPolicy(nn.Module):
    def __init__(self, state_dim=29, action_dim=8, init_log_std=-0.5):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )

        # action head
        self.action_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, action_dim * 2),
        )

        # value head
        self.value_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Learnable log standard deviation
        # self.log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)


    def determine_actions(self, states):
        """states is (N, state_dim) tensor, returns (N, action_dim) actions tensor. 
        This function returns deterministic actions based on the input states. This would be used for control. """
        params = self.action_layers(self.shared_layers(states))  # map states to distribution parameters
        mu, _ = torch.chunk(params, 2, -1)  # split the parameters into mean and std, return mean
        return mu

    def sample_actions(self, states):
        """states is (T, N, state_dim) tensor, returns (T, N, action_dim) actions tensor. 
        This function returns probabilistically sampled actions. This would be used for training the policy."""
        params = self.action_layers(self.shared_layers(states))  # map states to distribution parameters
        mu, sigma = torch.chunk(params, 2, -1)  # split the parameters into mean and std
        sigma = torch.nn.functional.softplus(sigma)  # make sure std is positive
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        actions = distribution.sample()  # sample actions
        return actions

    def log_prob(self, actions, states):
        """states is (T, N, state_dim) tensor. actions is (T, N, action_dim) tensor.
        This function returns the log-probabilities of the actions given the states. $\log \pi_\theta(a_t | s_t)$"""
        params = self.action_layers(self.shared_layers(states))  # map states to distribution parameters
        mu, sigma = torch.chunk(params, 2, -1)  # split the parameters into mean and std
        sigma = torch.nn.functional.softplus(sigma)  # make sure std is positive
        distribution = Normal(mu, sigma)  # create distribution of size (T, N, action_dim)
        logp = distribution.log_prob(actions)
        if len(logp.shape) == 3 and logp.shape[2] > 1:  # this allows generalization to multi-dim action spaces
            logp = logp.sum(dim=2, keepdim=True)  # sum over the action dimension
        return logp

    def value_estimates(self, states):
        """states is (T, N, state_dim) tensor, returns (T, N) values tensor. Useful for value estimation during training."""
        return self.value_layers(self.shared_layers(states)).squeeze()
    
    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        self.load_state_dict(torch.load(path))
