import torch
from torch import nn



class WalkerPolicy(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 8, load_weights: bool = False):
        super().__init__()

        # load learned stored network weights after initialization
        if load_weights:
            self.load_weights()

    # TODO: implement a determine_actions() function mapping from (N, state_dim) states into (N, action_dim) actions

    def determine_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Given states tensor, returns the deterministic actions tensor.
        This would be used for control.

        Args:
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            actions (torch.Tensor): (N, action_dim) tensor
        """

        raise NotImplementedError("Implement the determine_actions function")



    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to save your network weights
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to load your network weights
        self.load_state_dict(torch.load(path))
