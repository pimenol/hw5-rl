import torch


def policy_gradient_loss_simple(logp: torch.Tensor, tensor_r: torch.Tensor) -> torch.Tensor:
    """
    Given the log-probabilities of the policy and the rewards, compute the scalar loss
    representing the policy gradient.

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        tensor_r: (T, N) tensor of rewards, detached from any computation graph

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: start by calculating the cumulative returns of the trajectories, and then compute the policy gradient
    # with torch.no_grad():

    raise NotImplementedError("policy_gradient_loss_simple not implemented!")


def discount_cum_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Given the rewards and the discount factor gamma, compute the discounted cumulative sum of rewards.
    The cumulative sum follows the reward-to-go formulation. This means we want to compute the discounted
    trajectory returns at each timestep. We do that by calculating an exponentially weighted
    sum of (only) the following rewards.
    i.e.
    $R(\tau_i, t) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$

    Args:
        rewards: (T, N) tensor of rewards
        gamma: discount factor

    Returns:
        discounted_cumulative_returns: (T, N) tensor of discounted cumulative returns
    """
    # TODO: implement the discounted cummulative sum, i.e. the discounted returns computed from rewards and gamma

    raise NotImplementedError("discount_cum_sum not implemented!")


def policy_gradient_loss_discounted(logp: torch.Tensor, tensor_r: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Given the policy log0probabilities, rewards and the discount factor gamma, compute the
    policy gradient loss using discounted returns.

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        tensor_r: (T, N) tensor of rewards, detached from any computation graph
        gamma: discount factor

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # with torch.no_grad():
    # TODO: compute discounted returns of the trajectories from the reward tensor, then compute the policy gradient

    raise NotImplementedError("policy_gradient_loss_discounted not implemented!")


def policy_gradient_loss_advantages(logp: torch.Tensor, advantage_estimates: torch.Tensor) -> torch.Tensor:
    """
    Given the policy log-probabilities and the advantage estimates, compute the policy gradient loss

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        advantage_estimates: (T, N) tensor of advantage estimates

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: compute the policy gradient estimate using the advantage estimate weighting

    raise NotImplementedError("policy_gradient_loss_advantages not implemented!")


def value_loss(values: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    """ 
    Given the values and the value targets, compute the value function regression loss
    """
    # TODO: compute the value function L2 loss

    raise NotImplementedError("value_loss not implemented!")


def ppo_loss(p_ratios, advantage_estimates, epsilon):
    """ 
    Given the probability ratios, advantage estimates and the clipping parameter epsilon, compute the PPO loss
    based on the clipped surrogate objective
    """
    # TODO: compute the PPO loss

    raise NotImplementedError("ppo_loss not implemented!")
