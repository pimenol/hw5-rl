# UROB HW5 - Deep reinforcement learning through policy optimization

### About

This task should provide a minimalist introduction to policy gradient methods for deep reinforcement learning and illustrate the relative effectiveness of each method using simple, interpretable environments.

### Code

Use python 3.10/11/12 for this assignment. The script `make_venv.sh` will create a virtual environment with the necessary dependencies.

You can install the necessary libraries yourself by running `pip install -r requirements.txt`.

Please make sure to use the versions specified in the `requirements.txt` file.

Once this is done, you can test whether the environment is set up correctly by running `python -m pytest -s`.

### Pendulum environment

First, in the notebook `task.ipynb` you will go through methods starting with Vanilla Policy Gradient/REINFORCE and ending with PPO, the current state-of-the-art.

Your task is to implement the loss functions used for training, which in some cases includes the computation of specific reward or value terms.

All of the implemented code should be in the file `solution.py`. You only need to submit this file to Brute (along with files for the optional part if you choose to do it) as a zip/tar archive.

This is the mandatory part of the assignment and is worth **10 points** in total.

- `policy_gradient_loss_simple` - Vanilla Policy Gradient/REINFORCE - **1 point**

- `discount_cum_sum` - general function for computing discounted cumulative sums - **2 points**

- `policy_gradient_loss_discounted` - Policy Gradient with discounted rewards - **1 point**

- `policy_gradient_loss_advantages` - Policy Gradient with advantages - **2 points**

- `value_loss` - Value loss for training a value function - **2 points**

- `ppo_loss` - Proximal Policy Optimization loss - **2 points**

### Walker environment

Using the methods you implemented in the first part and the already-provided functions for working with the (pendulum) environment, train a policy for the Walker environment.

**The objective is to travel as far as possible in the positive x-direction in a limited number of steps (TBD).**

This is evaluated in a tournament with up to 3 additional points awarded as follows:

- **1 point** for traversing least 1 meter in the Walker environment (less than 5 simulation seconds)

- **1 point** if your policy falls within top 50% of the submissions

- **1 point** if your policy falls within top 10% of the submissions

Make sure you follow the exact specification described in `WalkerPolicy.py` and `walker_training.ipynb` to ensure your policy is evaluated correctly.

### Troubleshooting

If you run into some issues that you or an LLM of your choice can't resolve, please let me know at `korcadav@fel.cvut.cz`. I will ask my LLM :)
