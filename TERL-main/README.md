# TERL
Code for "Two-Stage Evolutionary Reinforcement Learning for Enhancing Exploration and Exploitation". Our codes are based on the implementation of TD3 and ERL.

# Dependency
python 3.8.13, pytorch 1.8.2, gym 0.23.1, mujoco210, tensorboard 2.13.0

# Run
python TERL.py -env 'Walker2d-v2' 

# Data
Visualize the learning curves: tensorboard --logdir .\learning_curves\Walker2d-v2\

# Note
Although the default value of exploration ratio is 0.25 in all environments, it's not the best value for each environment. For example, a smaller value will be better for HalfCheetah.

