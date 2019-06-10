# pytorch RL baseline 

all of the following methods are test on the `cartpole` module in `openai/gym`.
pytorch is selected for implementation.


## deep q network
 - dqn_cnn.py from [@official pytorch examples](https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py)
 - dqn_mlp.py from [@Pocuston](https://gist.github.com/Pocuston) (one issue is that the target net for model update isn't presented)
 
## advantage actor critic 
 - actor_critic.py from [@official pytorch examples](https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py)
 - a2c_mlp.py from [@floodsung](https://github.com/floodsung/a2c_cartpole_pytorch)