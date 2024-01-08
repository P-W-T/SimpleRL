
# Policy gradients 


## Description
Simple clean implementation of the policy gradients algorithm for reinforcement learning. 

Based on:
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- Duan, Y., Chen, X., Houthooft, R., Schulman, J., & Abbeel, P. (2016, June). Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning (pp. 1329-1338). PMLR.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.

## Testing
This algorithm was tested with any option (cuda/cpu with batching or without) and shows good performance on:
- CartPole-v1: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- InvertedPendulum-v4: https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/
- HalfCheetah-v4: https://gymnasium.farama.org/environments/mujoco/half_cheetah/
