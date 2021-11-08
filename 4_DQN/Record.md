### DQN ###

- state太多，不用表维护Q值，而是用网络来拟合Q值

- experience replay （因为是off-policy，打乱experience相关性，神经网络收敛更有效）

- fixed Q-targets（也是神经网络引起的后续影响，为了网络更新效果更好）

  > 莫烦的解释：这里要了解 qlearning 和 function approximator 的一些细节了。用一个神经网络当 q learning 的 function approximator，会使得对动作价值的评估存在偏差被放大的状况，因为这个function approximator 不是稳定的。为了稳定 target function approximator, 所以就采用了一个变化不快的网络。这样对于学习稳定性的提升十分有帮助。

