### 演员评论家

>  DDPG的前身，收敛性不好，但是是学习DDPG的基础

- actor：policy gradient网络，可以解决连续动作的问题
- critic: value-based policy，可以单步更新，用来学习policy gradient中的vt，这里称作td-error
- 示意图：![](AC.png)

- DQN输出的是更新的差值，而且评估的是当前state的value（不是action），用来告诉actor网络当前step(从state到next_state) logPr的重要程度（代替policy gradient的loss加权和）。
