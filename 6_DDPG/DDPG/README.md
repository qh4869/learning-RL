### DDPG

- actor critic的核心思想是把state to action的网络 以及评估网络（纯AC网络里用td error表示）分开，DDPG也是继承了这一根本思想，因此action是可以连续的，而且他们分开得更加彻底
  - actor网络直接负责从state输出action，而不是之前的概率分布，所以称为deterministic，该网络的目标是policy使得critic网络的Q值最大（数学上就是复合函数求导）
  - critic网络评估state和所选action的好坏，而不是之前单纯的state的好坏，所以评估的工作完全交给了critic网络，而不是之前有些含糊不清的，该网络的目标是跟随着reward而变化（就是DQN的原理）
- DDPG的模型物理意义是非常好理解的
- 继承了experience replay 和 target network的方法，促进收敛

