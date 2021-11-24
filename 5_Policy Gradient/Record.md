### 我的理解 ###

- policy gradient的最大作用是，可以处理连续action 的问题，但是相比q值得方式，初级版本的性能较弱。
- 实现方法就是，用action 的分布来代替原理的q值，这里的例子中action是离散的，所以网络的输出是action 的分布律，连续遗留到后续的章节学习
- 因为从q值变成了概率，
  - 概率值不能像q值一样，step之间传递，能用的反馈只有reward，需要把reward处理成具有episode内有记忆的方式，分出这个action的价值对整局游戏影响到底有多大
  - 所以只能每个episode之后去learning，所有的step组成一整个batch，
  - 所以policy gradient我理解最核心的部分就是reward到vt的处理（跟reward和discount factor-gamma的设计有关，ref:https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/policy-gradient-softmax2/
  - 目前对vt的理解，训练的时候计算所选action就当做真是分布的across-entropy，总的loss是每个样本交叉熵的加权和，推荐的step上vt是正数，不推荐的step上vt是负

