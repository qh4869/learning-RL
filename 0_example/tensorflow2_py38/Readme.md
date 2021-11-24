### Ref 笔记本

- 关于tensor的基本操作



### 区分概念：遗留

- loss
- metric
- episode
- batch
- step
- sample
- epoch



### 动态图建立网络

1. 定义layer结构
   - keras自带的
   - 从keras.layers.layer类继承
     - go on !!!
2. 定义网络模型
   - keras.Sequential类：是keras.model的子类，有一些需要设定的特性已经固定好了
     - 前馈过程不需要自己定义
   - 从keras.model继承：该类是一个抽象类，需要自定义各种特性
     - 定义构造函数，包括层的结构，model可以看做layer的容器类
     - 定义call()方法，包括网络的前馈过程
3. 确定优化器和loss：
   - keras.model的compile方法（用于支持后续的fit/train_on_batch方法）：
     - model.compile(optimizer=)
     - model.compile(loss=)
   - 自定义loss
     - 定义loss函数，传入model.compile中，（如果loss与除了y_pred和y_true值以外的tensor有关，可以写成闭包形式传入，遗留）
     - 把loss写进网络的最后一层，然后compile的时候loss给None，这样写实际网络的输出还得从中间拉出来
   - 不使用keras的compile：手动求loss，求导和训练更新（比较灵活）
     - ref例程
4. 更新（训练）：
   - keras compile and train: 
     - fit or train_on_batch,  ref: 4_DQN中main和test，资料网络课程视频
   - 不使用keras compile: 
     - ref例程
5. keras.model类常用的方法
   - compile
   - fit
   - evaluate：遗留
   - predict: 输出和输入类型一致
   - \__call__: 不管输入什么类型，输出tensor