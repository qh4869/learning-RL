### Ref 

- 笔记本：关于tensor的基本操作
- 官网资料：学习->tensorflow



### 区分概念：

- loss：训练中的目标函数
- metric：用来评估模型好坏，不用做训练
- episode: RL中一个回合，一局完整的交互
- batch: 得到一个loss
- step: RL中的一步
- sample: batch中的一条样本
- epoch: database的一次训练



### 激活函数

- 深层网络tanh和sigmoid端值区域饱和，容易收敛慢，常用relu
- 浅层网络一般可以采用tanh
- tanh输出是-1和+1之间，经常网络的输出喜欢0~1之间，sigmoid可以看成tanh的偏移版本，但少了tanh类似数据中心化的效果
- softmax，用户把输出归一化和等于1，概率的物理意义，也常用于输出



### 动态图建立网络

1. 定义layer结构
   - keras自带的
   - 从keras.layers.layer类继承
     - 定义构造函数：包括变量和初始化
     - 定义call()方法，层的计算过程
2. 定义网络模型（官网给的三种方式，ref：学习->tensorflow->指南->keras）
   - keras.Sequential类：是keras.model的子类，有一些需要设定的特性已经固定好了
     - 前馈过程不需要自己定义
   - 从keras.model继承：该类是一个抽象类，需要自定义各种特性
     - 定义构造函数，包括层的结构，model可以看做layer的容器类
     - 定义call()方法，包括网络的前馈过程
   - 函数式API：比较灵活建立有向无环图和多输入多输出模型，非有向五环图ref官网资料
3. 初始化model：确定网络输入shape
   - 自动初始化，模型第一次运行时候根据输入数据决定网络输入的shape
   - 有些方法已经定义了输入的shape，例如函数式API
   - model.build(input_shape=(None, 28*28))
4. 确定优化器和loss：
   - keras.model的compile方法（用于支持后续的fit/train_on_batch方法）：
     - model.compile(optimizer=)
     - model.compile(loss=)
   - 自定义loss
     - 定义loss函数，传入model.compile中，（如果loss与除了y_pred和y_true值以外的tensor有关，可以写成闭包形式传入，遗留）
     - 把loss写进网络的最后一层，然后compile的时候loss给None，这样写实际网络的输出还得从中间拉出来
   - 不使用keras的compile：手动求loss，求导和训练更新（比较灵活）
     - ref例程
5. 更新（训练）：
   - model.compile() and train: 
     - fit or train_on_batch,  ref: 4_DQN中main和test，官网资料，资料网络课程视频
   - 不使用keras compile: 手动求loss，求导和训练更新（比较灵活）
     - ref例程
6. 模型的应用
   - evaluate：test mode评估性能
   - predict: 输出和输入类型一致
   - \__call__: 不管输入什么类型，输出tensor
7. 相同模型的参数复制
   - ref：4_DQN
8. tensorboard:
   - ref: tf1_py36的example