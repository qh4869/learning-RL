import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

# 定义输入tensor
input_tensor1 = tf.placeholder(tf.float32, [None, 748], name='in1')
input_tensor2 = tf.placeholder(tf.float32, [None, 4], name='in2')

# 定义模型变量和前馈过程，变量是按照name来区分的，get表示存在则获取不存在则新建，variable_scope表示variable name前加前缀
with tf.variable_scope('eval_net'):
    # c_names(collections_names) are the collections to store variables 可用于提取模型的变量值
    c_names, n_l1, w_initializer, b_initializer = \
        ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
        tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

    # first layer. collections is used later when assign to target net
    with tf.variable_scope('l1'):
        w1 = tf.get_variable('w1', [10, n_l1], initializer=w_initializer, collections=c_names)
        b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
        l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

    # second layer. collections is used later when assign to target net
    with tf.variable_scope('l2'):
        w2 = tf.get_variable('w2', [n_l1, 4], initializer=w_initializer, collections=c_names)
        b2 = tf.get_variable('b2', [1, 4], initializer=b_initializer, collections=c_names)
        q_eval = tf.matmul(l1, w2) + b2

# 定义loss和优化器
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.squared_difference(input_tensor2, q_eval))
with tf.variable_scope('train'):
    train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

# 更新
sess = tf.Session()
sess.run(tf.global_variables_initializer())
_, cost = sess.run([train_op, loss], feed_dict={s: batch_memory[:, :self.n_features], q_target: q_target})

# 预测或评估，可以多个模型同时
q_next, q_eval = sess.run( [q_next, q_eval], feed_dict={
    s_: batch_memory[:, -n_features:],  # fixed params
    s: batch_memory[:, :n_features],  # newest params
})

# 模型变量提取和替换，前三句放在模型的定义处即可
t_params = tf.get_collection('target_net_params')
e_params = tf.get_collection('eval_net_params')
replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
sess.run(replace_target_op)

# tensorboard
if output_graph:
    # $ tensorboard --logdir=logs
    # tf.train.SummaryWriter soon be deprecated, use following
    tf.summary.FileWriter("logs/", sess.graph)