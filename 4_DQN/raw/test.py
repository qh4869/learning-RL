import tensorflow as tf

################# 静态图方式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32) # 静态图的输入, 如果shape参数的一维是None，代表batch方式的输入，batch size自动识别

out = tf.multiply(input1, input2) # 静态图输入输出都是tensor，运算需要使用预留的函数

with tf.Session() as sess: # 静态图的运行需要这样写
    res = sess.run(out, feed_dict={input1: 2, input2: 3})

print(res)

############## tf1定义变量
var0 = tf.get_variable('var_name', [1]) # 计算图中的节点是以name来区分的(即'var_name')，var0是编程中的变量，此时表示计算图中叫'var_name'的node
with tf.variable_scope('scope1'): # 加了这个scope之后，节点的名字前面加上这个scope，防止变量太多之后，编程的变量名字前缀太乱，这个scope可以嵌套使用
    var1 = tf.get_variable('var', [1])
with tf.variable_scope('scope2'):
    var2 = tf.get_variable('var', [2])
with tf.variable_scope('scope1', reuse=True): # 如果想再次使用同一个参数（node），就用reuse功能，遗留：为啥不复用变量var1？
    var3 = tf.get_variable('var', [1])

print(var1.name)
print(var2.name)
print(var3.name)