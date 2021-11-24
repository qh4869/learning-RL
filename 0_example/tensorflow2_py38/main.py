import tensorflow as tf
from tensorflow import keras

# 例程，详细内容ref Readme.md

# 定义网络模型

## 利用keras.Sequential类
model1 = keras.model = keras.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10)
])

## 从keras.model继承
class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('network_name')
        self.layer1 = keras.layers.Dense(10, activation='relu')
        self.layer2 = keras.layers.Dense(num_actions, activation=None)
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        res = self.layer2(layer1)
        return res
model2 = Eval_Model(1)

## 不使用keras的API compile，可以把下面这些代码和model的定义封装在一起，或者直接在自定义model类中写
x = tf.constant(1) # 输入
y_true = tf.constant(1) # 真值
with tf.GradientTape() as tape:
    out = model2(x)
    loss = tf.reduce_sum(tf.square(out - y_true)) / x.shape[0]
grads = tape.gradient(loss, model2.trainable_variables)
keras.optimizers.SGD(learning_rate=0.001).apply_gradients(zip(grads, model2.trainable_variables))