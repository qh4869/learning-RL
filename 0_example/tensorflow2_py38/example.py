import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
np.random.seed(1)
tf.random.set_seed(1)

# 子类化自定义layer
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b # @在python3.5之后就是矩阵相乘


# 模型定义
## 利用keras.Sequential类
model1 = keras.model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), bias_initializer=tf.constant_initializer(0.1)),
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
## 函数式API
inputs = keras.Input(shape=(784,)) # 这里不用管batch size的维度
x = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10)(x)
model3 = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
## 打印信息
model.summary()
keras.utils.plot_model(model3, "my_first_model_with_shape_info.png", show_shapes=True)


## 手动求loss，求导和更新
x = tf.constant(1) # 输入
y_true = tf.constant(1) # 真值
with tf.GradientTape() as tape:
    out = model2(x)
    loss = tf.reduce_sum(tf.square(out - y_true)) / x.shape[0]
grads = tape.gradient(loss, model2.trainable_variables)
keras.optimizers.SGD(learning_rate=0.001).apply_gradients(zip(grads, model2.trainable_variables))