import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# download data
(x_train, y_train), _ = datasets.mnist.load_data() # training/test data numpy format
print('datasets:', x_train.shape, y_train.shape, type(x_train), type(y_train))

# data preprocess
x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255. # xs to tensor
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_train_onehot = tf.one_hot(y_train_tensor, depth=10)
db = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_onehot)).batch(32) # tensor to dataset (to iterate)
it = db.__iter__()
x, y = it.__next__()
print(x.shape, y.shape, type(x), type(y)) # x,y are both tensors

# model and optimizer prepare
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])
optimizer = optimizers.SGD(learning_rate=0.001)
model.build(input_shape=(None, 28*28)) # 告诉模型输入层的size
model.summary()

# train
def train_epoch(epoch):
    for step, (x, y) in enumerate(db): # each batch for one loss and grad
        # go forward
        with tf.GradientTape() as tape:
            x = tf.reshape(x, (-1, 28*28)) # 传给网络的数据第一维，必须是batch维度
            out = model(x) # 训练时候用model， 如果单纯预测用mode.predict()返回类型不是tensor，而是跟输入一致
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0] # one-hot ???
        # optimize
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # print
        if step % 100 == 0:
            print(epoch, step, loss.numpy())


if __name__ == "__main__":
    for epoch in range(30):
        train_epoch(epoch)