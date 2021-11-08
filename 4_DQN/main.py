import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers

# download data
(xs, ys), _ = datasets.mnist.load_data() # training/test data numpy format
print('datasets:', xs.shape, ys.shape, type(xs), type(ys))

# data preprocess
xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255. # xs to tensor
db = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(32) # tensor to dataset (to iterate)
it = db.__iter__()
x, y = it.__next__()
print(x.shape, y.shape, type(x), type(y)) # x,y are both tensors

# model and optimizer
model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])
optimizer = optimizers.SGD(learning_rate=0.001)

# go on with keyword and following video