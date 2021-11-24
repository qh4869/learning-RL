"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, optimizers

# reproducible
np.random.seed(1)
tf.random.set_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

    def custom_loss(self, tf_vt):
        def loss(y_true, y_pred):
            neg_log_prob = tf.reduce_sum(-tf.math.log(y_true) * y_pred, axis=1)
            res = tf.reduce_mean(neg_log_prob * tf_vt)
            return res
        return loss

    def _build_net(self):
        self.model = keras.Sequential([
            layers.Dense(10,
                         activation='tanh',
                         kernel_initializer=initializers.RandomNormal(stddev=0.3),
                         bias_initializer=initializers.Constant(value=0.1)
                         ),
            layers.Dense(self.n_actions,
                         kernel_initializer=initializers.RandomNormal(stddev=0.3),
                         bias_initializer=initializers.Constant(value=0.1)
                         ),
            layers.Softmax()
        ])
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.lr))
        self.model.build(input_shape=(None, self.n_features))
        self.model.summary()

    def choose_action(self, observation):
        prob_weights = self.model.predict(observation[np.newaxis, :])
        try:
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        except:
            print('bug')
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()


        self.model.compile(loss=self.custom_loss(discounted_ep_rs_norm))
        self.model.train_on_batch(np.vstack(self.ep_obs), tf.one_hot(np.array(self.ep_as), self.n_actions))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



