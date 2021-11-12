from maze_env import Maze
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers

np.random.seed(1)
tf.random.set_seed(1)

class Deep_Q_Network:
    def __init__(self, n_actions, n_features, epsilon, epsilon_step=0, max_episode=200,
                 memory_size=2000, batch_size=32, decay_factor=0.9, learning_rate=0.01, replace_target_step=200):
        self.n_actions = n_actions
        self.n_features = n_features
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.epsilon_step = epsilon_step # 训练阶段，每个episode更新epsilon
        self.eval_model = keras.Sequential([
            layers.Dense(10,
                         activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.3),
                         bias_initializer=initializers.Constant(value=0.1)
                         ),
            layers.Dense(
                self.n_actions,
                kernel_initializer=initializers.RandomNormal(stddev=0.3),
                bias_initializer=initializers.Constant(value=0.1)
            )
        ])
        self.learning_rate = learning_rate
        self.eval_model.compile(optimizer=optimizers.RMSprop(learning_rate=self.learning_rate), loss='mse')
        self.target_model = keras.Sequential([
            layers.Dense(10,
                         activation='relu',
                         kernel_initializer=initializers.RandomNormal(stddev=0.3),
                         bias_initializer=initializers.Constant(value=0.1)
                         ),
            layers.Dense(
                self.n_actions,
                kernel_initializer=initializers.RandomNormal(stddev=0.3),
                bias_initializer=initializers.Constant(value=0.1)
            )
        ])
        self.eval_model.build(input_shape=(None, 2))
        self.target_model.build(input_shape=(None, 2))
        self.eval_model.summary()
        self.target_model.summary()
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2)) # two features, one reward, one action
        self.memory_cnt = 0
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.learning_cnt = 0
        self.replace_target_step = replace_target_step

    def choose_action(self, state) -> int:
        state = state[np.newaxis, :] # tensor是不可变变量，值传递
        if np.random.rand() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action_qvals = self.eval_model.predict(state)
            print("state & Q_Val:", state, action_qvals)
            action = np.argmax(action_qvals)

        return action

    def save_memory(self, state, action, reward, next_state):
        row_index = self.memory_cnt % self.memory_size
        self.memory[row_index, :] = np.concatenate((state, [action ,reward], next_state))
        self.memory_cnt += 1

    def learn(self):
        idx_max = self.memory_size if self.memory_cnt >= self.memory_size else self.memory_cnt
        replay_rows = np.random.randint(0, idx_max, self.batch_size, dtype=int)
        curr_state_batch = self.memory[replay_rows, :self.n_features]
        next_state_batch = self.memory[replay_rows, -self.n_features:]
        action_batch = self.memory[replay_rows, self.n_features].astype(int)
        reward_batch = self.memory[replay_rows, self.n_features + 1]
        curr_q_val_batch = self.eval_model(curr_state_batch).numpy()
        next_q_val_batch = self.target_model(next_state_batch).numpy()
        next_q_val_max_batch = np.max(next_q_val_batch, axis=1)
        target_batch_1d = reward_batch + self.decay_factor * next_q_val_max_batch # 只有action对应的那一维
        target_batch = curr_q_val_batch.copy()
        batch_idx = np.arange(self.batch_size)
        target_batch[batch_idx, action_batch] = target_batch_1d

        cost = self.eval_model.train_on_batch(curr_state_batch, target_batch)

        self.learning_cnt += 1
        if self.learning_cnt % self.replace_target_step == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print("target replace\n")

    def plot_cost(self):
        pass

    def update_epsilon(self):
        self.epsilon += self.epsilon_step


def train_maze():
    """
    main loop for all episodes
    :return:
    """
    step = 0
    for episode in range(rl.max_episode):
        print("episode, epsilon:", episode, rl.epsilon)
        state = env.reset()
        while True:
            env.render()
            action = rl.choose_action(state)
            state_, reward, done = env.step(action)
            rl.save_memory(state, action, reward, state_)
            if (step > 200) and (step % 5 == 0):
                rl.learn()
            state = state_
            step += 1
            if done:
                rl.update_epsilon()
                break
    env.destroy()


def run_maze():
    rl.eval_model.load_weights('train_model.h5')
    for episode in range(rl.max_episode):
        print("episode, epsilon:", episode, rl.epsilon)
        state = env.reset()
        while True:
            env.render()
            action = rl.choose_action(state)
            state_, _, done = env.step(action)
            state = state_
            if done:
                break
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # rl = Deep_Q_Network(env.n_actions, env.n_features, epsilon=0.9, epsilon_step=0)
    # env.after(100, train_maze)
    rl = Deep_Q_Network(env.n_actions, env.n_features, epsilon=0.9)
    env.after(100, run_maze)
    env.mainloop()
    rl.plot_cost()
    rl.eval_model.save_weights('train_model.h5')