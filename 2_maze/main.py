import pandas as pd
import numpy as np
from maze_env import Maze
import math

class q_learning:
    def __init__(self):
        self.actions = ['up', 'down', 'right', 'left'] # 顺序和环境模块一致
        self.n_actions = len(self.actions)
        self.q_table = pd.DataFrame(columns=list(range(self.n_actions)), dtype=np.float64)
        self.max_episode = 100
        self.env = Maze()
        self.epsilon = 0.9
        self.gamma = 0.9
        self.alpha = 0.01

    def is_state_existed(self, state):
        """
        检查是否q表中，某个状态存在，如果不存在则创建
        :param state:
        :return:
        """
        if str(state) not in self.q_table.index:
            state_item = pd.Series([0] * len(self.actions), index = list(range(self.n_actions)), name = str(state))
            self.q_table = self.q_table.append(state_item)
            return False
        else:
            return True

    def state_qval_equal_zero(self, state) -> bool:
        for val in self.q_table.loc[str(state)]:
            if not math.isclose(val, 0):
                return False
        return True

    def select_action(self, state):
        self.is_state_existed(state)
        if np.random.rand() > self.epsilon or self.state_qval_equal_zero(state):
            action = np.random.randint(0, self.n_actions)
        else:
            state_item = self.q_table.loc[str(state)]
            action = np.random.choice(state_item.loc[state_item==state_item.max()].index)
        return action

    def __call__(self):
        for episode in range(self.max_episode):
            step = 0
            state = self.env.reset()
            while state != 'terminal':
                self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action) # 环境自己也维护了state，所以只需告诉它action就行
                if next_state != 'terminal':
                    self.is_state_existed(next_state)
                    q_target = reward + self.gamma * self.q_table.loc[str(next_state)].max()
                else:
                    q_target = reward
                self.q_table.loc[str(state), action] += self.alpha * (q_target - self.q_table.loc[str(state), action])
                state = next_state
                step += 1
            self.env.render()
            print("episode:{0}, steps:{1}".format(episode, step))
        return self.q_table


if __name__ == "__main__":
    rl = q_learning()
    Q = rl()
    print(Q.to_string())

