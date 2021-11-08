# Q learning
import time
import numpy as np
import pandas as pd

N_STATES = 6   # the length of the 1 dimensional world
FRESH_TIME = 0.3    # fresh time for one move
ACTIONS = ['left', 'right']
MAX_EPISODES = 13
EPSILON = 0.9 # epsilon greedy parameter
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor in RL

def update_env(S, episode, step_counter):
    """
    环境显示刷新
    :param S: 状态
    :param episode: 用于打印
    :param step_counter: 用于打印
    :return:
    """
    # This is how environment be updated
    env_list = ['-']*(N_STATES) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def create_q_table():
    """
    生成空白的Q表
    :return: Q表, panda.DataFrame
    """
    array = np.zeros((N_STATES, len(ACTIONS)))
    table = pd.DataFrame(array, columns=ACTIONS)
    return table


def select_action(q_table, S) -> str:
    """
    根据epsilon greedy选择action
    :param q_table:
    :param S: curr state
    :param step:
    :return:
    """
    if np.random.rand() > EPSILON or (q_table.loc[S] == 0).all():
        action = np.random.choice(ACTIONS)
    else:
        q_line = q_table.loc[S]
        action = q_line.idxmax()
    return action


def state_transform(state, action):
    if state == 0 and action == 'left':
        return 0, 0
    elif state == N_STATES - 1 and action == 'right':
        return 'terminal', 1
    elif action == 'left':
        return state-1, 0
    else:
        return state+1, 0


def q_learning():
    q_table = create_q_table()
    for episode in range(MAX_EPISODES):
        step = 0
        state = 0
        while state != 'terminal':
            update_env(state, episode, step)
            action = select_action(q_table, state)
            next_state, reward = state_transform(state, action)
            if next_state != 'terminal':
                q_target = reward + GAMMA * q_table.loc[next_state].max()
            else:
                q_target = reward
            q_table.loc[state, action] += ALPHA * (q_target - q_table.loc[state, action])
            state = next_state
            step += 1
        update_env(state, episode, step)
    return q_table


def sarsa():
    q_table = create_q_table()
    for episode in range(MAX_EPISODES):
        step = 0
        state = 0
        action = select_action(q_table, state)
        while state != 'terminal':
            update_env(state, episode, step)
            next_state, reward = state_transform(state, action)
            if next_state != 'terminal':
                next_action = select_action(q_table, next_state)
                q_target = reward + GAMMA * (q_table.loc[next_state, next_action])
            else:
                q_target = reward
            q_table.loc[state, action] += ALPHA * (q_target - q_table.loc[state, action])
            action = next_action
            step += 1
            state = next_state
        update_env(state, episode, step)
    return q_table


if __name__ == "__main__":
    # q = q_learning()
    q = sarsa()
    print(q)
