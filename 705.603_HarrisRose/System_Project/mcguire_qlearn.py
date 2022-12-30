"""
This module contains q learning code from the article used in
Assignment 10

A train function is provided at the bottom to allow training from an
outside script.  No documentation was added to code taken from:

https://github.com/trevormcguire/blogposts/blob/main/RL/RL%20Blackjack.ipynb

"""

import gym
import numpy as np
import random

if gym.__version__ > "0.25.0":
    print("must use gym 0.25")
    exit()


def get_state_idxs(state):
    idx1, idx2, idx3 = state
    idx3 = int(idx3)
    return idx1, idx2, idx3


def update_qtable(qtable, state, action, reward, next_state, alpha, gamma):
    curr_idx1, curr_idx2, curr_idx3 = get_state_idxs(state)
    next_idx1, next_idx2, next_idx3 = get_state_idxs(next_state)
    curr_state_q = qtable[curr_idx1][curr_idx2][curr_idx3]
    next_state_q = qtable[next_idx1][next_idx2][next_idx3]
    qtable[curr_idx1][curr_idx2][curr_idx3][action] += alpha * (
            reward + gamma * np.max(next_state_q) - curr_state_q[action])
    return qtable


def get_action(qtable, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        idx1, idx2, idx3 = get_state_idxs(state)
        action = np.argmax(qtable[idx1][idx2][idx3])
    return action


def train_agent(env,
                qtable: np.ndarray,
                num_episodes: int,
                alpha: float,
                gamma: float,
                epsilon: float,
                epsilon_decay: float) -> np.ndarray:
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while True:
            action = get_action(qtable, state, epsilon)
            new_state, reward, done, info = env.step(action)
            qtable = update_qtable(qtable, state, action, reward, new_state, alpha, gamma)
            state = new_state
            if done:
                break
        epsilon = np.exp(-epsilon_decay * episode)
    return qtable

env = gym.make("Blackjack-v1")
env.action_space.seed(42)

def train(filename:str):

    """
    Function to train model taken from:
    https://github.com/trevormcguire/blogposts/blob/main/RL/RL%20Blackjack.ipynb
    :param filename:location to save qtable as .npy file
    :return: numpy array - qtable
    """


    # get initial state
    state = env.reset()

    state_size = [x.n for x in env.observation_space]
    action_size = env.action_space.n

    qtable = np.zeros(state_size + [action_size])  # init with zeros

    alpha = 0.3  # learning rate
    gamma = 0.1  # discount rate
    epsilon = 0.9  # probability that our agent will explore
    decay_rate = 0.005

    # training variables
    num_hands = 500_000

    qtable = train_agent(env,
                         qtable,
                         num_hands,
                         alpha,
                         gamma,
                         epsilon,
                         decay_rate)

    np.save(filename, qtable)
    return qtable
