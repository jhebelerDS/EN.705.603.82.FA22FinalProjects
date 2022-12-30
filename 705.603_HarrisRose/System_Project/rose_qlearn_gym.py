"""
File: rose_qlearn_gym.py
Author: Harris Rose

This is a module that performs Q-learning on the Blackjack-v1 environment from Open AI's gym.

"""

import random
import numpy as np
import statistics as stats
import gym
from typing import Dict, List, Tuple
import warnings

import gym_utils

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

class QLearningAgent:
    """
    Class represents a Q-learning agent for the gym balckjack environment

    Attributes
    ----------
    q_table : numpy array
        array that stores the q table for the algorithm
    learning rate: float
        parameter for rate of update of qtable
    discount_factor : float
        amount of discount for future actions
    game : gym.Env
        the environment from whcih to learn
    epsilon : float
        control exploration/exploitation
    epsilon_decay : float
        controls rate of decay of epsilon

    Methods
    ----------
    train(game, s_a_dims)
        trains the q_table
    get_action(state)
        return an action for a state using either exploration or exploitation
    update_q_table(state, action, reward, new_state, done)
        takes (s, a, r, s) data from a transition and updates the q table
    test()
        runs a series of black jack hands to evaluate current perfomance of q table
    """

    def __init__(self, game: gym.Env, s_a_dims: List[int] = [32, 11, 2, 2]):
        """
        Constructor for class
        :param game: envionment to test
        :param s_a_dims: dimensions of q_table
        """
        self.q_table = np.zeros(s_a_dims)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.game = game
        self.epsilon = 1.0
        self.epsilon_decay = 0.000005


    def train(self, num_episodes: int) -> None:
        """
        Method to train q table
        :param num_episodes: number of episode of self.game to play
        :return: None
        """
        for episode in range(num_episodes):
            state = self.game.reset()
            state = (state[0], state[1], int(state[2]))
            done = False
            while not done:
                action = self.get_action(state)
                new_state, reward, done, *_ = self.game.step(action)
                new_state = (new_state[0], new_state[1], int(new_state[2]))
                self.update_q_table(state, action, reward, new_state, done)
                state = new_state
            self.epsilon = np.exp(-self.epsilon_decay * episode)

            if episode % 5000 == 0 and episode > 0:
                test = self.test()
                print(f'Training Q Table, episode {episode}:  \n'
                      f'    Test Reward: {test[0]*100:0.2f}+-{test[1]*100:0.2f}%  \n'
                      f'    Qmax:{self.q_table.max():0.4f} Qmin: {self.q_table.min():0.4f} \n'
                      f'    Epsilon: {self.epsilon}')

        print(f'Q Table Updates Complete.')

    def get_action(self, state: Tuple[int, int, int]) -> int:
        """
        Method to get action dependent on epsilon
        :param state: describes the current state of the game
        :return: value for selected action
        """
        action_space = [0, 1]
        if random.random() < self.epsilon:
            # Explore
            action = random.choice(action_space)
        else:
            # Exploit
            action = self.q_table[state].argmax()
        return action

    def update_q_table(self, state:Tuple[int, int, int],
                       action: int,
                       reward: int,
                       new_state: Tuple[int, int, int],
                       done: bool) -> None:
        """
        Method to update q table for an observation.
        :param state: current state of env
        :param action: action taken
        :param reward: reward observed
        :param new_state: next state after action
        :param done: if next state is terminal
        :return: None
        """
        # Define parameters
        q_table = self.q_table
        lr = self.learning_rate
        df = self.discount_factor
        sa = state + (action,)

        # Ensure expected value of terminal states is zero
        max_exp_future_reward = 0 if done else np.max(q_table[new_state])

        # Update q_table
        q_table[sa] += lr * (reward + df * max_exp_future_reward - q_table[sa])

    def test(self) -> Tuple[float, float]:
        """
        Method to assess performance of model  To get an idea of the variability
        of the results, multiple runs of games are played and the mean return
        of the runs along with the st dev of the results is returned.
        :return: tuple with mean and st dev of performance
        """
        TEST_SETS = 10
        EPISODES_PER_SET = 1000
        set_returns = []
        for set in range(TEST_SETS):
            set_return = 0
            reward = 0
            for episode in range(EPISODES_PER_SET):
                state = self.game.reset()
                done = False
                while not done:
                    state = (state[0], state[1], int(state[2]))
                    action = self.q_table[state].argmax()
                    next_state, reward, done, *_ = self.game.step(action)
                    state = next_state
                set_return += reward
            set_returns.append(set_return / EPISODES_PER_SET)
        return stats.mean(set_returns), stats.stdev(set_returns)

if __name__ == '__main__':
    env = gym.make('Blackjack-v1')
    agent = QLearningAgent(env)
    agent.train(50000)
    np.save('TrainedModels/rose_gym_qtable.npy', agent.q_table)
    strategy = gym_utils.convert_gym_qtable_to_strategy(agent.q_table)
    strategy.to_pickle('TrainedStrategies/rose_qlearn_gym_strategy.pkl')


