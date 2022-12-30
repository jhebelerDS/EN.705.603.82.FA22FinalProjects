"""
Helper functions for system project
Author: Harris Rose
"""
import gym
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import strategies

DEALER_STRATEGY_MAP = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T'}
DEALER_POLICY_MAP = {v: k for k, v in DEALER_STRATEGY_MAP.items()}


def evaluate_strategies(game: gym.Env,
                        strategies: Dict[str, pd.DataFrame],
                        episodes: int = 5000,
                        verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Function to evaluate multiple strategies, returning a dataframe with statistics and a dict of
    results for graphing the outcomes
    :param game: gym environment to test
    :param strategies: the strategies to be tested
    :param episodes: number of hands to play
    :param verbose: flag to allow debugging
    :return: returns a dataframe summary of results and the raw results for plotting
    """
    # results is for keeping the tokens for plotting
    results = {}
    # data is for building the dataframe
    data = {'Policy Score': [], 'Expected Return': []}
    # index for storing dataframe index
    index = []
    for strategy in strategies.keys():
        index.append(strategy)
        if strategy == 'naive':
            data['Policy Score'].append(np.nan)
        else:
            data['Policy Score'].append(score_strategy(strategies[strategy], strategies['expert']) * 100)
        results[strategy] = []
        net_reward = 0
        for e in range(episodes):
            reward = play_one_bj_hand(game, strategies[strategy], verbose)
            net_reward += reward
            results[strategy].append(net_reward)
        data['Expected Return'].append(net_reward / episodes * 100)
    df = pd.DataFrame(data, index=index).sort_values('Expected Return', ascending=False)
    df = df.applymap(lambda x: 'n/a' if str(x) == 'nan' else str(f'{x:0.1f}%'))
    return df, results


def plot_results(results: dict, order: List):
    """
    Function to plot results from evaluate_strategies
    :param results: dict of results from evaluate strategies
    :param order: order in which to plot results.  Usually the sorted index from df from

    :return: None
    """
    for s in order:
        plt.plot(range(1, len(results[s]) + 1), results[s], '.-', label=s)
    plt.ylabel('Total Return')
    plt.xlabel('Hands Played')
    plt.legend()
    plt.show()


def score_strategy(learned: pd.DataFrame, expert: pd.DataFrame) -> float:
    """
    Method to compare learned strategy to expert strategy
    :param learned: dataframe strategy to compare to expert
    :param expert: expert dataframe strategy
    :return: percentage correct as float
    """
    assert learned.shape == expert.shape
    score = 0
    items = learned.shape[0] * learned.shape[1]
    for row in learned.index:
        for col in learned.columns:
            if learned.loc[row, col] == expert.loc[row, col]:
                score += 1
    return score / items


def play_one_bj_hand(game: gym.Env, strategy: pd.DataFrame, verbose: bool = False) -> int:
    """
    Function to play one hand of blackjack using a strategy
    :param game: environment to test
    :param strategy: strategy to test
    :param verbose: flag for debugging
    :return: reward from hand
    """
    ACTION_MAP = {0: 'Stand', 1: 'Hit', 2: 'Double', 3: 'Surrender', 4: 'Split'}
    reward = 0
    net_reward = 0
    state = game.reset()
    done = False
    while not done:
        action = get_action_from_strategy(state, strategy)
        new_state, reward, done, *_ = game.step(action)
        net_reward += reward
        if verbose: print(f'State: {state}  Action: {ACTION_MAP[action]}  Reward: {reward}')
        state = new_state
    if verbose: print(f'Game Finished.  Reward: {reward}')
    return net_reward


def get_action_from_strategy(state: Tuple[int, int, bool],
                             strategy: pd.DataFrame,
                             action_space: List[int] = [0, 1]) -> int:
    """
    Function to get action from strategy
    :param state: state in gym format
    :param strategy: dataframe of strategy
    :param action_space: actions from which to choose.
    :return: int representing action chosen
    """
    if type(strategy) == str and strategy == 'random':
        return random.choice(action_space)

    ACTION_MAP_TO_INT = {'S': [0], 'H': [1], 'Dh': [2, 1], 'Ds': [2, 0],
                         'Uh': [3, 1], 'Usp': [3, 1], 'Us': [3, 0], 'SP': [4]}
    row, col = parse_gym_state_to_strategy_state(state)
    actions = ACTION_MAP_TO_INT[strategy.loc[row, col]]
    return actions[0] if actions[0] in action_space else actions[1]


def parse_gym_state_to_strategy_state(state: Tuple[int, int, bool]) -> Tuple[str, str]:
    """
    Function to convert gym state to strategy state
    :param state: gym state
    :return: strategy state
    """
    p, d, s = state
    row = 'S' if s else 'H'
    row += str(p)
    col = DEALER_STRATEGY_MAP[d]
    return row, col


def parse_strategy_state_to_gym_state(row: str, col: str) -> Tuple[int, int, int]:
    """
    Function to convert strategy state to gym state
    :param row: row of strategy table
    :param col: col of strategy table
    :return: state in gym format
    """
    soft = 1 if row[0] == 'S' else 0
    player_value = int(row[1:])
    dealer_value = DEALER_POLICY_MAP[col]
    return player_value, dealer_value, soft


def convert_gym_qtable_to_strategy(qtable: np.ndarray) -> pd.DataFrame:
    """
    Function to convert numpy q table to strategy dataframe
    :param qtable: qtable
    :return: strategy dataframe
    """
    action_map = {1: 'H', 0: 'S'}
    strategy = strategies.BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
    for row in strategy.index:
        for col in strategy.columns:
            state = parse_strategy_state_to_gym_state(row, col)
            strategy.loc[row, col] = action_map[qtable[state].argmax()]
    return strategy


if __name__ == "__main__":
    pass
