"""
Script to train mcguire models and save parameters and strategies
Author: Harris Rose
"""

import numpy as np
import torch
import os

import mcguire_dqn
import mcguire_qlearn
import strategies
import gym_utils


def convert_mcguire_dqn_to_strategy(filename: str):
    """
    Function to create strategy datatable from dqn parameters
    :param filename: str - filename to save
    :return: returns strategy dataframe
    """
    action_map = {1: 'H', 0: 'S'}
    strategy = strategies.BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
    agent = mcguire_dqn.Agent(state_size=3, action_size=2, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load(filename))
    agent.qnetwork_local.eval()
    for row in strategy.index:
        for col in strategy.columns:
            state = gym_utils.parse_strategy_state_to_gym_state(row, col)
            state = np.array([state[0] / 32, state[1] / 10, int(state[2])])
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = agent.qnetwork_local(state).cpu().data.numpy()
            action = action_values.argmax()
            strategy.loc[row, col] = action_map[action]
    return strategy


if __name__ == '__main__':

    # Check and create directory structure
    if not os.path.exists('TrainedModels'):
        os.mkdir('TrainedModels')
    if not os.path.exists('TrainedStrategies'):
        os.mkdir('TrainedStrategies')

    # Train mcguire Q Learning and save qtable and strategy
    mcg_q_table_filename = 'TrainedModels/mcguire_qlearn.npy'
    mcg_q_strategy_filename = 'TrainedStrategies/mcguire_Q_strategy.pkl'
    mcg_qtable = mcguire_qlearn.train(mcg_q_table_filename)
    mcg_q_strategy = gym_utils.convert_gym_qtable_to_strategy(mcg_qtable)
    mcg_q_strategy.to_pickle(mcg_q_strategy_filename)
    print("Q table trained")

    # Train mcguire DQN and save nn weights and strategy
    mcg_dqn_filename = 'TrainedModels/mcguire_dqn.pth'
    mcg_dqn_strategy_filename = 'TrainedStrategies/mcguire_DQN_strategy.pkl'
    mcguire_dqn.train(mcg_dqn_filename)
    mcg_dqn_strategy = convert_mcguire_dqn_to_strategy(mcg_dqn_filename)
    mcg_dqn_strategy.to_pickle(mcg_dqn_strategy_filename)
    print('DQN trained')
