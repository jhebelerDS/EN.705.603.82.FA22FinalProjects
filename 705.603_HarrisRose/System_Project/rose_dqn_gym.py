"""
File: rose_dqn_gym.py
Author: Harris Rose

This is a module that performs Deep Q-learning on the Blackjack-v1 environment from Open AI's gym.

"""
import random
import statistics as stats
import warnings
from collections import namedtuple, deque
from typing import List

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym_utils
import strategies

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = 200  # replay buffer size
BATCH_SIZE = 50  # minibatch size

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = .001
UPDATE_EVERY = 4
SYNC_TARGET_EPISODES = 100
EPSILON_DECAY_LAST_FRAME = 10000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
MAX_EPISODES = 2500


class DQN(nn.Module):
    """
    Neural Network Used for Agent to Approximate Q-Values.  Adapted from:
    https://github.com/trevormcguire/blogposts/blob/main/RL/RL%20Blackjack.ipynb
    -------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    """

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Named tuple to store transitions in memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    """
    Class to create memory for DQN training.

    Attributes
    ----------
    memory: deque
        A deque to store the prior observations.

    Methods
    ----------
    push(*args)
        adds Transition with included arguments to the memory
    sample(batch_size)
        returns batch size random selections for updating nn
    """

    def __init__(self, capacity: int):
        """
        Constructor
        :param capacity: length of memory
        """
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Method to allow a Transition to be added to the deque.
        :param args: arguments for adding to a Transition named tuple to be added to the list
                        should be state, action, reward, next_state, done where:
                        state: tensor, action: int, reward: float, next_state: tensor, done: bool
        :return: None
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """
        Method to return batch of Transitions randomly selected without replacement.
        :param batch_size: int - Length of batch
        :return: returns a Transition namedtuple with lists of states, actions, rewards,
                    next states, and dones
        """
        # Get list of Transitions to return
        s = random.sample(self.memory, batch_size)
        # Convert list to transitions to Transition of lists
        return Transition(*zip(*s))

    def __len__(self):
        return len(self.memory)


class Agent:
    """
    Deep Q-Learning Agent

    Attributes
    __________
    state_action_dims : list
        dimensions of state action space
    game : gym.Env
        the environment from which to learn
    policy_net : DQN
        NN represent policy
    target_net : DQN
        NN representing target values for Q table
    optimizer : torch.optim
        Optimizer to update NN
    memory : ReplayBuffer
        memory to hold transitions

    Methods
    --------
    get_action(state, epsilon)
        return an action for a state using either exploration or exploitation
    learn(transitions)
        updates policy_net based on random batch of transitions from memory
    train()
        the training algorithm
    test()
        runs a series of black jack hands to evaluate current performance of model
    get_strategy()
        creates and returns a strategy dataframe from policy_net.

    """

    def __init__(self, game: gym.Env,
                 state_action_dims: List[int] = [32, 11, 2, 2]):
        """
        Constructor.  Sets of policy and target nets, optimizer and memory
        :param game: environment to test
        :param state_action_dims: dimensions of state action space
        """

        self.state_action_dims = state_action_dims
        self.game = game
        self.policy_net = DQN(len(state_action_dims) - 1, state_action_dims[-1]).to(device)
        self.target_net = DQN(len(state_action_dims) - 1, state_action_dims[-1]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(BUFFER_SIZE)

    def get_action(self, state: torch.FloatTensor, epsilon: float = 0.0) -> int:
        """
        Method to get action with exploration or exploitation.
        :param state: describes the current state of the game
        :param epsilon: parameter to explore ore exploit
        :return: integer representing action to be used.
        """

        if random.random() < epsilon:
            return self.game.action_space.sample()
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def learn(self, transitions: Transition) -> float:
        """
        Method to update local policy.  S
        :param transitions: Transition of lists of sarsd values
        :return: loss after update
        """
        # Break up transitions into torch tensor of appropriate shapes
        state_batch = torch.stack(transitions.state)                    # batch size x number obs
        next_state_batch = torch.stack(transitions.next_state)          # batch size x number obs
        action_batch = torch.tensor(transitions.action).unsqueeze(1)    # batch size x 1
        reward_batch = torch.tensor(transitions.reward).unsqueeze(1)    # batch size x 1
        done_batch = torch.tensor(transitions.done, dtype=int).unsqueeze(1)  # batch size x 1

        # Use target net to determine expected values of NN
        with torch.no_grad():
            # y = r + df * max Q'(s',a')
            # max Q'(s',a') mus be 0 for terminal states
            target_max_qs = self.target_net(next_state_batch).max(1, keepdims=True).values  # batch_size x 1
            expected_state_action_values = (reward_batch +
                                            DISCOUNT_FACTOR * target_max_qs * (1 - done_batch))  # batch_size x 1
        # predict Q(s,a) for entire batch
        state_action_values = torch.gather(self.policy_net(state_batch), 1, action_batch)  # batch_size X 1

        # Calculate loss and update policy net
        loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values.squeeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        """
        The DRL algorithm
        :return: None
        """

        losses = []
        for episode in range(1, MAX_EPISODES + 1):

            # Run one episode
            # States are converted to torch tensors between 0 and 1
            state = env.reset()
            state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
            done = False
            while not done:
                epsilon = max(EPSILON_FINAL, EPSILON_START - episode / EPSILON_DECAY_LAST_FRAME)
                action = self.get_action(state, epsilon)
                next_state, reward, done, *_ = self.game.step(action)
                next_state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                self.memory.push(state, action, reward, next_state, done)
                state = next_state

            # Learn after EVERY_UPDATE episodes
            if episode % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                loss = self.learn(experiences)
                losses.append(loss)

            # Update target network
            if episode % SYNC_TARGET_EPISODES == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Report status every 1000 episodes
            if episode % 250 == 0:
                results = self.test()
                print(
                    f'Episode {episode}  Avg loss: {sum(losses) / len(losses)}  '
                    f'Test Reward: {results[0]*100:0.2f}% +- {results[1]*100:0.2f}%'
                                   )
                torch.save(agent.policy_net.state_dict(), 'rose_dqn_checkpoint.pth')
                losses = []

    def test(self):
        """
        Method to assess performance of model  To get an idea of the variability
        of the results, multiple runs of games are played and the mean return
        of the runs along with the st dev of the results is returned.
        :return: tuple with mean and st dev of performance
        """

        TEST_SETS = 10
        EPISODES_PER_SET = 1000
        set_returns = []
        for set_ in range(TEST_SETS):
            reward = 0
            set_return = 0
            for episode in range(EPISODES_PER_SET):
                state = env.reset()
                done = False
                while not done:
                    state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                    action = self.get_action(state)
                    next_state, reward, done, *_ = env.step(action)
                    state = next_state
                set_return += reward
            set_returns.append(set_return/EPISODES_PER_SET)
        return stats.mean(set_returns), stats.stdev(set_returns)

    def get_strategy(self):
        """
        Method to convert policy net into strategy table
        :return:
        """
        action_map = {1: 'H', 0: 'S'}
        strategy = strategies.BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
        self.policy_net.eval()
        for row in strategy.index:
            for col in strategy.columns:
                state = gym_utils.parse_strategy_state_to_gym_state(row, col)
                state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                with torch.no_grad():
                    action = self.policy_net(state).argmax().item()
                strategy.loc[row, col] = action_map[action]
        self.policy_net.train()
        return strategy


if __name__ == '__main__':
    env = gym.make("Blackjack-v1")
    agent = Agent(env, [32, 11, 2, 2])
    agent.train()
    torch.save(agent.policy_net.state_dict(), 'TrainedModels/rose_dqn__gym_model.pth')
    strategy = agent.get_strategy()
    strategy.to_pickle('TrainedStrategies/rose_dqn_gym_strategy')