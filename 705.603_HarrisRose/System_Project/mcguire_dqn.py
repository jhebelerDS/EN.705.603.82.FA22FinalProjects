"""
This module contains DRL code from the article used in
Assignment 10

A train function is provided at the bottom to allow training from an
outside script.  No documentation was added to code taken from:

https://github.com/trevormcguire/blogposts/blob/main/RL/RL%20Blackjack.ipynb
"""

import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 3e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


def get_state_idxs(state):
    idx1, idx2, idx3 = state
    idx3 = int(idx3)
    return idx1, idx2, idx3


class QNetwork(nn.Module):
    """
    -------
    Neural Network Used for Agent to Approximate Q-Values
    -------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    """

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    """
    --------
    Deep Q-Learning Agent
    --------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    --------
    """

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        --------------
        Take an action given the current state (S(i))
        --------------
        [Params]
            'state' -> current state
            'eps' -> current epsilon value
        --------------
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).cpu().data.numpy()
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values), np.max(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module, tau: float):
        """
        --------
        Update our target network with the weights from the local network
        --------
        Formula for each param (w): w_target = τ*w_local + (1 - τ)*w_target
        See https://arxiv.org/pdf/1509.02971.pdf
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


from dataclasses import dataclass


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    ------------
    Used to store agent experiences
    ------------
    [Params]
        'action_size' -> length of the action space
        'buffer_size' -> Max size of our memory buffer
        'batch_size' -> how many memories to randomly sample
        'seed' -> seed for random module
    ------------
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


from collections import deque


def dqn(n_episodes=2000, eps_start=0.99, eps_end=0.02, eps_decay=0.995):
    """
    -------------
    Train a Deep Q-Learning Agent
    -------------
    [Params]
        'n_episodes' -> number of episodes to train for
        'eps_start' -> epsilon starting value
        'eps_end' -> epsilon minimum value
        'eps_decay' -> how much to decrease epsilon every iteration
    -------------
    """

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for episode in range(1, n_episodes + 1):
        done = False
        episode_score = 0

        state = env.reset()
        state = np.array(get_state_idxs(state), dtype=float)
        state[0] = state[0] / 32
        state[1] = state[1] / 10

        while not done:
            action = agent.act(state, eps)
            if isinstance(action, tuple):
                action, value = action
            else:
                value = 1.
            next_state, reward, done, _ = env.step(action)
            reward *= value
            next_state = np.array(get_state_idxs(next_state), dtype=float)
            next_state[0] = next_state[0] / 32
            next_state[1] = next_state[1] / 10

            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_score += reward

        scores_window.append(episode_score)
        scores.append(episode_score)

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 5000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'mcguire_checkpoint.pth')

    return scores


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("Blackjack-v1")
env.action_space.seed(42)
agent = Agent(state_size=3, action_size=2, seed=0)


def train(filename):
    """
    Function to allow training from external script.
    :param filename: str with location to save model parameters.
    :return:
    """

    scores = dqn(n_episodes=70_000)
    torch.save(agent.qnetwork_local.state_dict(), filename)