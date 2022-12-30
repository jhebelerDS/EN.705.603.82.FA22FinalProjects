from core import mod_utils as utils
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):

    def __init__(self, args, init=False):
        super(Actor, self).__init__()
        self.args = args
        l1 = 128; l2 = 128; l3 = l2

        # Construct Hidden Layer 1
        #print("state_dim " + str(args.state_dim))
        self.w_l1 = nn.Linear(args.state_dim, l1)
        if self.args.use_ln: self.lnorm1 = LayerNorm(l1)

        #Hidden Layer 2
        self.w_l2 = nn.Linear(l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)


        #Out
        #print(args.action_dim)
        #print(args.action_dim.n)
        if args.env_tag == 'gym-go':
            self.w_out = nn.Linear(l3, args.action_dim.n)
        else:
            self.w_out = nn.Linear(l3, args.action_dim)

        #Init
        if init:
            self.w_out.weight.data.mul_(0.1)
            self.w_out.bias.data.mul_(0.1)

        if args.is_cuda: self.cuda()

    def forward(self, input):

        #Hidden Layer 1
        #print("shape input: " + int(input.shape))
        #print("input: " + str(input))
        #print('input shape: ' + str(input.flatten.shape))
        out = self.w_l1(input)
        if self.args.use_ln: out = self.lnorm1(out)
        out = F.tanh(out)

        #Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.tanh(out)


        #Out
        out = F.tanh(self.w_out(out))
        return out
    
    def batch_forward(self, batch_in):
        actions = []
        #print("batch_forward")
        for state in batch_in:
            #print(state)
            action = self.forward(state)
            actions.append(action)
        return actions





class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args

        l1 = 200; l2 = 300; l3 = l2

        # Construct input interface (Hidden Layer 1)
        self.w_state_l1 = nn.Linear(args.state_dim, l1)
        if args.env_tag == 'gym-go':
            self.w_action_l1 = nn.Linear(args.action_dim.n, l1)
        else:
            self.w_action_l1 = nn.Linear(args.action_dim, l1)

        #Hidden Layer 2
        self.w_l2 = nn.Linear(2*l1, l2)
        if self.args.use_ln: self.lnorm2 = LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out.weight.data.mul_(0.1)
        self.w_out.bias.data.mul_(0.1)

        if args.is_cuda: self.cuda()

    def forward(self, input, action):
        
        #print(input)
        #print(action)

        #Hidden Layer 1 (Input Interface)
        out_state = F.elu(self.w_state_l1(input))
        out_action = F.elu(self.w_action_l1(action)) # STB: TODO May want to test if it makes a major difference to input the filtered action space (remove invalid actions) Default to NOT FILTERED in first tests.
        
        #print(out_state.size())
        #print(out_action.size())
        
        #out = torch.cat((out_state, out_action), 1)
        out = torch.cat((out_state, out_action), -1)
        #print(out.size())
        # Hidden Layer 2
        out = self.w_l2(out)
        if self.args.use_ln: out = self.lnorm2(out)
        out = F.elu(out)

        # Output interface
        out = self.w_out(out)

        return out
    def batch_forward(self, input_batch, action_batch):
        q_list = []

        #print(action_batch)
        #print(type(action_batch))
        #print(len(action_batch))
        #print(action_batch[0])
        #print(type(action_batch[0]))
        #print(len(action_batch[0]))

        for i in range(len(input_batch)):
            
            out = self.forward(input_batch[i], action_batch[i])
            q_list.append(out)

        return torch.cat(q_list)



class DDPG(object):
    def __init__(self, args):

        self.args = args
        #print("AHHHHHH", self.args.state_dim)

        self.actor = Actor(args, init=True)
        self.actor_target = Actor(args, init=True)
        self.actor_optim = Adam(self.actor.parameters(), lr=0.5e-4)

        self.critic = Critic(args)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        #print("next_state_batch", batch.next_state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        if self.args.use_done_mask: done_batch = torch.cat(batch.done)
        state_batch.volatile = False; next_state_batch.volatile = True; action_batch.volatile = False

        #Load everything to GPU if not already
        if self.args.is_memory_cuda and not self.args.is_cuda:
            self.actor.cuda(); self.actor_target.cuda(); self.critic_target.cuda(); self.critic.cuda()
            state_batch = state_batch.cuda(); next_state_batch = next_state_batch.cuda(); action_batch = action_batch.cuda(); reward_batch = reward_batch.cuda()
            if self.args.use_done_mask: done_batch = done_batch.cuda()




        #Critic Update
        #print("full_batch: ", next_state_batch)
        #print("len full batch: ", len(next_state_batch))

        #next_action_batch = self.actor_target.forward(next_state_batch)
        next_action_batch = self.actor_target.batch_forward(batch.next_state)
        #next_q = self.critic_target.forward(next_state_batch, next_action_batch)
        #print("critic next")
        next_q = self.critic_target.batch_forward(batch.next_state, next_action_batch)
        if self.args.use_done_mask: next_q = next_q * ( 1 - done_batch.float()) #Done mask
        target_q = reward_batch + (self.gamma * next_q)

        self.critic_optim.zero_grad()
        #current_q = self.critic.forward((state_batch), (action_batch))
        #print("\n critic curr")
        current_q = self.critic.batch_forward(batch.state, batch.action)
        dt = self.loss(current_q, target_q)
        dt.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), 10)
        self.critic_optim.step()

        #Actor Update
        self.actor_optim.zero_grad()
        policy_loss = -self.critic.forward((state_batch),self.actor.forward((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm(self.critic.parameters(), 10)
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        #Nets back to CPU if using memory_cuda
        if self.args.is_memory_cuda and not self.args.is_cuda: self.actor.cpu(); self.actor_target.cpu(); self.critic_target.cpu(); self.critic.cpu()


def fanin_init(size, fanin=None):
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def actfn_none(inp): return inp

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OUNoise:

    def __init__(self, action_dimension, env_tag, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        if env_tag == 'gym-go':
            self.action_dimension = action_dimension.n
        else:
            self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        #print(self.action_dimension)
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
