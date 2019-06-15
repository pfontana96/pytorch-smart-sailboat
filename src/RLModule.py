# -*- coding: utf-8 -*-

import gym
import gym_voilier
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from numpy import pi, arange

import matplotlib.pyplot as plt
import argparse

# Arguments parser
parser = argparse.ArgumentParser(prog = "pytorch-smart-sailboat")
parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-f', '--file', type=str, default=None, help='Model Parameters File')
parser.add_argument('-e', '--episodes', type=int, default=1000, help='Number of episodes')
parser.add_argument('-g', '--graphics', action='store_true', help='Enable Graphics')
args = parser.parse_args()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x.float()) #Il y avait un erreur: Expected Float but got Double instead
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def save_model(model, i):
    path = "../data/model_{:d}"
    # path = "../data/model_2000"
    print('Saving model parameters "', path.format(i), '"..')
    torch.save(model.state_dict(), path.format(i))

def load_model(model, path):
    print("Loading model parameters from ", path, "..")
    model.load_state_dict(torch.load(path))
    model.eval()

if __name__ == '__main__':
    model_file = None
    
    if args.file:
        model_file = args.file
    
    
    env = gym.make('voilier-v2').unwrapped
    ######################################################################
    # Replay Memory
    # -------------
    #
    # We'll be using experience replay memory for training our DQN. It stores
    # the transitions that the agent observes, allowing us to reuse this data
    # later. By sampling from it randomly, the transitions that build up a
    # batch are decorrelated. It has been shown that this greatly stabilizes
    # and improves the DQN training procedure.
    #
    # For this, we're going to need two classses:
    #
    # -  ``Transition`` - a named tuple representing a single transition in
    #    our environment. It maps essentially maps (state, action) pairs
    #    to their (next_state, reward) result, with the state being the
    #    screen difference image as described later on.
    # -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
    #    transitions observed recently. It also implements a ``.sample()``
    #    method for selecting a random batch of transitions for training.
    #

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    
    env.reset()

    ######################################################################
    # Training
    # --------
    #
    # Hyperparameters and utilities
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # This cell instantiates our model and its optimizer, and defines some
    # utilities:
    #
    # -  ``select_action`` - will select an action accordingly to an epsilon
    #    greedy policy. Simply put, we'll sometimes use our model for choosing
    #    the action, and sometimes we'll just sample one uniformly. The
    #    probability of choosing a random action will start at ``EPS_START``
    #    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
    #    controls the rate of the decay.
    # -  ``plot_durations`` - a helper for plotting the durations of episodes,
    #    along with an average over the last 100 episodes (the measure used in
    #    the official evaluations). The plot will be underneath the cell
    #    containing the main training loop, and will update after every
    #    episode.
    #

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    # Get number of actions from gym action space
    input_size = env.observation_space.shape[0]  # Relative position to target and wind conditions
    hidden_size = 8    #Arbitrary
    n_actions = env.action_space.n

    policy_net =  Net(input_size, n_actions, hidden_size).to(device)
    if model_file is not None:
        load_model(policy_net, model_file)
        EPS_START = 0.0
        EPS_END = 0.0
    target_net = Net(input_size, n_actions, hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    steps_done = 0

    possible_actions = [-pi/2, -pi/3, -pi/4, -pi/5, 0, pi/5, pi/4, pi/3, pi/2]

    rewards = []

    t_max = 100
    dt = 0.2
    num_episodes = args.episodes
    global_r = -2.0

    ######################################################################
    #
    # Below, you can find the main training loop. At the beginning we reset
    # the environment and initialize the ``state`` Tensor. Then, we sample
    # an action, execute it, observe the next screen and the reward (always
    # 1), and optimize our model once. When the episode ends (our model
    # fails), we restart the loop.
    #
    #

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset() # to_target[2], norm(to_target), theta derive, theta voile, wind[2]
        state = torch.from_numpy(np.array([state], dtype = np.float32)).to(device)

        ep_r = 0.
        for t in arange(0,t_max,dt):
            if args.graphics: env.render()


            # Select and perform an action
            # action = select_action(state)
            # u = np.array([possible_actions[theta_voile], possible_actions[theta_derive]]) / (pi/2)
            action = select_action(state)
            action_v = np.zeros(n_actions).reshape(1,n_actions)
            action_v[0,action] = 1

            next_state, reward, done, _ = env.step(action_v)
            next_state = torch.from_numpy(np.array([next_state], dtype = np.float32)).to(device)

            ep_r += reward
            reward = torch.tensor([reward], device=device, dtype = torch.float)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                break
        rewards.append(ep_r)
        if global_r == 0.:
            global_r = -2.
        else:
            global_r = global_r*0.99 + ep_r*0.01
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        template = "Episode: {:05d}/{:d} | Ep Reward: {:.3f} | Global Reward: {:.3f}"
        print(template.format(i_episode+1, num_episodes, ep_r, global_r))

        # Each 1000 episodes I save the project
        if i_episode%1000==0:
            save_model(policy_net, i_episode)

    print('Complete')
    if args.graphics: env.render()
    env.close()
    save_model(policy_net, num_episodes)
    plt.plot(rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Episodes')
    template = '../data/reward_{:d}'
    plt.savefig(template.format(num_episodes))

