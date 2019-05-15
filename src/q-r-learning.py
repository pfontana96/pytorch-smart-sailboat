import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
from itertools import count

from roblib import *  # available at https://www.ensta-bretagne.fr/jaulin/roblib.py
from numpy import concatenate, float32, array_equal, angle
from time import *

from sim import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hardcoded for now
def decodeAction(act):
    return act//9, act%9 # voile, derive

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



def get_reward(pos, target):
    #HARDCODED!!!!!!!
    x = [-100, 100] # Map boundaries
    y = [-60, 60]   # Map boundaries
    status = 'not over'
    reward = 0
    if norm(target-pos) < 3:
        status = 'win'
        reward = 1000
    elif (pos[0] > x[1] or pos[0] < x[0] or pos[1] < y[0] or pos[1]>y[1]):
        #Out of bondaries
        status = 'lose'
        reward = -1000
    else:
        status = 'not over'
        reward = -int(norm(target))
    return (status, reward)

# -  ``select_action`` - will select an action accordingly to an epsilon
    #    greedy policy. Simply put, we'll sometimes use our model for choosing
    #    the action, and sometimes we'll just sample one uniformly. The
    #    probability of choosing a random action will start at ``EPS_START``
    #    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
    #    controls the rate of the decay.

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
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

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

    return loss

def save_model(model, i):
    ltime = time.localtime(time.time())
    #path = "../data/{:04d}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}.h5"
    path = "../data/model_{:03d}"
    torch.save(model, path.format(i))

def clamp_target(target, dist=10.0):
    if norm(target) > dist:
        return target*dist/norm(target)
    return target

if __name__ == '__main__':

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Graphics (Not recommended for training)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    use_display = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #  -pi -3pi/4 -pi/2 -pi/4 0 pi/4 pi/2 3pi/4 pi
    #  [ ) [    ) [    ) [   ) (   ](   ](     ]( ] 
    #   1     2     3     4   5  6    7     8    9
    
    #possible_actions = [-pi, -3*pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3*pi/4, pi]
    possible_actions = [-pi/2, -pi/3, -pi/4, -pi/5, 0, pi/5, pi/4, pi/3, pi/2]


    #-----------------------------------------------------
    # NN-parameters
    #-----------------------------------------------------

    input_size = 4 # Relative position to target and wind conditions    
    output_size = 81       
    hidden_size = 40       # Au Choix
    BATCH_SIZE = 128       # Au Choix 
    GAMMA = 0.999

    
    #Exploration rate

    EPS_START = 0.9
    EPS_END = 0.05
    #EPS_START = 0.0
    #EPS_END = 0.0
    EPS_DECAY = 200
    TARGET_UPDATE = 10


    num_episodes = 1000
    steps_done = 0
    n_actions = output_size # Angle timon et angle voile

    policy_net = Net(input_size, output_size, n_actions).to(device)
    #policy_net = torch.load("../data/model_1000") #Loads model trained (1000 episodes)
    target_net = Net(input_size, output_size, n_actions).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # memory
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # episode sizes
    episode_durations = []

    #-----------------------------------------------------
    # Simmulation-Parameters
    #-----------------------------------------------------

    params = {
        'awind':    2,      # wind force
        'ψ':        -1.57,  # wind angle
        'p0':        0.1,
        'p1':        1,
        'p2':        6000,
        'p3':        1000,
        'p4':        2000,
        'p5':        1,
        'p6':        1,
        'p7':        2,
        'p8':        300,
        'p9':        10000
    }

    t_max = 100
    dt = 0.2

    #-----------------------------------------------------
    # Plot-Parameters
    #-----------------------------------------------------
    a = array([[-50],[-100]])   
    b = array([[50],[100]])
    figure_params = {
        'width' :   200,
        'height':   120
    }
    if use_display == True:
        ax=init_figure(-figure_params['width']/2,figure_params['width']/2,-figure_params['height']/2,figure_params['height']/2)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #               Training Loop
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               

    win_history = []
    win_rate = 0.0

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        #x = array([[10,-40,-3,1,0]]).T   #x=(x,y,θ,v,w)
        x = array([[0.0, 0.0, -3, 1, 0]]).T   #x=(x,y,θ,v,w)
        u = array([0, 1])
        #target = array([random.randrange(-figure_params['width']/2, figure_params['width']/2), random.randrange(-figure_params['height']/2, figure_params['height']/2)])
        #target = array([80, 40])
        targetRange = 5.0 + i_episode*0.05
        target = np.random.uniform(-targetRange,targetRange,2)
        to_target = array([[target[0]-x[0][0], target[1]-x[1][0]]], dtype = float32)
        to_target = clamp_target(to_target)
        wind = array([params['awind'], params['ψ']], dtype = float32)
        #state = torch.from_numpy(to_target).to(device)
        state = torch.from_numpy(concatenate((to_target, wind.reshape(1,2)), axis = None).reshape(1,4)).to(device)
         
        for t in arange(0,t_max,dt):
            # Select and perform an action
            action = select_action(state)
            theta_voile, theta_derive = decodeAction(action)
            u = array([[possible_actions[theta_voile], possible_actions[theta_derive]]])

            x, δs = step(x, u, dt, wind)
            
            #Graphics
            if use_display == True:
                clear(ax)
                plot(target[0], target[1], '*b')
                draw_sailboat(x,δs,u[0,0],params['ψ'],params['awind'])
                draw_arrow(x[0][0], x[1][0], angle(to_target), norm(to_target), 'blue')

            status, reward = get_reward(x[0:2,0], target)
            reward = torch.tensor([reward], device=device, dtype=torch.float)

            if status == 'not over':
                to_target = array([[target[0]-x[0][0], target[1]-x[1][0]]], dtype = float32)
                to_target = clamp_target(to_target)
                #next_state = torch.from_numpy(to_target).to(device)
                next_state = torch.from_numpy(concatenate((to_target, wind.reshape(1,2)), axis = None).reshape(1,4)).to(device)
                done = False
            else:
                next_state = None
                done = True

            # Store the transition in memory
            memory.push(state.double(), action, next_state, reward) 

            # Move to the next state
            state = next_state
            # Perform one step of the optimization (on the target network)
            loss = optimize_model()

            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in NN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if status == 'win':
            win_history.append(1)
        else:
            win_history.append(0)
        win_rate = sum(win_history)/len(win_history) if len(win_history)>0 else 0.0
        if loss is not None:
            template = "Episode: {:03d}/{:d} | Loss: {:.3f} | Win Count: {:03d} | Win Rate: {:.3f}%"
            print(template.format(i_episode, num_episodes-1, loss, sum(win_history), win_rate))
        else:
            template = "Episode: {:03d}/{:d} | Loss: N/A | Win Count: {:03d} | Win Rate: {:.3f}%"
            print(template.format(i_episode, num_episodes-1, sum(win_history), win_rate))

    #save_model(policy_net, num_episodes)
    print('Complete')



        