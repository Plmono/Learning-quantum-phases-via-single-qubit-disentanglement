import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import gym
import gym_disen1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import warnings

import argparse



class DQN(nn.Module):
    def __init__(self, in_channels=36, n_actions=6):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 512)


        self.head = nn.Linear(512, n_actions)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        return self.head(x)
warnings.filterwarnings("ignore", category=UserWarning)

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))

Transition_with_time = namedtuple('Transion',
                                  ('state', 'action', 'next_state', 'reward', 'time'))




Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ret = random.sample(self.memory, batch_size)
        # A list of BATCH_SIZE trasition object, each of them: ('state', 'action', 'next_state', 'reward'))
        return ret

    def __len__(self):
        return len(self.memory)


class NaivePrioritizedMemory(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):

        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state)

        self.priorities[self.position] = max_prio

        # TODO Maybe another way
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # Stardardized formula
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


def Logistic(x):
    x = math.log(x + 1, 2)
    return math.exp(x) / (1 + math.exp(x))


Transition_with_time = namedtuple('Transion',
                                  ('state', 'action', 'next_state', 'reward', 'time'))


class REPERMemory(object):
    def __init__(self, capacity, balancing_param=0.05, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.balancing_param = balancing_param
        self.time = 0

    def push(self, state, action, reward, next_state, done):

        max_prio = self.priorities.max() if self.memory else 1.0
        time_logits = Logistic(self.time)

        time_logits *= np.mean(self.priorities)

        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_with_time(state, action, reward, next_state, self.time)

        self.priorities[self.position] = max_prio * (1 - self.balancing_param) + self.balancing_param * time_logits
        self.time += 1

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # Stardardized formula
        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        # print(self.priorities)
        # exit()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1, 1),eps_threshold
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long),eps_threshold


def optimize_model_random(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def optimize_model_REPER(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)

    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition_with_time(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    # print(state_batch.size())
    # exit()

    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # expected_state_action_values = (1 - done) * (next_state_values * GAMMA) + reward_batch

    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Compute los by our own
    weights = torch.tensor(weights).float().to('cuda')

    loss = (state_action_values - expected_state_action_values) ** 2 * weights

    prios = loss + 1e-5

    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()

    memory.update_priorities(indices, prios)
    optimizer.step()
    return loss

def optimize_model_PER(memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions, indices, weights = memory.sample(BATCH_SIZE)
    print(transitions)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    # print(state_batch.size())
    # exit()

    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # expected_state_action_values = (1 - done) * (next_state_values * GAMMA) + reward_batch

    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Compute los by our own
    weights = torch.tensor(weights).float().to('cuda')

    loss = (state_action_values - expected_state_action_values) ** 2 * weights

    prios = loss + 1e-5

    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()

    memory.update_priorities(indices, prios)
    optimizer.step()
    return loss

def get_state(obs):
    state = np.array(obs)
    # state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    # state.unsqueeze(0)
    return state.unsqueeze(0).float()


def train(env, n_episodes, memory, render, sample):

    best=0
    best_a=[]
    loss=0
    for episode in range(n_episodes):
        a=[]
        state = get_state(env.reset())
        total_reward = 0.0
        # print(state)
        for t in count():
            action,epsilon = select_action(state)
            a.append(action.item())
            if render:
                env.render()

            next_state, reward, done = env.step(action)

            total_reward += reward

            if not done:
                next_state=get_state(next_state)
            else:
                next_state = None
            r=reward
            reward = torch.tensor([reward], device=device)

            # REPER
            if sample == 'PER':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'), done)
            elif sample == 'random':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            elif sample == 'REPER':
                memory.push(state, action.to('cpu'), next_state, reward.to('cpu'), done)

            state = next_state

            if steps_done > INITIAL_MEMORY:

                if sample == 'REPER':
                    loss=optimize_model_REPER(memory)

                elif sample == 'random':
                    loss=optimize_model_random(memory)

                elif sample == 'PER':
                    loss=optimize_model_PER(memory)

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                if r>best:
                    best=r
                    best_a=a
                    # print(a)
                print('Episode: {}'.format(episode),
                      'Total reward: ',r,
                      'Explore P: {:.4f}'.format(epsilon),
                        'loss: ',loss,
                      'Best reward:',best)
                break

    env.close()
    print(best_a)
    return


def test(env, n_episodes, policy, render, sample_method):
    _time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    # env = gym.wrappers.Monitor(env, './videos-{0}-{1}/dqn_pong_video'.format(_time, sample_method))
    for episode in range(n_episodes):
        a=[]
        state = get_state(env.reset())
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1, 1)
            a.append(action.item())
            # if render:
            #     env.render()
            #     time.sleep(0.02)

            next_state, reward, done = env.step(action)

            total_reward += reward

            if not done:
                next_state=get_state(next_state)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    print(a)
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=str, default='REPER', help="PER/REPER/random")

    args = parser.parse_args()
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 150
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 50000 #800 15000 50000
    TARGET_UPDATE = 1000 #50 100 100
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 2000
    MEMORY_SIZE=100*INITIAL_MEMORY

    # create networks
    policy_net = DQN(in_channels=10,n_actions=6).to(device)
    target_net = DQN(in_channels=10,n_actions=6).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("dis-v1")
    # env = gym.make_env(env)

    # initialize replay memory
    if args.sample_method == 'PER':
        # 2nd param default
        memory = NaivePrioritizedMemory(MEMORY_SIZE)

    elif args.sample_method == 'random':
        memory = ReplayMemory(MEMORY_SIZE)

    elif args.sample_method == 'REPER':
        memory = REPERMemory(MEMORY_SIZE)

    # train model
    train(env,20000, memory, render=False, sample=args.sample_method)
    #1200 7000 20000
    #500 6000 20000 40000
    # torch.save(policy_net, "dqn_dis13_2_1lay" + args.sample_method)
    # policy_net = torch.load("dqn_dis13_2_1lay" + args.sample_method)
    # test(env, 1, policy_net, render=False, sample_method=args.sample_method)