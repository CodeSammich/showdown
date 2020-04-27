import numpy as np
import random
from collections import namedtuple, deque

##Importing the model (function approximator for Q-table)
from showdown.battle_bots.nn_bot.deep_q_network import DeepQNetwork as QNetwork

from data import all_move_json as moves_lib
import asyncio

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 16  # minibatch size
GAMMA = 0.80  # discount factor
TAU = 1e-2  # for soft update of target parameters
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cpu")

# mutex locking for multiprocessing
_lock_table = {'locked': False}


class Agent():
    """Interacts with and learns form environment."""

    def __init__(self, state_size, action_size, seed, config):
        """Initialize an Agent object.

        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(self.seed)
        self.config = config

        # Store previous info
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = 0
        self.lossList = []

        # Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # New parameter. Determines if agent is in train or eval mode
        self._train = False
    def set_previous(self, state, action, reward):
        self.previous_state = state
        self.previous_action = action
        self.previous_reward = reward

    async def step(self, state, action, reward, next_step, done):
        if not self._train:
            return
            # print("Will not update network because it is in eval mode")

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                await self.learn(experience, GAMMA)

    def act(self, state, my_options, all_switches, eps=0.2):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Output:
            Move string chosen, see nn_bot/main.py
        """
        if not self._train:
            eps = 0

        state = state.float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            # logits output
            logits = self.qnetwork_local(state)
        self.qnetwork_local.train()

        def pick_move_based_on_logits(logits, my_options, all_switches, pick_random=False):
            '''
            How we choose the mask is dependant on the new 
            moves categories (whatever flags are ticked and correspond w/ moveset
            
            logits: switches 0-5, isStatus, 18 attack move types (physical or special)
            
            Special and physical are equally important since pokemon are specialized
            Logits layer determines what kind of move we want
            If move is not available, picks move with highest basePower

            Inputs:
                logits: output of the neural network, a vector of preferences for move
                my_options: all current possible actions
                all_switches: all potential switches, including self and fainted

            Output:
                Best valid move string preferred by neural network
            '''
            #   SWITCHES
            #       encode index (switches 1-6) to each switch in my_options
            #       have a dict for valid switches (index -> my option string)
            #       if my option string is not in my_options, ask network again 
            #
            #   MOVES
            #       if network wants something, look up in dict O(1) and see if it matches
            #       if not, ask network again
            #
            #   In either "ask network again" cases, set the max value of action_values to 0
            #   Note: could also use mask, but I prefer this due to not having to keep track of a third component
            #
            types = { # attack move types, in this order for logits:
                7: 'normal',
                8: 'fire',
                9: 'fighting',
                10: 'water',
                11: 'flying',
                12: 'grass',
                13: 'poison',
                14: 'electric',
                15: 'ground',
                16: 'psychic',
                17: 'rock',
                18: 'ice',
                19: 'bug',
                20: 'dragon',
                21: 'ghost',
                22: 'dark',
                23: 'steel',
                24: 'fairy'}

            logits = logits.cpu().data.numpy()[0] # convert to numpy for argmax
            while True: # run until move is found
                if pick_random:
                    index = np.random.randint(len(logits)) # index of random move 
                else:
                    index = np.argmax(logits) # index of NN chosen move
                logits[index] = 0 # remove that choice b/c we repeat if what NN is asking for is invalid

                if index < 6: # switch to selected pokemon
                    switch = all_switches[index]
                    if switch in my_options:
                        return index, switch
                else:
                    for move in my_options: # check all moves for status and type
                        if move in moves_lib: # if valid pokemon attack, return if it's what NN is asking for
                            if index == 6: # NN wants a status move
                                if moves_lib[move]['category'] == 'status':
                                    return index, move
                            elif types[index] == moves_lib[move]['type']: # NN wants a type move
                                return index, move

        if random.random() > eps: # epsilon-greedy selection
            return pick_move_based_on_logits(logits, my_options, all_switches)
        else:
            # pick a random index and return both it and the element
            return pick_move_based_on_logits(logits, my_options, all_switches, pick_random=True)

    async def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # LOCK
        while _lock_table['locked']:
            await asyncio.sleep(1)

        _lock_table['locked'] = True
        states, actions, rewards, next_state, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.lossList.append(loss.detach().numpy())  # convert to numpy
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        # UNLOCK
        _lock_table['locked'] = False

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)


    def train(self):
        self._train = True
        self.qnetwork_local.train()

    def eval(self):
        self._train = False
        self.qnetwork_target.eval()

class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                                 "action",
                                                                 "reward",
                                                                 "next_state",
                                                                 "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
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
        """Return the current size of internal memory."""
        return len(self.memory)
