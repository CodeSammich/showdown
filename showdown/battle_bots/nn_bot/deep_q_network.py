import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=17):
        super(DeepQNetwork, self).__init__() # calls nn.Module __init__

        self.seed = torch.manual_seed(seed)

        # start by copying http://cs230.stanford.edu/projects_fall_2018/reports/12447633.pdf network
        self.fc1 = nn.Linear(state_size, 128)
        self.dropout = nn.Dropout(p=.3)
        self.fc2 = nn.Linear(128, 64)
        self.logits = nn.Linear(64, action_size)
        self.softmax = nn.Softmax()

    def forward(self, state):
        # one vector of the entire state
        # state vector contains: weather, field, etc..., opponent's active, my active, my remaining
        # drawback: it forgets previously released opponent's pokemon
        state = self.fc1(state)
        state = F.relu(state)
        state = self.dropout(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.logits(state)
        state = F.softmax(state)
        return state

    #TODO: pre-training function before the RL Agent class uses it for state generation
