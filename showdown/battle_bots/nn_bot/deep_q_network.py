import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=17):
        super(DeepQNetwork, self).__init__() # calls nn.Module __init__

        self.seed = torch.manual_seed(seed)

        # start by copying http://cs230.stanford.edu/projects_fall_2018/reports/12447633.pdf network
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.logits = nn.Linear(512, action_size)
        self.softmax = nn.Softmax()

    def forward(self, state):
        # one vector of the entire state
        # state vector contains: weather, field, etc..., opponent's active, my active, my remaining
        # drawback: it forgets previously released opponent's pokemon
        state = self.fc1(state)
        state = F.relu(state)
        state = self.fc2(state)
        state = F.relu(state)
        state = self.fc3(state)
        state = F.relu(state)
        state = self.logits(state)
        state = F.softmax(state)
        return state

