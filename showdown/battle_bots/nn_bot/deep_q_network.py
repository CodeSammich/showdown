import torch.nn as nn
import torch.nn,.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=17):
        super(DeepQNetwork, self).__init__() # calls nn.Module __init__

        self.seed = torch.manual_seed(seed)

        # start by copying http://cs230.stanford.edu/projects_fall_2018/reports/12447633.pdf network
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.logits = nn.Linear(512, 10)

        # Loss / Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters) # lr = 0.001, momentum = 0.9 default

    def forward(self, state):
        # one vector of the entire state
        # state vector contains: weather, field, etc..., opponent's active, my active, my remaining
        # drawback: it forgets previously released opponent's pokemon
        moves = self.fc1(moves)
        moves = F.relu(moves)
        moves = self.fc2(moves)
        moves = F.relu(moves)
        moves = self.fc3(moves)
        moves = F.relu(moves)
        return moves