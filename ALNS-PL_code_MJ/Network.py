import torch.nn as nn
import torch.nn.functional as F
'''
choose the network by initialize different classes
'''
class reward_net_1(nn.Module):
    def __init__(self, state_dim=5, action_dim=1):
        super(reward_net_1, self).__init__()
        self.linear1 = nn.Linear(state_dim,action_dim)

    def forward(self, x):
        x = self.linear1(x)
        return x

class reward_net_2(nn.Module):
    def __init__(self, input_feature=5, output_feature=1):
        super(reward_net_2, self).__init__()

        self.estimated_reward = nn.Sequential(
            nn.Linear(input_feature, 64),
            nn.ReLU(),
            nn.Linear(64, output_feature)
        )
    def forward(self):
        raise NotImplementedError

class reward_net_3(nn.Module):
    def __init__(self, input_feature=5, output_feature=1):
        super(reward_net_3, self).__init__()

        self.estimated_reward = nn.Sequential(
            nn.Linear(input_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_feature)
        )

        def forward(self):
            raise NotImplementedError

class reward_net_4(nn.Module):
    def __init__(self, input_feature=5, output_feature=1):
        super(reward_net_4, self).__init__()

        self.estimated_reward = nn.Sequential(
            nn.Linear(input_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_feature)
        )

        def forward(self):
            raise NotImplementedError


class reward_net_5(nn.Module):
    def __init__(self, state_dim=5, action_dim=1):
        super(reward_net_5, self).__init__()
        self.linear1 = nn.Linear(state_dim,64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

