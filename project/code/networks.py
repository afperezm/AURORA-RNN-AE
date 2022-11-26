import torch
from torch import nn as nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=64):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        p = self.fc3(x)
        return p


class WalkerNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, neurons_list=None, normalise=False, affine=False):
        super(WalkerNet, self).__init__()
        if neurons_list is None:
            neurons_list = [128, 128]
        self.weight_init_fn = nn.init.xavier_uniform_
        self.num_layers = len(neurons_list)
        self.normalise = normalise
        self.affine = affine

        if self.num_layers == 1:

            self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
            self.l2 = nn.Linear(neurons_list[0], action_dim, bias=(not self.affine))

            if self.normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
                self.n2 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

        if self.num_layers == 2:
            self.l1 = nn.Linear(state_dim, neurons_list[0], bias=(not self.affine))
            self.l2 = nn.Linear(neurons_list[0], neurons_list[1], bias=(not self.affine))
            self.l3 = nn.Linear(neurons_list[1], action_dim, bias=(not self.affine))

            if self.normalise:
                self.n1 = nn.LayerNorm(neurons_list[0], elementwise_affine=self.affine)
                self.n2 = nn.LayerNorm(neurons_list[1], elementwise_affine=self.affine)
                self.n3 = nn.LayerNorm(action_dim, elementwise_affine=self.affine)

        self.max_action = max_action

        for param in self.parameters():
            param.requires_grad = False

        # self.apply(self.init_weights)

    def forward(self, state):
        if self.num_layers == 1:
            if self.normalise:
                a = F.relu(self.n1(self.l1(state)))
                return self.max_action * torch.tanh(self.n2(self.l2(a)))
            else:
                a = F.relu(self.l1(state))
                return self.max_action * torch.tanh(self.l2(a))
        if self.num_layers == 2:
            if self.normalise:
                a = F.relu(self.n1(self.l1(state)))
                a = F.relu(self.n2(self.l2(a)))
                return self.max_action * torch.tanh(self.n3(self.l3(a)))
            else:
                a = F.relu(self.l1(state))
                a = F.relu(self.l2(a))
                return self.max_action * torch.tanh(self.l3(a))
