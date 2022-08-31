import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import os, sys
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

writer = SummaryWriter("runs/tensor_visualization_test_1")

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from rcognita.objectives import RunningObjective

running_objective = RunningObjective(running_obj_model="quadratic")


class ModelNN(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_hidden=5, weights=None):
        super().__init__()

        self.model_name = "NN"
        self.fc1 = nn.Linear(dim_observation + dim_action, dim_hidden, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_observation, bias=False)

        if weights is not None:
            self.load_state_dict(weights)

        self.double()

    def forward(self, x, weights=None):
        # print("x type", type(x))

        x = x.double()

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.linalg.vector_norm(x)

        return x  # .detach().numpy()

    def __call__(self, observation, action, weights=None):

        concat = np.concatenate((observation, action))
        to_torch = torch.tensor(concat)

        if not weights is None:
            state_dict = self.weights2dict(weights)
            self.load_state_dict(state_dict)

        return self.forward(to_torch)


model = ModelNN(3, 2)

input_tensor_1 = torch.randn(1, 5)
input_tensor_2 = torch.randn(1, 5)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

print([x for x in model.parameters()])

loss = model.forward(input_tensor_1) - model.forward(input_tensor_2)


loss.backward()

for _ in range(100):
    optimizer.step()

print([x for x in model.parameters()])
print(model.parameters)

