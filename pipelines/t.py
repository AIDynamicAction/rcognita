import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


# class Exp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i.exp()
#         ctx.save_for_backward(result)
#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         (result,) = ctx.saved_tensors
#         return grad_output * result


# # Use it by calling the apply method:
# tensor_input = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
# output = Exp.apply(tensor_input)


class ModelRNN(nn.Module):
    """
    Class of recurrent neural network models
    .. math::
        \\begin{array}{ll}
			\\hat y^+ & = \\vaprhi(y, u)
        \\end{array}
    Attributes
    ----------
    weights: : array of proper shape
        Neural weights.
    observation_est_init : : array
        Initial estimate of observation.
    """

    def __init__(self, weights, dim_observation, dim_action, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation + dim_action, dim_hidden)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_observation)

        if weights is not None:
            self.load_state_dict(weights)

        self.double()

    def forward(self, x):
        # print("x type", type(x))

        x = x.double()

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def model_out(self, observation, action):
        """
        Output estimated observation
        """

        # return RNN(observation, action, self.weights)  # Implement RNN output via torch

        concat = np.concatenate((observation, action))
        to_torch = torch.tensor(concat)

        return self.forward(to_torch)

    def updateIC(self, observation_est_init_new):
        """
        Update initial condition
        """

        self.observation_est_init = observation_est_init_new


print(output)
