#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes to be used in fitting system models.

Updates to come.

"""
from audioop import mul
import numpy as np
import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from utilities import uptria2vec, nc
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import math


class ModelSS:
    """
    State-space model
            
    .. math::
        \\begin{array}{ll}
			\\hat x^+ & = A \\hat x + B u, \\newline
			y^+  & = C \\hat x + D u.
        \\end{array}                 
        
    Attributes
    ---------- 
    A, B, C, D : : arrays of proper shape
        State-space model parameters.
    x0set : : array
        Initial state estimate.
            
    """

    def __init__(self, A, B, C, D, x0est):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x0est = x0est
        self.model_name = "state-space"

    def upd_pars(self, Anew, Bnew, Cnew, Dnew):
        self.A = Anew
        self.B = Bnew
        self.C = Cnew
        self.D = Dnew

    def updateIC(self, x0setNew):
        self.x0set = x0setNew


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
        x = self.relu1(x)
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


class ModelNN(nn.Module):
    def __init__(self, dim_observation, dim_action, dim_hidden=5, weights=None):
        super().__init__()

        self.model_name = "NN"
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
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.linalg.vector_norm(x)

        return x.detach().numpy()

    def __call__(self, observation, action, weights=None):

        concat = np.concatenate((observation, action))
        to_torch = torch.tensor(concat)

        if not weights is None:
            state_dict = self.weights2dict(weights)
            self.load_state_dict(state_dict)

        return self.forward(to_torch)

    def get_weights(self):

        weights_all = np.array([])

        for param_tensor in self.state_dict():
            weights = self.state_dict()[param_tensor].detach().numpy().reshape(-1)
            weights_all = np.hstack((weights_all, weights))

        return weights_all

    def weights2dict(self, weights_to_parse):

        weights_to_parse = torch.tensor(weights_to_parse)

        new_state_dict = {}

        len_prev = 0

        for param_tensor in self.state_dict():
            weights_size = self.state_dict()[param_tensor].size()
            weights_lenght = math.prod(self.state_dict()[param_tensor].size())
            new_state_dict[param_tensor] = torch.reshape(
                weights_to_parse[len_prev : len_prev + weights_lenght],
                tuple(weights_size),
            )
            len_prev = weights_lenght

        return new_state_dict


class ModelPolynomial:
    def __init__(self, model_name="quad-lin"):
        self.model_name = model_name

    def __call__(self, v1, v2, weights):
        return self.compute(v1, v2, weights)

    def compute(self, v1, v2, weights):

        v1 = nc.to_col(v1)
        v2 = nc.to_col(v2)
        chi = nc.concatenate([v1, v2])
        if self.model_name == "quad-lin":
            polynom = nc.to_col(uptria2vec(nc.outer(chi, chi)))
            polynom = nc.concatenate([polynom, chi]) ** 2
        elif self.model_name == "quadratic":
            polynom = nc.to_col(uptria2vec(nc.outer(chi, chi))) ** 2
        elif self.model_name == "quad-nomix":
            polynom = chi * chi
        elif self.model_name == "quad-mix":
            polynom = nc.concatenate([v1 ** 2, nc.kron(v1, v2), v2 ** 2])

        result = nc.dot(weights, polynom)

        return result

    def gradient(self, vector, weights):

        gradient = nc.autograd(self.compute, vector, [], weights)

        return gradient


class ModelQuadForm:
    def __init__(self, R1=None, R2=None, model_name="quadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.compute(v1, v2)

    def compute(self, v1, v2):
        result = nc.matmul(nc.matmul(v1.T, self.R1), v2)

        result = nc.squeeze(result)

        return result


class ModelBiquadForm:
    def __init__(self, R1=None, R2=None, model_name="biquadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.compute(v1, v2)

    def compute(self, v1, v2):
        result = nc.matmul(nc.matmul(v1.T ** 2, self.R2), v2 ** 2) + nc.matmul(
            nc.matmul(v1.T, self.R1), v2
        )
        result = nc.squeeze(result)

        return result
