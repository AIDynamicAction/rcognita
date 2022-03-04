#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes to be used in fitting system models.

Updates to come.

"""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import copy


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

    def upd_pars(self, Anew, Bnew, Cnew, Dnew):
        self.A = Anew
        self.B = Bnew
        self.C = Cnew
        self.D = Dnew

    def updateIC(self, x0setNew):
        self.x0set = x0setNew


# class ModelNN:
#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError(f"Class {self.__class__} is not yet implemented.")
# if self.dt != dt -> error

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
        self.fc2 = nn.Linear(dim_hidden, dim_observation)
        self.relu = nn.LeakyReLU()

        if (weights is not None):
            self.load_state_dict(weights)

        self.double()

    def forward(self, x):
        #print("x type", type(x))

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
