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

from utilities import rc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod
from copy import deepcopy


class ModelAbstract(ABC):
    @property
    @abstractmethod
    def model_name(self):
        return "model_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    def cache_cur_state(self):
        self.cache = None
        self.cache = deepcopy(self)

    def __call__(self, argin, weights=None, use_fixed_weights=False):
        if use_fixed_weights is False:
            if weights is not None:
                return self.forward(argin, weights)
            else:
                return self.forward(argin, self.weights)
        else:
            if weights is not None:
                return self.cache.forward(argin, weights)
            else:
                return self.cache.forward(argin, self.weights)

    def update(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    @property
    def weights(self):
        return self.get_weights()

    @weights.setter
    def weights(self, weights):
        self.cache_cur_state()
        self.update(weights)


class ModelSS:
    model_name = "state-space"
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


class ModelQuadLin(ModelAbstract):
    model_name = "quad-lin"

    def __init__(self, input_dim, Wmin=1.0, Wmax=1e3):
        self.dim_weights = int((input_dim + 1) * input_dim / 2 + input_dim)
        self.Wmin = Wmin * np.ones(self.dim_weights)
        self.Wmax = Wmax * np.ones(self.dim_weights)
        self.weights = self.Wmin

    def forward(self, vec, weights):

        polynom = rc.uptria2vec(rc.outer(vec, vec))
        polynom = rc.concatenate([polynom, vec]) ** 2
        result = rc.dot(weights, polynom)

        return result


class ModelQuadratic(ModelAbstract):
    model_name = "quadratic"

    def __init__(self, input_dim, Wmin=1.0, Wmax=1e3):
        self.dim_weights = int((input_dim + 1) * input_dim / 2)
        self.Wmin = Wmin * np.ones(self.dim_weights)
        self.Wmax = Wmax * np.ones(self.dim_weights)
        self.weights = self.Wmin

    def forward(self, vec, weights):

        polynom = rc.to_col(rc.uptria2vec(rc.outer(vec, vec))) ** 2
        result = rc.dot(weights, polynom)

        return result


class ModelQuadNoMix(ModelAbstract):
    model_name = "quad-nomix"

    def __init__(self, input_dim, Wmin=1.0, Wmax=1e3):
        self.dim_weights = input_dim
        self.Wmin = Wmin * np.ones(self.dim_weights)
        self.Wmax = Wmax * np.ones(self.dim_weights)
        self.weights = self.Wmin

    def forward(self, vec, weights):

        polynom = vec * vec
        result = rc.dot(weights, polynom)

        return result


# class ModelQuadMix(ModelBase):
#     model_name = "quad-mix"

#     def __init__(self, input_dim, Wmin=1.0, Wmax=1e3):
#         self.dim_weights = int(
#             self.dim_output + self.dim_output * self.dim_input + self.dim_input
#         )
#         self.Wmin = Wmin * np.ones(self.dim_weights)
#         self.Wmax = Wmax * np.ones(self.dim_weights)

#     def _forward(self, vec, weights):

#         v1 = rc.to_col(v1)
#         v2 = rc.to_col(v2)

#         polynom = rc.concatenate([v1 ** 2, rc.kron(v1, v2), v2 ** 2])
#         result = rc.dot(weights, polynom)

#         return result


class ModelQuadForm:
    model_name = "quad_form"

    def __init__(self, R1=None, R2=None, model_name="quadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.forward(v1, v2)

    def forward(self, v1, v2):
        result = rc.matmul(rc.matmul(v1.T, self.R1), v2)

        result = rc.squeeze(result)

        return result


class ModelBiquadForm:
    model_name = "biquad_form"

    def __init__(self, R1=None, R2=None, model_name="biquadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.forward(v1, v2)

    def forward(self, v1, v2):
        result = rc.matmul(rc.matmul(v1.T ** 2, self.R2), v2 ** 2) + rc.matmul(
            rc.matmul(v1.T, self.R1), v2
        )
        result = rc.squeeze(result)

        return result


class ModelNN(nn.Module, ModelAbstract):
    model_name = "NN"

    def __init__(self, dim_observation, dim_action, dim_hidden=5, weights=None):
        super().__init__()

        self.fc1 = nn.Linear(dim_observation + dim_action, dim_hidden, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(dim_hidden, dim_observation, bias=False)

        if weights is not None:
            self.load_state_dict(weights)

        self.double()
        self.cache_cur_state()

    def forward(self, input_tensor):
        # print("x type", type(x))

        # input_tensor = input_tensor.double()

        x = self.fc1(input_tensor)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.linalg.vector_norm(
            x
        )  # + 1e-3 * torch.linalg.vector_norm(input_tensor)

        return x

    # def __call__(self, model_input):

    #     model_input = torch.tensor(model_input)

    #     concated = torch.cat((observation_tensor, action_tensor))

    #     return self.forward(model_input)

    def detach_cur_model(self):
        for variable in self.parameters():
            variable.detach_()

    def get_weights(self):

        weights_all = np.array([])

        for param_tensor in self.state_dict():
            weights = self.state_dict()[param_tensor].detach().numpy().reshape(-1)
            weights_all = np.hstack((weights_all, weights))

        return weights_all

    def cache_cur_state(self):
        # self.cache = None
        # self.cache = deepcopy(self)
        super().cache_cur_state()
        self.cache.detach_cur_model()

    def update(self, weights):
        weights_dict = self.weights2dict(weights)
        self.load_state_dict(weights_dict)

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

    def soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (Torch model): weights will be copied from
            target_model (Torch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            self.cache.parameters(), self.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def __call__(self, argin, weights=None, use_fixed_weights=False):
        argin = torch.tensor(argin)
        if use_fixed_weights is False:
            if weights is not None:
                result = self.forward(argin, weights)
            else:
                result = self.forward(argin)
        else:
            if weights is not None:
                result = self.cache.forward(argin, weights)
            else:
                result = self.cache.forward(argin)

        if use_fixed_weights:
            return result.detach().numpy()
        else:
            return result


class LookupTable2D(ModelAbstract):
    model_name = "tabular"

    def __init__(self, dims=None):
        self.table = rc.zeros(dims)
        self.indices = tuple([(i, j) for i in range(dims[0]) for j in range(dims[1])])

    def __call__(self, argin, use_fixed_weights=False):

        if use_fixed_weights is False:
            result = self.forward(argin)
        else:
            result = self.cache.forward(argin)
        return result

    def forward(self, argin):
        self.table[argin]

    def update(self, table):
        self.table = table

