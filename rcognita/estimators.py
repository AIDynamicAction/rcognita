#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from .utilities import upd_line
from .utilities import reset_line
from .utilities import upd_scatter
from .utilities import upd_text
from .utilities import to_col_vec
from .utilities import push_vec

import torch
import torch.optim as optim
import torch.nn as nn

import rcognita.models as models


class EstimatorRNN:
    """
    Class of model estimators based on recurrent neural networks
    """

    def __init__(self, dim_observation, dim_action, dim_hidden, buffer_size,
                 model=None, Nbackprops=1, t0=0, sampling_time=0.1):
        self.buffer_size = buffer_size

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        if (model is None):
            self.model = models.ModelRNN(None, self.dim_observation, self.dim_action, self.dim_hidden)

        else:
            self.model = model

        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

        self.observation_buffer = np.zeros((self.buffer_size, self.dim_observation), dtype=np.float64)
        self.action_buffer = np.zeros((self.buffer_size, self.dim_action), dtype=np.float64)

        self.Nbackprops = Nbackprops

    def receive_sys_IO(self, t, observation, action):
        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            self.ctrl_clock = t

            self.observation_buffer = push_vec(self.observation_buffer, observation)
            self.action_buffer = push_vec(self.action_buffer, action)

            return True

        return False

    def update_params(self):
        """
        Update neural network weights
        """

        #############################################
        # YOUR CODE BELOW
        #############################################

        self.loss.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        self.optimizer.step()

        #############################################
        # YOUR CODE ABOVE
        #############################################

    def get_last_pred(self):
        return self.last_pred

    def output_loss(self):
        """
        Return current loss
        """

        self.loss = 0
        self.optimizer.zero_grad()

        #############################################
        # YOUR CODE BELOW
        #############################################

        for i in range(self.buffer_size - 1):
            y_pred = self.model.model_out(self.observation_buffer[i, :],
                                          self.action_buffer[i, :])

            self.loss += self.criterion(y_pred, torch.tensor
            (self.observation_buffer[i + 1, :]))

            self.last_pred = y_pred

        print("observation_buffer")
        print(self.observation_buffer)

        #############################################
        # YOUR CODE ABOVE
        #############################################

        return self.loss.detach().numpy()