import numpy as np
from .utilities import push_vec

import torch
import torch.optim as optim
import torch.nn as nn

import rcognita.models as models


class Estimator_RNN:
    """
    Class of model estimators based on recurrent neural networks
    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden,
        buffer_size,
        model=None,
        Nbackprops=1,
        t0=0,
        sampling_time=0.1,
    ):
        self.buffer_size = buffer_size

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        if model is None:
            self.model = models.ModelRNN(
                None, self.dim_observation, self.dim_action, self.dim_hidden
            )

        else:
            self.model = model

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.observation_buffer = np.zeros((self.buffer_size, self.dim_observation))
        self.action_buffer = np.zeros((self.buffer_size, self.dim_action))

        self.Nbackprops = Nbackprops

    def receive_sys_IO(self, t, observation, action):
        # push observation, action to buffers -- see functionality in controllers.py, line 1463

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            self.ctrl_clock = t

            self.observation_buffer = push_vec(self.observation_buffer, observation)
            self.action_buffer = push_vec(self.action_buffer, action)

    def update_params(self):
        """
        Update neural network weights
        """

        # Torch backprop (Nbackprops times, say) on loss = model accuracy over buffers

        self.loss.backward()

    def output_loss(self):
        """
        Return current loss
        """

        self.loss = 0

        for i in range(self.buffer_size - 1):
            # y_pred = self.model.model_out(np.concatenate((self.observation_buffer[i, :], self.action_buffer[i, :])))
            y_pred = self.model.model_out(
                self.observation_buffer[i, :], self.action_buffer[i, :]
            )

            # loss += np.linalg.norm((y_pred.detach().numpy() - self.observation_buffer[i + 1, :]))
            self.loss += self.criterion(
                y_pred, torch.tensor(self.observation_buffer[i + 1, :])
            )

        return self.loss.detach().numpy()
