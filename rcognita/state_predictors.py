import numpy as np
from abc import ABCMeta, abstractmethod

from .utilities import nc


class BaseStatePredictor(metaclass=ABCMeta):
    @abstractmethod
    def predict_state(self):
        pass

    @abstractmethod
    def predict_state_sqn(self):
        pass


class EulerStatePredictor(BaseStatePredictor):
    def __init__(self, pred_step_size, state_dyn, sys_out, dim_output, Nsteps):
        self.pred_step_size = pred_step_size
        self.state_dyn = state_dyn
        self.sys_out = sys_out
        self.dim_output = dim_output
        self.Nsteps = Nsteps

    def predict_state(self, current_state, action):
        next_state = current_state + self.pred_step_size * self.state_dyn(
            [], current_state, action
        )
        return next_state

    def predict_state_sqn(self, observation, my_action_sqn):

        observation_sqn = nc.zeros(
            [self.Nsteps, self.dim_output], prototype=my_action_sqn
        )
        current_observation = observation

        for k in range(self.Nsteps):
            current_action = my_action_sqn[k, :]
            next_observation = self.predict_state(current_observation, current_action)
            observation_sqn[k, :] = self.sys_out(next_observation)
            current_observation = next_observation
        return observation_sqn
