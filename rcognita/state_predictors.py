import numpy as np
from abc import ABCMeta, abstractmethod
from .npcasadi_api import SymbolicHandler


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

    def predict_state(self, current_state, action, is_symbolic=False):
        next_state = current_state + self.pred_step_size * self.state_dyn(
            [], current_state, action, is_symbolic=is_symbolic
        )
        return next_state

    def predict_state_sqn(self, observation, my_action_sqn, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)

        observation_sqn = npcsd.zeros(
            [self.Nsteps + 1, self.dim_output], array_type="SX"
        )
        observation_sqn[0, :] = observation
        current_observation = observation

        for k in range(1, self.Nsteps + 1):
            current_action = my_action_sqn[k - 1, :]
            next_observation = self.predict_state(
                current_observation, current_action, is_symbolic
            )
            observation_sqn[k, :] = self.sys_out(
                next_observation, is_symbolic=is_symbolic
            )
            current_observation = next_observation
        return observation_sqn
