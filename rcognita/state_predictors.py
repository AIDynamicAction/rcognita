import numpy as np
from abc import ABCMeta, abstractmethod


class BaseStatePredictor(metaclass=ABCMeta):
    @abstractmethod
    def predict_state(self):
        pass

    @abstractmethod
    def predict_state_sqn(self):
        pass


class EulerStatePredictor(BaseStatePredictor):
    def __init__(self, pred_step_size, sys_rhs, sys_out, dim_output, Nsteps):
        self.pred_step_size = pred_step_size
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.dim_output = dim_output
        self.Nsteps = Nsteps

    def predict_state(self, cur_state, action_sqn, k):
        next_state = cur_state + self.pred_step_size * self.sys_rhs(
            [], cur_state, action_sqn[k - 1, :]
        )
        return next_state

    def predict_state_sqn(self, state_start, observation, action_sqn):
        observation_sqn = np.zeros([self.Nsteps, self.dim_output])
        observation_sqn[0, :] = observation
        state = state_start
        for k in range(1, self.Nsteps):
            state = self.predict_state(state, action_sqn, k)
            observation_sqn[k, :] = self.sys_out(state)
        return observation_sqn
