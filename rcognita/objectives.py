from .utilities import rc
from abc import ABC, abstractmethod
from torch.nn import Module
import actors


class Objective(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class RunningObjective(Objective):
    def __init__(self, running_obj_model):
        self.running_obj_model = running_obj_model

    def __call__(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running objective etc.
        
        See class documentation.
        """
        observation = rc.to_col(observation)
        action = rc.to_col(action)

        chi = rc.concatenate([observation, action])

        running_obj = self.running_obj_model(chi, chi)

        return running_obj

class TabularObjective(Objective):

    def __init__(self, dim_state_space):



