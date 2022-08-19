from .utilities import nc
from abc import ABC, abstractmethod


class Objective(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class StageObjective(Objective):
    def __init__(self, stage_obj_model, observation_target=[]):
        self.stage_obj_model = stage_obj_model
        self.observation_target = observation_target

    def __call__(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running objective etc.
        
        See class documentation.
        """
        observation = nc.to_col(observation)
        action = nc.to_col(action)

        if self.observation_target == []:
            chi = nc.concatenate([observation, action])
        else:
            chi = nc.concatenate([(observation - self.observation_target), action])

        stage_obj = self.stage_obj_model(chi, chi)

        return stage_obj

