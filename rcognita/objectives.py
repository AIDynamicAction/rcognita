from .utilities import nc
from abc import ABC, abstractmethod
from torch.nn import Module
import actors


class Objective(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class StageObjective(Objective):
    def __init__(self, stage_obj_model):
        self.stage_obj_model = stage_obj_model

    def __call__(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running objective etc.
        
        See class documentation.
        """
        observation = nc.to_col(observation)
        action = nc.to_col(action)

        chi = nc.concatenate([observation, action])

        stage_obj = self.stage_obj_model(chi, chi)

        return stage_obj


class TorchMPCObjective(Module, actors.ActorMPC):
    def forward(self, action_sqn, observation):
        self.objective(action_sqn, observation)


class TorchSQLObjective(Module, actors.ActorMPC):
    def forward(self, action_sqn, observation):
        self.objective(action_sqn, observation)


class TorchRQLObjective(Module, actors.ActorMPC):
    def forward(self, action_sqn, observation):
        self.objective(action_sqn, observation)
