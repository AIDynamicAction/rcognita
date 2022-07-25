from .npcasadi_api import SymbolicHandler


class Objectives:
    def __init__(self, stage_obj_model, observation_target=[]):
        self.stage_obj_model = stage_obj_model
        self.observation_target = observation_target

    def stage_obj(self, observation, action, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = npcsd.concatenate([observation, action])
        else:
            chi = npcsd.concatenate(
                [(observation - self.observation_target).T, action.T]
            )

        stage_obj = self.stage_obj_model(chi, chi, is_symbolic=is_symbolic)

        return stage_obj

