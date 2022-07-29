from .utilities import nc


class Objectives:
    def __init__(self, stage_obj_model, observation_target=[]):
        self.stage_obj_model = stage_obj_model
        self.observation_target = observation_target

    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """

        if self.observation_target == []:
            chi = nc.concatenate([observation, action])
        else:
            chi = nc.concatenate([(observation - self.observation_target), action])

        stage_obj = self.stage_obj_model(chi, chi)

        return stage_obj

