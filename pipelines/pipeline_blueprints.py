from abc import ABCMeta, abstractmethod
from rcognita.utilities import nc


class AbstractPipeline(metaclass=ABCMeta):
    def load_config(self, env_config):
        self.env_config = env_config()

    def setup_env(self):
        self.__dict__.update(self.env_config.get_env())
        self.trajectory = []

    def config_to_pickle(self):
        self.env_config.config_to_pickle()

    @abstractmethod
    def system_initialization(self):
        pass

    @abstractmethod
    def state_predictor_initialization(self):
        pass

    @abstractmethod
    def controller_initialization(self):
        pass

    @abstractmethod
    def controller_initialization(self):
        pass

    @abstractmethod
    def simulator_initialization(self):
        pass

    @abstractmethod
    def logger_initialization(self):
        pass

    @abstractmethod
    def main_loop_raw(self):
        pass

    @abstractmethod
    def pipeline_execution(self):
        pass

    def upd_accum_obj(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).
        
        """

        self.accum_obj_val += self.stage_objective(observation, action) * delta
