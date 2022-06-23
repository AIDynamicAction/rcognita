from abc import ABCMeta, abstractmethod


class AbstractPresetPipeline(metaclass=ABCMeta):
    def load_config(self, env_config):
        self.env_config = env_config()

    def setup_env(self, args={}):
        self.__dict__.update(self.env_config.get_env())
        self.__dict__.update(args)
        self.trajectory = []
        return args

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
