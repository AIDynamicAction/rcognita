import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita

from config_blueprints import Config3WRobotNI
from pipeline_blueprints import PipelineWithDefaults

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita import (
    controllers,
    visuals,
    simulator,
    systems,
    loggers,
    state_predictors,
    optimizers,
    objectives,
    models,
    utilities,
)
from rcognita.loggers import logger3WRobotNI
from datetime import datetime
from rcognita.utilities import on_key_press
from rcognita.actors import ActorTabular

from rcognita.critics import CriticTabular

from rcognita.utilities import rc

from rcognita.scenarios import TabularScenario

from enum import IntEnum


class Actions(IntEnum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


RIGHT = Actions.RIGHT
LEFT = Actions.LEFT
UP = Actions.UP
DOWN = Actions.DOWN

action_space = [RIGHT, LEFT, UP, DOWN]
grid_size = [10, 10]


class PipelineTabular(PipelineWithDefaults):
    config = None

    def initialize_logger(self):
        pass

    def initialize_system(self):
        self.my_sys = systems.TabularSystem(grid_size)

    def initialize_state_predictor(self):
        self.state_predictor = state_predictors.TrivialStatePredictor(self.my_sys)

    def initialize_optimizers(self):
        self.actor_optimizer = optimizers.BruteForceOptimizer(5, action_space)

    def initialize_models(self):
        self.actor_model = models.LookupTable(grid_size)

    def initialize_objectives(self):
        self.running_objective = 

    def initialize_actor_critic(self):
        self.critic = CriticTabular(grid_size)
        self.actor = ActorTabular(
            dim_world=grid_size,
            Nactor=0,
            dim_input=len(grid_size),
            dim_output=len(grid_size),
            control_mode="tabular",
            action_init=RIGHT,
            state_predictor=self.state_predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            running_obj=[],
            model=self.actor_model,
            gamma=1,
            action_space=action_space,
        )

    def initialize_scenario(self):
        self.scenario = TabularScenario(self.actor, self.critic, 10)

    def execute_pipeline(self):
        self.initialize_system()
        self.initialize_state_predictor()
        self.initialize_optimizers()
        self.initialize_models()
        self.initialize_actor_critic()
        self.initialize_scenario()
        self.scenario.run()


if __name__ == "__main__":

    PipelineTabular().execute_pipeline()
