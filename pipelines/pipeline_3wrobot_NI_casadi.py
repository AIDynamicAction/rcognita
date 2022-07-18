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
import numpy as np
from rcognita.utilities import rep_mat
from rcognita.npcasadi_api import SymbolicHandler

from config_blueprints import Config3WRobotNI
from pipeline_blueprints import AbstractPipeline

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
)
from datetime import datetime
from rcognita.utilities import on_key_press
from pipeline_3wrobot_NI import Pipeline3WRobotNI


class Pipeline3WRobotNICasadi(Pipeline3WRobotNI):
    def optimizers_initialization(self):

        self.actor_optimizer = optimizers.RcognitaOptimizer.casadi_actor_optimizer(
            actor_opt_method="ipopt", ctrl_bnds=self.ctrl_bnds, Nactor=self.Nactor
        )
        self.critic_optimizer = optimizers.RcognitaOptimizer.casadi_critic_optimizer(
            critic_opt_method="ipopt",
            critic_struct=self.critic_struct,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
        )


if __name__ == "__main__":

    Pipeline3WRobotNICasadi().pipeline_execution()
