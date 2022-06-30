#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator) 
with connection to ROS.

"""

import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import rcognita

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        + f"rcognita ({rcognita.__version__}) "
        + f"located in cloned repository at '{PARENT_DIR}'. "
        + f"If you are willing to use your locally installed rcognita, "
        + f"run this script ('{os.path.basename(__file__)}') outside "
        + f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        + f"locally installed rcognita ({rcognita.__version__}). "
        + f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita.ROS_harnesses import ROSHarness
from pipeline_3wrobot_NI import Pipeline3WRobotNI
from config_blueprints import Config3WRobotNI

# ------------------------------------imports for interaction with ROS

import rospy

import os


class PipelineROS3wrobotNI(Pipeline3WRobotNI):
    def ros_harness_initialization(self):
        self.ros_preset_task = ROSHarness(
            self.ctrl_mode,
            [0, 0, 0],
            self.state_init,
            self.my_ctrl_nominal,
            self.my_sys,
            self.my_ctrl_benchm,
            self.action_manual,
            self.my_logger,
            self.datafiles,
            self.dt,
            self.pred_step_size,
        )

    def pipeline_execution(self, **kwargs):
        self.load_config(Config3WRobotNI)
        self.setup_env()
        self.__dict__.update(kwargs)
        self.system_initialization()
        self.state_predictor_initialization()
        self.optimizers_initialization()
        self.controller_initialization()
        self.simulator_initialization()
        self.logger_initialization()
        self.ros_harness_initialization()
        self.ros_preset_task.spin(
            is_print_sim_step=~self.no_print, is_log_data=self.is_log
        )


if __name__ == "__main__":
    rospy.init_node("ROS_preset_node")
    PipelineROS3wrobotNI().pipeline_execution()
