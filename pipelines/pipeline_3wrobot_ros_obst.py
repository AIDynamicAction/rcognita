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
from pipeline_3wrobot_NI_casadi import Pipeline3WRobotNICasadi
from config_blueprints import ConfigROS3WRobotNI

# ------------------------------------imports for interaction with ROS

import rospy

import os


class PipelineROS3wrobotNI(Pipeline3WRobotNI):
    def ros_harness_initialization(self):
        self.ros_preset_task = ROSHarness(
            control_mode=self.control_mode,
            state_init=[0, 0, 0],
            state_goal=self.state_init,
            my_ctrl_nominal=self.my_ctrl_nominal,
            my_sys=self.my_sys,
            my_ctrl_benchm=self.my_ctrl_benchm,
            action_manual=self.action_manual,
            stage_objective=self.stage_objective,
            my_logger=self.my_logger,
            datafiles=self.datafiles,
            dt=self.dt,
            pred_step_size=self.pred_step_size,
        )

    def execute_pipeline(self, **kwargs):
        self.load_config(ConfigROS3WRobotNI)
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_state_predictor()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        self.ros_harness_initialization()
        self.ros_preset_task.spin(
            is_print_sim_step=~self.no_print, is_log_data=self.is_log
        )


if __name__ == "__main__":
    rospy.init_node("ROS_preset_node")
    PipelineROS3wrobotNI().execute_pipeline()
