#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the logger interface along with concrete realizations for each separate system.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from tabulate import tabulate

import csv
import numpy as np


class Logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.Logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required).
        | :func:`~loggers.Logger.log_data_row` :
        | same as above, but write to a file (required).
    
    """

    def __init__(self, state_components_strings, action_components_strings):
        self.state_components_strings = state_components_strings
        self.action_components_strings = action_components_strings
        self.row_header = [
            "t [s]",
            *self.state_components_strings,
            *self.action_components_strings,
            "running_obj",
            "accum_obj",
        ]
        self.row_format = tuple(["8.3f" for _ in self.row_header])

    def print_sim_step(self, t, state_full, action, running_obj, accum_obj):
        row_data = [t, *np.array(state_full), *np.array(action), running_obj, accum_obj]

        table = tabulate(
            [self.row_header, row_data],
            floatfmt=self.row_format,
            headers="firstrow",
            tablefmt="grid",
        )

        print(table)

    def log_data_row(self, datafile, t, state_full, action, running_obj, accum_obj):
        with open(datafile, "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [t, *state_full, *action, running_obj, accum_obj,]
            )


logger3WRobot = Logger(
    ["x [m]", "y [m]", "alpha [rad]", "v [m/s]", "omega [rad/s]"], ["F [N]", "M [N m]"]
)

logger3WRobotNI = Logger(
    ["x [m]", "y [m]", "alpha [rad]"], ["v [m/s]", "omega [rad/s]"]
)


class Logger3WRobotNI(Logger):
    """
    Data logger for a 3-wheel robot with static actuators.
    
    """

    def print_sim_step(self, t, xCoord, yCoord, alpha, running_obj, accum_obj, action):
        # alphaDeg = alpha/np.pi*180

        row_header = [
            "t [s]",
            "x [m]",
            "y [m]",
            "alpha [rad]",
            "running_obj",
            "accum_obj",
            "v [m/s]",
            "omega [rad/s]",
        ]
        row_data = [
            t,
            xCoord,
            yCoord,
            alpha,
            running_obj,
            accum_obj,
            action[0],
            action[1],
        ]
        row_format = ("8.3f", "8.3f", "8.3f", "8.3f", "8.1f", "8.1f", "8.3f", "8.3f")
        table = tabulate(
            [row_header, row_data],
            floatfmt=row_format,
            headers="firstrow",
            tablefmt="grid",
        )

        print(table)

    def log_data_row(
        self, datafile, t, xCoord, yCoord, alpha, running_obj, accum_obj, action
    ):
        with open(datafile, "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [t, xCoord, yCoord, alpha, running_obj, accum_obj, action[0], action[1]]
            )


class Logger2Tank(Logger):
    """
    Data logger for a 2-tank system.
    
    """

    def print_sim_step(self, t, h1, h2, p, running_obj, accum_obj):
        # alphaDeg = alpha/np.pi*180

        row_header = ["t [s]", "h1", "h2", "p", "running_obj", "accum_obj"]
        row_data = [t, h1, h2, p, running_obj, accum_obj]
        row_format = ("8.1f", "8.4f", "8.4f", "8.4f", "8.4f", "8.2f")
        table = tabulate(
            [row_header, row_data],
            floatfmt=row_format,
            headers="firstrow",
            tablefmt="grid",
        )

        print(table)

    def log_data_row(self, datafile, t, h1, h2, p, running_obj, accum_obj):
        with open(datafile, "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, h1, h2, p, running_obj, accum_obj])
