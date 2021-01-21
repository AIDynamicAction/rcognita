#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:38:59 2021

@author: Pavel Osinenko
"""

"""
=============================================================================
rcognita

https://github.com/AIDynamicAction/rcognita

Python framework for hybrid simulation of predictive reinforcement learning agents and classical controllers 

=============================================================================

This module:

loggers

=============================================================================

Remark: 

All vectors are treated as of type [n,]
All buffers are treated as of type [L, n] where each row is a vector
Buffers are updated from bottom to top
"""
# !pip install tabulate <-- to install this
from tabulate import tabulate

import csv

class logger:
    """
    Interface class for data loggers.
    Concrete loggers, associated with concrete system-controller setups, are should be built upon this class.
    To design a concrete logger: inherit this class, override:
        | :func:`~loggers.logger.print_sim_step` :
        | print a row of data of a single simulation step, typically into the console (required)
        | :func:`~loggers.logger.log_data_row` :
        | same as above, but write to a file (required)
    
    """
    
    def print_sim_step():
        pass
    
    def log_data_row():
        pass
    
class logger_3wrobot:
    """
    Data logger for a 3-wheel robot.
    
    """
    def print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, r, icost, u):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt', 'F [N]', 'M [N m]']  
        row_data = [t, xCoord, yCoord, alpha, v, omega, r, icost, u[0], u[1]]  
        row_format = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, v, omega, r, icost, u):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, v, omega, r, icost, u[0], u[1]])