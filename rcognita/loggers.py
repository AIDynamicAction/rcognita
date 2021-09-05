#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the logger interface along with concrete realizations for each separate system

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from tabulate import tabulate

import csv

class Logger:
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
    
class Logger3WRobot(Logger):
    """
    Data logger for a 3-wheel robot with dynamic actuators.
    
    """
    def print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, r, icost, u):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt', 'F [N]', 'M [N m]']  
        row_data = [t, xCoord, yCoord, alpha, v, omega, r, icost, u[0], u[1]]  
        row_format = ('8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, v, omega, r, icost, u):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, v, omega, r, icost, u[0], u[1]])

class Logger3WRobotNI(Logger):
    """
    Data logger for a 3-wheel robot with static actuators.
    
    """
    def print_sim_step(self, t, xCoord, yCoord, alpha, r, icost, u):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'r', 'int r dt', 'v [m/s]', 'omega [rad/s]']  
        row_data = [t, xCoord, yCoord, alpha, r, icost, u[0], u[1]]  
        row_format = ('8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, xCoord, yCoord, alpha, r, icost, u):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, r, icost, u[0], u[1]])
                
class Logger2Tank(Logger):
    """
    Data logger for a 2-tank system.
    
    """
    def print_sim_step(self, t, h1, h2, p, r, icost):
    # alphaDeg = alpha/np.pi*180      
    
        row_header = ['t [s]', 'h1', 'h2', 'p', 'r', 'int r dt']  
        row_data = [t, h1, h2, p, r, icost]  
        row_format = ('8.1f', '8.4f', '8.4f', '8.4f', '8.4f', '8.2f')   
        table = tabulate([row_header, row_data], floatfmt=row_format, headers='firstrow', tablefmt='grid')
    
        print(table)
    
    def log_data_row(self, datafile, t, h1, h2, p, r, icost):
        with open(datafile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, h1, h2, p, r, icost])                