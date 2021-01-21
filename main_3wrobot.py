#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:40:52 2021

@author: Pavel Osinenko
"""

"""
=============================================================================
rcognita

https://github.com/AIDynamicAction/rcognita

Python framework for hybrid simulation of predictive reinforcement learning agents and classical controllers

=============================================================================

This module:

main loop for a 3-wheel robot

=============================================================================

Remark:

All vectors are treated as of type [n,]
All buffers are treated as of type [L, n] where each row is a vector
Buffers are updated from bottom to top
"""

import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import simulator
import systems
import controllers
import loggers
import visuals
from utilities import on_key_press, gen_init_coords_angles
from argparse import ArgumentParser
import os
import warnings
warnings.filterwarnings('ignore')

#------------------------------------user settings : : system
# System
parser = ArgumentParser()
parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--init_x', type=int, default=None)
parser.add_argument('--init_y', type=int, default=None)
parser.add_argument('--init_alpha', type=float, default=None)
parser.add_argument('--ndots', type=int, default=25)
parser.add_argument('--radius', type=int, default=5)
parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--dt_bash', type=int, default=1)
parser.add_argument('--is_bash', type=bool, default=False)
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--Nactor', type=int, default=6)
parser.add_argument('--pred_step_size', type=int, default=5)
parser.add_argument('--is_log_data', type=bool, default=True)
parser.add_argument('--is_print_sim_step', type=bool, default=False)
parser.add_argument('--is_visualization', type=bool, default=False)

args = parser.parse_args()


date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh")

data_folder = 'data'
new_path = os.path.join(data_folder, date)
os.makedirs(new_path, exist_ok=True)
if args.folder == None:
    data_folder = os.path.join(new_path, time)
else:
    os.makedirs(f'{new_path}/{args.folder}', exist_ok=True)
    data_folder = os.path.join(new_path, args.folder)

os.makedirs(data_folder, exist_ok=True)

dim_state = 5
dim_input = 2
dim_output = 5
dim_disturb = 2

# System parameters
m = 10 # [kg]
I = 1 # [kg m^2]

# Disturbance
sigma_q = 1e-3 * np.ones(dim_disturb)
mu_q = np.zeros(dim_disturb)
tau_q = np.ones(dim_disturb)

#------------------------------------user settings : : simulation
t0 = 0
t1 = 300
Nruns = 1

x0 = np.zeros(dim_state)
coords = gen_init_coords_angles(args.radius,  args.ndots)
# x0[0] = coords[0]
# x0[1] = coords[1]
# x0[2] = coords[2]

u0 = 0 * np.ones(dim_input)
q0 = 0 * np.ones(dim_disturb)

# Solver
atol = 1e-5
rtol = 1e-3

# xy-plane
xMin = -10
xMax = 10
yMin = -10
yMax = 10

#------------------------------------user settings : : digital elements
# Digital elements sampling time
if args.is_bash == True:
    dt = args.dt_bash * 0.01 # [s], controller sampling time
else:
    dt = args.dt
# sampleFreq = 1/dt # [Hz]

# Parameters
# cutoff = 1 # [Hz]

# Digital differentiator filter order
# diffFiltOrd = 4

#------------------------------------user settings : : model estimator
model_est_stage = 2 # [s]
model_est_period = 1*dt # [s]

model_order = 5

prob_noise_pow = 8

# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

#------------------------------------user settings : : controller
# u[0]: Pushing force F [N]
# u[1]: Steering torque M [N m]

# Manual control
Fman = -3
Nman = -1
uMan = np.array([Fman, Nman])

# Control constraints
Fmin = -5
Fmax = 5
Mmin = -1
Mmax = 1

# Control horizon length
Nactor = args.Nactor

# Should be a multiple of dt
pred_step_size = args.pred_step_size * dt # [s]

# Size of data buffers (used, e.g., in model estimation and critic)
buffer_size = 200

#------------------------------------user settings : : RL
# Running cost structure and parameters
# Notation: chi = [y, u]
# 1     - quadratic chi.T R1 chi
# 2     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
rcost_struct = 1

R1 = np.diag([10, 15, 10, 0, 0, 5, 17])  # No mixed terms, full-state measurement
# R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
# R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# Critic stack size, not greater than buffer_size
Ncritic = 50

# Discounting factor
gamma = 1

# Critic is updated every critic_period seconds
critic_period = 5*dt # [s]

# Critic structure choice
# 1 - quadratic-linear
# 2 - quadratic
# 3 - quadratic, no mixed terms
# 4 - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...] u[0]^2 + ...
critic_struct = 3

#------------------------------------user settings : : main switches
is_log_data = args.is_log_data
is_visualization = args.is_visualization
is_print_sim_step = args.is_print_sim_step

is_disturb = 0

# Static or dynamic controller
is_dyn_ctrl = 0

# Control mode
#
#   Modes with online model estimation are experimental
#
# 0     - manual constant control (only for basic testing)
# -1    - nominal parking controller (for benchmarking optimal controllers)
# 1     - model-predictive control (MPC). Prediction via discretized true model
# 2     - adaptive MPC. Prediction via estimated model
# 3     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via discretized true model
# 4     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via estimated model
# 5     - RL: stacked Q-learning. Prediction via discretized true model
# 6     - RL: stacked Q-learning. Prediction via estimated model
ctrl_mode = args.mode

#------------------------------------initialization : : system

if args.init_x != None and args.init_y != None and args.init_alpha != None:
    coords = [(args.init_x, args.init_y, args.init_alpha)]
if args.init_alpha != None and args.init_x == None and args.init_y == None:
    x = args.radius * np.cos(args.init_alpha)
    y = args.radius * np.sin(args.init_alpha)
    coords = [(x, y, args.init_alpha)]
for coord in coords:
    x0[0] = coord[0]
    x0[1] = coord[1]
    x0[2] = coord[2]
    my_3wrobot = systems.sys_3wrobot(sys_type="diff_eqn", dim_state=dim_state, dim_input=dim_input, dim_output=dim_output, dim_disturb=dim_disturb,
                                     pars=[m, I],
                                     ctrl_bnds=np.array([[Fmin, Fmax], [Mmin, Mmax]]))

    y0 = my_3wrobot.out(x0)

    xCoord0 = x0[0]
    yCoord0 = x0[1]
    alpha0 = x0[2]
    alpha_deg_0 = alpha0/2/np.pi

    #------------------------------------initialization : : controller
    ctrl_bnds = np.array([[Fmin, Fmax], [Mmin, Mmax]])

    my_ctrl_nominal_3wrobot = controllers.ctrl_nominal_3wrobot(m, I, ctrl_gain=0.5, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)

    my_ctrl_RL = controllers.ctrl_RL_pred(dim_input, dim_output,
                                          ctrl_mode, ctrl_bnds=ctrl_bnds,
                                          t0=t0, sampling_time=dt, Nactor=Nactor, pred_step_size=pred_step_size,
                                          sys_rhs=my_3wrobot._state_dyn, sys_out=my_3wrobot.out,
                                          x_sys=x0,
                                          prob_noise_pow = prob_noise_pow, model_est_stage=model_est_stage, model_est_period=model_est_period,
                                          buffer_size=buffer_size,
                                          model_order=model_order, model_est_checks=model_est_checks,
                                          gamma=gamma, Ncritic=Ncritic, critic_period=critic_period, critic_struct=critic_struct, rcost_struct=rcost_struct, rcost_pars=[R1, R2])

    #------------------------------------initialization : : simulator
    my_simulator = simulator.simulator(sys_type="diff_eqn",
                                       closed_loop_rhs=my_3wrobot.closed_loop_rhs,
                                       sys_out=my_3wrobot.out,
                                       x0=x0, q0=q0, u0=u0, t0=t0, t1=t1, dt=dt, max_step=dt/2, first_step=1e-6, atol=atol, rtol=rtol, is_dyn_ctrl=is_dyn_ctrl)

    #------------------------------------initialization : : logger

    datafiles = [None] * Nruns
    for k in range(0, Nruns):
        datafiles[k] = f'{data_folder}/RLsim__{ctrl_mode}_{dt}_{Nactor}_{args.pred_step_size}_{x0[0]:.2f}_{x0[1]:.2f}__run{k+1}.csv'.format(run=k+1)

        if is_log_data:
            with open(datafiles[k], 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt', 'F [N]', 'M [N m]'] )

    # Do not display annoying warnings when print is on
    if is_print_sim_step:
        warnings.filterwarnings('ignore')

    my_logger = loggers.logger_3wrobot()

    #------------------------------------main loop
    if is_visualization:

        ksi0 = my_simulator.ksi

        my_animator = visuals.animator_3wrobot(objects=(my_simulator, my_3wrobot, my_ctrl_nominal_3wrobot, my_ctrl_RL, datafiles, controllers.ctrl_selector, my_logger),
                                               pars=(x0, u0, t0, t1, ksi0, xMin, xMax, yMin, yMax, ctrl_mode, uMan, Fmin, Mmin, Fmax, Mmax, Nruns,
                                                     is_print_sim_step, is_log_data, 0, []))

        anm = animation.FuncAnimation(my_animator.fig_sim,
                                      my_animator.animate,
                                      init_func=my_animator.init_anim,
                                      blit=False, interval=dt/1e6, repeat=False)

        cId = my_animator.fig_sim.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, anm))

        anm.running = True

        my_animator.fig_sim.tight_layout()

        plt.show()

    else:
        run_curr = 1
        datafile = datafiles[0]

        while True:

            my_simulator.sim_step()

            t, x, y, ksi = my_simulator.get_sim_step_data()

            u = controllers.ctrl_selector(t, y, uMan, my_ctrl_nominal_3wrobot, my_ctrl_RL, ctrl_mode)

            my_3wrobot.receive_action(u)
            my_ctrl_RL.receive_sys_state(my_3wrobot._x)
            my_ctrl_RL.upd_icost(y, u)

            xCoord = ksi[0]
            yCoord = ksi[1]
            alpha = ksi[2]
            v = ksi[3]
            omega = ksi[4]

            r = my_ctrl_RL.rcost(y, u)
            icost = my_ctrl_RL.icost_val

            if is_print_sim_step:
                my_logger.print_sim_step(t, xCoord, yCoord, alpha, v, omega, r, icost, u)

            if is_log_data:
                my_logger.log_data_row(datafile, t, xCoord, yCoord, alpha, v, omega, r, icost, u)

            if t >= t1:
                if is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = run_curr))

                run_curr += 1

                if run_curr > Nruns:
                    break

                if is_log_data:
                    datafile = datafiles[run_curr-1]

                # Reset simulator
                my_simulator.status = 'running'
                my_simulator.t = t0
                my_simulator.y = ksi0

                if ctrl_mode > 0:
                    my_ctrl_RL.reset(t0)
                else:
                    my_ctrl_nominal_3wrobot.reset(t0)

                icost = 0
                t = 0
                r = 0
