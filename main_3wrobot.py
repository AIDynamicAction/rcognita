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
import rcognita.simulator as simulator
import rcognita.systems as systems
import rcognita.controllers as controllers
import rcognita.loggers as loggers
import rcognita.visuals as visuals
from rcognita.models import model_SS, model_NN
from rcognita.utilities import on_key_press
from argparse import ArgumentParser
import os
import warnings
import shapely
warnings.filterwarnings('ignore')

#------------------------------------user settings : : system
# System
parser = ArgumentParser()
parser.add_argument('--mode', type=int, default=3)
parser.add_argument('--system', type=str, default='kinematic')
parser.add_argument('--init_x', type=int, default=None)
parser.add_argument('--init_y', type=int, default=None)
parser.add_argument('--init_alpha', type=float, default=None)
parser.add_argument('--ndots', type=int, default=25)
parser.add_argument('--radius', type=int, default=5)

parser.add_argument('--data_folder', type=str, default=None)
parser.add_argument('--weights_path', type=str, default=None)

parser.add_argument('--dt', type=float, default=0.05)
parser.add_argument('--Nactor', type=int, default=6)
parser.add_argument('--pred_step_size', type=int, default=5)

parser.add_argument('--is_log_data', type=bool, default=True)
parser.add_argument('--is_print_sim_step', type=bool, default=False)
parser.add_argument('--is_visualization', type=bool, default=False)
parser.add_argument('--is_estimate_model', type=bool, default=False)
parser.add_argument('--is_use_offline_model', type=bool, default=False)
parser.add_argument('--is_prob_noise', type=bool, default=False)

args = parser.parse_args()

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%H-%M-%S")

if args.data_folder == None:
    new_path = 'data'
    data_folder = os.path.join(new_path, date)
    os.makedirs(data_folder, exist_ok=True)
else:
    os.makedirs(args.data_folder, exist_ok=True)
    data_folder = args.data_folder

if args.system == 'kinematic':
    dim_state = 3
    dim_input = 2
    dim_output = 3
    dim_disturb = 2

    Fmin = -0.22
    Fmax = 0.22
    Mmin = -2
    Mmax = 2

    m = 10 # [kg]
    I = 1 # [kg m^2]

    R1 = np.diag([1, 100, 0.001, 0, 0])
    R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


elif args.system == 'endi':
    dim_state = 5
    dim_input = 2
    dim_output = 5
    dim_disturb = 2

    Fmin = -5
    Fmax = 5
    Mmin = -1
    Mmax = 1

    # System parameters
    m = 10 # [kg]
    I = 1 # [kg m^2]

    R1 = np.diag([10, 15, 10, 0, 0, 5, 17])  # No mixed terms, full-state measurement
    R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y

# Disturbance
sigma_q = 1e-3 * np.ones(dim_disturb)
mu_q = np.zeros(dim_disturb)
tau_q = np.ones(dim_disturb)

#------------------------------------user settings : : simulation
t0 = 0
t1 = 300
Nruns = 1
dt = args.dt

x0 = np.zeros(dim_state)
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

#------------------------------------user settings : : model estimator
model_est_stage = 2 # [s]
model_est_period = 1*dt # [s]
model_order = 5
prob_noise_pow = 8
# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

#------------------------------------user settings : : controller

# Manual control
Fman = -3
Nman = -1
uMan = np.array([Fman, Nman])

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
critic_struct_Q = 3
critic_struct_V = 3

#------------------------------------user settings : : main switches
is_estimate_model = args.is_estimate_model
is_use_offline_model = args.is_use_offline_model
ctrl_mode = args.mode

is_log_data = args.is_log_data
is_visualization = args.is_visualization
is_print_sim_step = args.is_print_sim_step
is_prob_noise = args.is_prob_noise

is_disturb = 0

# Static or dynamic controller
is_dyn_ctrl = 0
#------------------------------------model settings and initialization
lr = 10e-2
feature_size = 5
output_shape = 3
layers = 1
hidden_size = 24
epochs = 500

model = model_NN(feature_size, output_shape, hidden_size, layers)
optimizer = None
criterion = None
#
# if args.is_use_offline_model:
#     model.upload_weights(args.weights_path)
    #

if args.init_alpha != None and args.init_x == None and args.init_y == None:
    x = args.radius * np.cos(args.init_alpha)
    y = args.radius * np.sin(args.init_alpha)
    coords = [(x, y, args.init_alpha)]
else:
    coords = [(args.init_x, args.init_y, args.init_alpha)]

#------------------------------------initialization : : system
for coord in coords:
    x0[0] = coord[0]
    x0[1] = coord[1]
    x0[2] = coord[2]

    if args.system == 'kinematic':
        my_3wrobot = systems.sys_3wrobot_kinematic(sys_type="diff_eqn", dim_state=dim_state, dim_input=dim_input, dim_output=dim_output, dim_disturb=dim_disturb,
                                         pars=[m, I],
                                         ctrl_bnds=np.array([[Fmin, Fmax], [Mmin, Mmax]]))
        my_logger = loggers.logger_3wrobot_kinematic()
    elif args.system =='endi':
        my_3wrobot = systems.sys_3wrobot_endi(sys_type="diff_eqn", dim_state=dim_state, dim_input=dim_input, dim_output=dim_output, dim_disturb=dim_disturb,
                                         pars=[m, I],
                                         ctrl_bnds=np.array([[Fmin, Fmax], [Mmin, Mmax]]))
        my_logger = loggers.logger_3wrobot_endi()

    y0 = my_3wrobot.out(x0)

    xCoord0 = x0[0]
    yCoord0 = x0[1]
    alpha0 = x0[2]
    alpha_deg_0 = alpha0/2/np.pi

    #------------------------------------initialization : : controller
    ctrl_bnds = np.array([[Fmin, Fmax], [Mmin, Mmax]])

    if args.system.lower() == 'kinematic':
        my_ctrl_nominal_3wrobot = controllers.ctrl_nominal_kinematic_3wrobot(ctrl_bnds=ctrl_bnds, pose_goal = [2.0, 3.0, 0.0], t0=t0, sampling_time=dt)
    elif args.system.lower() == 'endi':
        my_ctrl_nominal_3wrobot = controllers.ctrl_nominal_3wrobot(m, I, ctrl_gain=0.5, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)

    my_ctrl_RL = controllers.ctrl_RL_pred(dim_input, dim_output,
                                          ctrl_mode, ctrl_bnds=ctrl_bnds,
                                          t0=t0, sampling_time=dt, Nactor=Nactor, pred_step_size=pred_step_size,
                                          sys_rhs=my_3wrobot._state_dyn, sys_out=my_3wrobot.out,
                                          x_sys=x0, is_prob_noise=is_prob_noise,
                                          prob_noise_pow = prob_noise_pow, model_est_stage=model_est_stage, model_est_period=model_est_period,
                                          buffer_size=buffer_size,
                                          model_order=model_order, model_est_checks=model_est_checks,
                                          gamma=gamma, Ncritic=Ncritic, critic_period=critic_period, critic_struct_Q=critic_struct_Q,
                                          critic_struct_V=critic_struct_V, rcost_struct=rcost_struct, model=model, optimizer=optimizer,criterion=criterion,
                                          is_estimate_model=is_estimate_model, is_use_offline_model=is_use_offline_model, rcost_pars=[R1, R2],
                                          lr = lr, feature_size = feature_size, output_shape = output_shape, layers = layers, hidden_size = hidden_size, epochs = epochs)

    #------------------------------------initialization : : simulator
    my_simulator = simulator.simulator(sys_type="diff_eqn",
                                       closed_loop_rhs=my_3wrobot.closed_loop_rhs,
                                       sys_out=my_3wrobot.out,
                                       x0=x0, q0=q0, u0=u0, t0=t0, t1=t1, dt=dt, max_step=dt/2, first_step=1e-6, atol=atol, rtol=rtol, is_dyn_ctrl=is_dyn_ctrl)

    #------------------------------------initialization : : logger

    datafiles = [None] * Nruns
    for k in range(0, Nruns):
        datafiles[k] = f'{data_folder}/RLsim__{ctrl_mode}_{dt}_{Nactor}_{args.pred_step_size}_{x0[0]:.2f}_{x0[1]:.2f}__{time}__run{k+1}.csv'

        if is_log_data:
            with open(datafiles[k], 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                if args.system == 'kinematic':
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'r', 'int r dt', 'v [m/s]', 'omega [rad/s]'])
                elif args.system == 'endi':
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt', 'F [N]', 'M [N m]'] )


    # Do not display annoying warnings when print is on
    if is_print_sim_step:
        warnings.filterwarnings('ignore')

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
