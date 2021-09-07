#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: nonlinear double-tank system

"""

import warnings
import csv
from datetime import datetime
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from rcognita import simulator
from rcognita import systems
from rcognita import controllers
from rcognita import loggers
from rcognita import visuals
from rcognita.utilities import on_key_press

#------------------------------------user settings : : main switches
is_log_data = 0
is_visualization = 1
is_print_sim_step = 1

is_disturb = 0

# Static or dynamic controller
is_dyn_ctrl = 0

# Control mode
#
# Modes with online model estimation are experimental
#
# 'manual'      - manual constant control (only for basic testing)
# 'nominal'     - nominal parking controller (for benchmarking optimal controllers)
# 'MPC'         - model-predictive control (MPC)
# 'RQL'         - reinforcement learning: Q-learning with Ncritic roll-outs of running cost
# 'SQL'         - reinforcement learning: stacked Q-learning
# 'JACS'        - Joint actor-critic (stabilizing)
ctrl_mode = 'MPC'

#------------------------------------user settings : : system
# System
dim_state = 2
dim_input = 1
dim_output = 2
dim_disturb = 1

# System parameters
tau1 = 18.4
tau2 = 24.4
K1 = 1.3
K2 = 1
K3 = 0.2

#------------------------------------user settings : : simulation
t0 = 0
t1 = 100
Nruns = 1

# x0 = np.ones(dim_state)
x0 = np.array([0.25, 0.75])

u0 = 0.5 * np.ones(dim_input)

q0 = 0 * np.ones(dim_disturb)

# Solver
atol = 1e-5
rtol = 1e-3

#------------------------------------user settings : : digital elements
# Digital elements sampling time
dt = 0.1 # [s], controller sampling time
# sampleFreq = 1/dt # [Hz]

# Parameters
# cutoff = 1 # [Hz]

# Digital differentiator filter order
# diffFiltOrd = 4

#------------------------------------user settings : : model estimator
is_est_model = 0
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
uMan = np.array([0.1])

# Control constraints
u_min = 0
u_max = 1

# Control horizon length
Nactor = 5

# Should be a multiple of dt
pred_step_size = 5*dt # [s]

# Size of data buffers (used, e.g., in model estimation and critic)
buffer_size = 200

#------------------------------------user settings : : RL
# Running cost structure and parameters
# Notation: chi = [y, u]
# 'quadratic'     - quadratic chi.T R1 chi 
# 'biquadratic'     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
rcost_struct = 'quadratic'

R1 = np.diag([10, 10, 0])  # No mixed terms, full-state measurement

# Target filling of the tanks
y_target = np.array([0.5, 0.5])

# Critic stack size, not greater than buffer_size
Ncritic = 50

# Discounting factor
gamma = 1

# Critic is updated every critic_period seconds
critic_period = 5*dt # [s]

# Actor and critic structure choice
# 'quad-lin' - quadratic-linear
# 'quadratic' - quadratic
# 'quad-nomix' - quadratic, no mixed terms
# 'quad-mix' - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...] u[0]^2 + ... (only Q-function critic)
critic_struct = 'quad-nomix'
actor_struct = 'quad-nomix'

#------------------------------------initialization : : system
ctrl_bnds = np.array([[u_min], [u_max]]).T

my_2tank = systems.Sys2Tank(sys_type="diff_eqn", dim_state=dim_state, dim_input=dim_input, dim_output=dim_output, dim_disturb=dim_disturb,
                             pars=[tau1, tau2, K1, K2, K3],
                             ctrl_bnds=ctrl_bnds)

y0 = my_2tank.out(x0)

#------------------------------------initialization : : model

# Euler scheme
# get_next_state = lambda x: x + dt * self.sys_rhs([], x, myU[k-1, :], [])
# sys_out = my_3wrobot.out

# If is_use_offline_est_model
# if model_type == 1 # SS
# get_next_state = lambda x: my_NN.get_next(x, ...dt)
# ... 2 # NN

#------------------------------------initialization : : controller
my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input, dim_output,
                                            ctrl_mode, ctrl_bnds=ctrl_bnds,
                                            t0=t0, sampling_time=dt, Nactor=Nactor, pred_step_size=pred_step_size,
                                            sys_rhs=my_2tank._state_dyn, sys_out=my_2tank.out,
                                            # get_next_state = get_next_state, sys_out = sys_out,
                                            x_sys=x0,
                                            prob_noise_pow = prob_noise_pow, is_est_model=is_est_model, model_est_stage=model_est_stage, model_est_period=model_est_period,
                                            buffer_size=buffer_size,
                                            model_order=model_order, model_est_checks=model_est_checks,
                                            gamma=gamma, Ncritic=Ncritic, critic_period=critic_period, critic_struct=critic_struct, rcost_struct=rcost_struct,
                                            rcost_pars=[R1],
                                            y_target=y_target)

#------------------------------------initialization : : simulator
my_simulator = simulator.Simulator(sys_type="diff_eqn",
                                   closed_loop_rhs=my_2tank.closed_loop_rhs,
                                   sys_out=my_2tank.out,
                                   x0=x0, q0=q0, u0=u0, t0=t0, t1=t1, dt=dt, max_step=dt/2, first_step=1e-6, atol=atol, rtol=rtol, is_dyn_ctrl=is_dyn_ctrl)

#------------------------------------initialization : : logger
data_folder = 'data'

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns
for k in range(0, Nruns):
    datafiles[k] = data_folder + '/RLsim__2tank' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
    
    if is_log_data:
        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['t [s]', 'h1', 'h2', 'p', 'r', 'int r dt'])

# Do not display annoying warnings when print is on
if is_print_sim_step:
    warnings.filterwarnings('ignore')
    
my_logger = loggers.Logger2Tank()

#------------------------------------main loop
if is_visualization:
    
    ksi0 = my_simulator.ksi
    
    my_animator = visuals.Animator2Tank(objects=(my_simulator, my_2tank, [], my_ctrl_opt_pred, datafiles, controllers.ctrl_selector, my_logger),
                                           pars=(x0, u0, t0, t1, ksi0, ctrl_mode, uMan, u_min, u_max, Nruns,
                                                 is_print_sim_step, is_log_data, 0, [], y_target))

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
        
        u = controllers.ctrl_selector(t, y, uMan, [], my_ctrl_opt_pred, ctrl_mode)
        
        my_2tank.receive_action(u)
        my_ctrl_opt_pred.receive_sys_state(my_2tank._x)
        my_ctrl_opt_pred.upd_icost(y, u)
        
        h1 = ksi[0]
        h2 = ksi[1]
        p = u
        
        r = my_ctrl_opt_pred.rcost(y, u)
        icost = my_ctrl_opt_pred.icost_val
        
        if is_print_sim_step:
            my_logger.print_sim_step(t, h1, h2, p, r, icost)
            
        if is_log_data:
            my_logger.log_data_row(datafile, t, h1, h2, p, r, icost)
        
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
                my_ctrl_opt_pred.reset(t0)
            else:
                pass
            
            icost = 0  