#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: nonlinear double-tank system

"""

import os, sys
PARENT_DIR = os.path.abspath(__file__ + '/../..')
sys.path.insert(0, PARENT_DIR)
import rcognita

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = f"this script is being run using " \
           f"rcognita ({rcognita.__version__}) " \
           f"located in cloned repository at '{PARENT_DIR}'. " \
           f"If you are willing to use your locally installed rcognita, " \
           f"remove this script ('{os.path.basename(__file__)}') from " \
           f"'rcognita/presets'."
else:
    info = f"this script is being run using " \
           f"locally installed rcognita ({rcognita.__version__}). " \
           f"Make sure the versions match."
print("INFO:", info)


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
import argparse


dim_state = 2
dim_input = 1
dim_output = 2
dim_disturb = 1


description = "Simulate a nonlinear double-tank system"
parser = argparse.ArgumentParser(description=description)
parser.add_argument('ctrl_mode', metavar='ctrl_mode', type=str,
                    help='control mode',
                    choices=['manual',
                             'nominal',
                             'MPC',
                             'RQL',
                             'SQL',
                             'JACS'])
parser.add_argument('dt', type=float, metavar='dt',
                    help='controller sampling time')
parser.add_argument('t1', type=float, metavar='t1',
                    help='final time')
parser.add_argument('x0', type=float, nargs="+", metavar='x0',
                    help='initial state (sequence of numbers); ' + 
                         'dimension preset-specific!')
parser.add_argument('--is_log_data', type=bool, default=False,
                    help='')
parser.add_argument('--is_visualization', type=bool, default=True,
                    help='')
parser.add_argument('--is_print_sim_step', type=bool, default=True,
                    help='')
parser.add_argument('--is_est_model', type=bool, default=False,
                    help='if a model of the env. is to be estimated online')
parser.add_argument('--model_est_stage', type=float, default=1.0,
                    help='seconds to learn model until benchmarking controller kicks in')
parser.add_argument('--model_est_period', type=float, default=None,
                    help='model is updated every model_est_period seconds')
parser.add_argument('--model_order', type=int, default=5,
                    help='order of state-space estimation model')
parser.add_argument('--prob_noise_pow', type=float, default=False,
                    help='power of probing noise')
parser.add_argument('--uMan', type=float, default=np.zeros(dim_input), nargs='+',
                    help='manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int, default=3,
                    help='horizon length (in steps) for predictive controllers')
parser.add_argument('--pred_step_size', type=float, default=None,
                    help='')
parser.add_argument('--buffer_size', type=int, default=10,
                    help='')
parser.add_argument('--rcost_struct', type=str,
                    default='quadratic', choices=['quadratic',
                                                  'biquadratic'],
                    help='structure of running cost function')
dim1 = 3
dim2 = 2
parser.add_argument('--R1', type=float, nargs='+',
                    default=np.eye(dim1),
                    help='must have proper dimension')
parser.add_argument('--R2', type=float, nargs='+',
                    default=np.eye(dim2),
                    help='must have proper dimension')
parser.add_argument('--Ncritic', type=int, default=4,
                    help='critic stack size (number of TDs)')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor')
parser.add_argument('--critic_period', type=float, default=None,
                    help='critic is updated every critic_period seconds')
parser.add_argument('--critic_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix'],
                    help='structure of critic features')
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix'],
                    help='structure of actor features')

args = parser.parse_args()
args.x0 = np.array(args.x0)
args.uMan = np.array(args.uMan)
if args.model_est_period is None:
    args.model_est_period = args.dt
if args.pred_step_size is None:
    args.pred_step_size = args.dt
if isinstance(args.R1, list):
    assert len(args.R1) == dim1 ** 2
    args.R1 = np.array(args.R1).reshape(dim1, dim1)
if isinstance(args.R2, list):
    assert len(args.R2) == dim2 ** 2
    args.R2 = np.array(args.R2).reshape(dim2, dim2)
if args.critic_period is None:
    args.critic_period = args.dt

assert args.t1 > args.dt > 0.0
assert args.x0.size == dim_state




globals().update(vars(args))





#------------------------------------user settings : : main switches


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
# ctrl_mode = 'MPC'

#------------------------------------user settings : : system
# System


# System parameters
tau1 = 18.4
tau2 = 24.4
K1 = 1.3
K2 = 1
K3 = 0.2

#------------------------------------user settings : : simulation
t0 = 0
Nruns = 1

# state_init = np.ones(dim_state)
state_init = x0

action_init = 0.5 * np.ones(dim_input)

disturb_init = 0 * np.ones(dim_disturb)

# Solver
atol = 1e-5
rtol = 1e-3

#------------------------------------user settings : : digital elements
# Digital elements sampling time
#dt = 0.1 # [s], controller sampling time
# sampleFreq = 1/dt # [Hz]

# Parameters
# cutoff = 1 # [Hz]

# Digital differentiator filter order
# diffFiltOrd = 4

#------------------------------------user settings : : model estimator
# is_est_model = 0
# model_est_stage = 2 # [s]
# model_est_period = 1*dt # [s]

# model_order = 5

# prob_noise_pow = 8

# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

#------------------------------------user settings : : controller
# u[0]: Pushing force F [N]
# u[1]: Steering torque M [N m]

# Manual control
action_manual = uMan

# Control constraints
action_min = 0
action_max = 1

# Control horizon length
#Nactor = 5

# Should be a multiple of dt
#pred_step_size = 5*dt # [s]

# Size of data buffers (used, e.g., in model estimation and critic)
#buffer_size = 200

#------------------------------------user settings : : RL
# Running cost structure and parameters
# Notation: chi = [observation, u]
# 'quadratic'     - quadratic chi.T R1 chi 
# 'biquadratic'     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
# rcost_struct = 'quadratic'

#R1 = np.diag([10, 10, 0])  # No mixed terms, full-state measurement

# Target filling of the tanks
observation_target = np.array([0.5, 0.5])

# Critic stack size, not greater than buffer_size
# Ncritic = 50

# Discounting factor
# gamma = 1

# Critic is updated every critic_period seconds
# critic_period = 5*dt # [s]

# Actor and critic structure choice
# 'quad-lin' - quadratic-linear
# 'quadratic' - quadratic
# 'quad-nomix' - quadratic, no mixed terms
# 'quad-mix' - W[0] observation[0]^2 + ... W[p-1] observation[p-1]^2 + W[p] observation[0] u[0] + ... W[...] u[0]^2 + ... (only Q-function critic)
# critic_struct = 'quad-nomix'
# actor_struct = 'quad-nomix'

#------------------------------------initialization : : system
ctrl_bnds = np.array([[action_min], [action_max]]).T

my_2tank = systems.Sys2Tank(sys_type="diff_eqn",
                            dim_state=dim_state,
                            dim_input=dim_input,
                            dim_output=dim_output,
                            dim_disturb=dim_disturb,
                            pars=[tau1, tau2, K1, K2, K3],
                            ctrl_bnds=ctrl_bnds)

observation_init = my_2tank.out(state_init)

#------------------------------------initialization : : model

#------------------------------------initialization : : controller
my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input, dim_output,
                                            ctrl_mode,
                                            ctrl_bnds=ctrl_bnds,
                                            t0=t0,
                                            sampling_time=dt,
                                            Nactor=Nactor,
                                            pred_step_size=pred_step_size,
                                            sys_rhs=my_2tank._state_dyn,
                                            sys_out=my_2tank.out,
                                            state_sys=state_init,
                                            prob_noise_pow = prob_noise_pow,
                                            is_est_model=is_est_model,
                                            model_est_stage=model_est_stage,
                                            model_est_period=model_est_period,
                                            buffer_size=buffer_size,
                                            model_order=model_order,
                                            model_est_checks=model_est_checks,
                                            gamma=gamma,
                                            Ncritic=Ncritic,
                                            critic_period=critic_period,
                                            critic_struct=critic_struct,
                                            rcost_struct=rcost_struct,
                                            rcost_pars=[R1],
                                            observation_target=observation_target)

#------------------------------------initialization : : simulator
my_simulator = simulator.Simulator(sys_type="diff_eqn",
                                   closed_loop_rhs=my_2tank.closed_loop_rhs,
                                   sys_out=my_2tank.out,
                                   state_init=state_init,
                                   disturb_init=disturb_init,
                                   action_init=action_init,
                                   t0=t0,
                                   t1=t1,
                                   dt=dt,
                                   max_step=dt/2,
                                   first_step=1e-6,
                                   atol=atol,
                                   rtol=rtol,
                                   is_dyn_ctrl=is_dyn_ctrl)

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
    
    state_full_init = my_simulator.state_full
    
    my_animator = visuals.Animator2Tank(objects=(my_simulator,
                                                 my_2tank,
                                                 [],
                                                 my_ctrl_opt_pred,
                                                 datafiles,
                                                 controllers.ctrl_selector,
                                                 my_logger),
                                           pars=(state_init,
                                                 action_init,
                                                 t0,
                                                 t1,
                                                 state_full_init,
                                                 ctrl_mode,
                                                 action_manual,
                                                 action_min,
                                                 action_max,
                                                 Nruns,
                                                 is_print_sim_step, is_log_data, 0, [], observation_target))

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
        
        t, state, observation, state_full = my_simulator.get_sim_step_data()
        
        u = controllers.ctrl_selector(t, observation, action_manual, [], my_ctrl_opt_pred, ctrl_mode)
        
        my_2tank.receive_action(u)
        my_ctrl_opt_pred.receive_sys_state(my_2tank._state)
        my_ctrl_opt_pred.upd_icost(observation, u)
        
        h1 = state_full[0]
        h2 = state_full[1]
        p = u
        
        r = my_ctrl_opt_pred.rcost(observation, u)
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
            my_simulator.observation = state_full_init
            
            if ctrl_mode > 0:
                my_ctrl_opt_pred.reset(t0)
            else:
                pass
            
            icost = 0  