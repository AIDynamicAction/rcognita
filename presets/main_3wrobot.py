#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: 3-wheel robot with dynamic actuators

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



dim_state = 5
dim_input = 2
dim_output = 5
dim_disturb = 2

description = "Simulate a 3-wheel robot with dynamic actuators"
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
dim1 = 7
dim2 = 5
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
# is_log_data = 0
# is_visualization = 1
# is_print_sim_step = 1

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
# ctrl_mode = 'nominal'

#------------------------------------user settings : : digital elements
# Digital elements sampling time
# dt = 0.01 # [s], controller sampling time

#------------------------------------user settings : : system
# System


# System parameters
m = 10 # [kg]
I = 1 # [kg m^2]

#------------------------------------user settings : : simulation
t0 = 0
# t1 = 40
Nruns = 1

state_init = x0


action_init = 0 * np.ones(dim_input)

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
#is_est_model = 0
#model_est_stage = 2 # [s]
#model_est_period = 1*dt # [s]

#model_order = 5

#prob_noise_pow = 8

# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

#------------------------------------user settings : : controller
# action[0]: Pushing force F [N]
# action[1]: Steering torque M [N m]

# Manual control
Fman = -3
Nman = -1
action_manual = uMan

# Control constraints
Fmin = -300
Fmax = 300
Mmin = -100
Mmax = 100

# Control horizon length
#Nactor = 6

# Should be a multiple of dt
#pred_step_size = 5*dt # [s]

# Size of data buffers (used, e.g., in model estimation and critic)
#buffer_size = 4 # 200 -- used for predictive RL

#------------------------------------user settings : : RL
# Running cost structure and parameters
# Notation: chi = [observation, action]
# 'quadratic'     - quadratic chi.T R1 chi 
# 'biquadratic'     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
#rcost_struct = 'quadratic'

#R1 = np.diag([10, 10, 1, 0, 0, 1, 1])  # No mixed terms, full-state measurement
# R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
# R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in observation
# R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
#R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in observation
# R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# Critic stack size, not greater than buffer_size
#Ncritic = 4 # 50 -- used for predictive RL

# Discounting factor
# gamma = 1

# Critic is updated every critic_period seconds
# critic_period = 5*dt # [s]

# Actor and critic structure choice
# 'quad-lin' - quadratic-linear
# 'quadratic' - quadratic
# 'quad-nomix' - quadratic, no mixed terms
# 'quad-mix' - w_critic[0] observation[0]^2 + ... w_critic[p-1] observation[p-1]^2 + w_critic[p] observation[0] action[0] + ... w_critic[...] action[0]^2 + ... (only Q-function critic)
# critic_struct = 'quad-nomix'
# actor_struct = 'quad-nomix'

#------------------------------------initialization : : system
my_3wrobot = systems.Sys3WRobot(sys_type="diff_eqn",
                                dim_state=dim_state,
                                dim_input=dim_input,
                                dim_output=dim_output,
                                dim_disturb=dim_disturb,
                                pars=[m, I],
                                ctrl_bnds=np.array([[Fmin, Fmax], [Mmin, Mmax]]),
                                is_dyn_ctrl=is_dyn_ctrl,
                                is_disturb=is_disturb,
                                pars_disturb=[])

observation_init = my_3wrobot.out(state_init)

xCoord0 = state_init[0]
yCoord0 = state_init[1]
alpha0 = state_init[2]
alpha_deg_0 = alpha0/2/np.pi

#------------------------------------initialization : : model

# Euler scheme
# get_next_state = lambda state: state + dt * self.sys_rhs([], state, myU[k-1, :], [])
# sys_out = my_3wrobot.out

# If is_use_offline_est_model
# if model_type == 1 # SS
# get_next_state = lambda state: my_NN.get_next(state, ...dt)
# ... 2 # NN

#------------------------------------initialization : : controller
ctrl_bnds = np.array([[Fmin, Fmax], [Mmin, Mmax]])

my_ctrl_nominal_3wrobot = controllers.CtrlNominal3WRobot(m, I, ctrl_gain=0.5, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)

# Predictive optimal controller
my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input,
                                           dim_output,
                                           ctrl_mode,
                                           ctrl_bnds=ctrl_bnds,
                                           t0=t0,
                                           sampling_time=dt,
                                           Nactor=Nactor,
                                           pred_step_size=pred_step_size,
                                           sys_rhs=my_3wrobot._state_dyn,
                                           sys_out=my_3wrobot.out,
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
                                           observation_target=[])

# Stabilizing RL agent
my_ctrl_RL_stab = controllers.CtrlRLStab(dim_input,
                                         dim_output,
                                         ctrl_mode,
                                         ctrl_bnds=ctrl_bnds,
                                         t0=t0,
                                         sampling_time=dt,
                                         Nactor=Nactor,
                                         pred_step_size=pred_step_size,
                                         sys_rhs=my_3wrobot._state_dyn,
                                         sys_out=my_3wrobot.out,
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
                                         actor_struct=actor_struct,
                                         rcost_struct=rcost_struct,
                                         rcost_pars=[R1],
                                         observation_target=[],
                                         safe_ctrl=my_ctrl_nominal_3wrobot,
                                         safe_decay_rate=1e-4)

if ctrl_mode == 'JACS':
    my_ctrl_benchm = my_ctrl_RL_stab
else:
    my_ctrl_benchm = my_ctrl_opt_pred
    
#------------------------------------initialization : : simulator
my_simulator = simulator.Simulator(sys_type="diff_eqn",
                                   closed_loop_rhs=my_3wrobot.closed_loop_rhs,
                                   sys_out=my_3wrobot.out,
                                   state_init=state_init,
                                   disturb_init=[],
                                   action_init=action_init,
                                   t0=t0,
                                   t1=t1,
                                   dt=dt,
                                   max_step=dt/2,
                                   first_step=1e-6,
                                   atol=atol,
                                   rtol=rtol,
                                   is_disturb=is_disturb,
                                   is_dyn_ctrl=is_dyn_ctrl)
#------------------------------------initialization : : logger
data_folder = 'data'

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns
for k in range(0, Nruns):
    datafiles[k] = data_folder + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
    
    if is_log_data:
        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt', 'F [N]', 'M [N m]'] )

# Do not display annoying warnings when print is on
if is_print_sim_step:
    warnings.filterwarnings('ignore')
    
my_logger = loggers.Logger3WRobot()

#------------------------------------main loop
if is_visualization:
    
    state_full_init = my_simulator.state_full
    
    my_animator = visuals.Animator3WRobot(objects=(my_simulator,
                                                   my_3wrobot,
                                                   my_ctrl_nominal_3wrobot,
                                                   my_ctrl_benchm,
                                                   datafiles,
                                                   controllers.ctrl_selector,
                                                   my_logger),
                                           pars=(state_init,
                                                 action_init,
                                                 t0,
                                                 t1,
                                                 state_full_init,
                                                 xMin,
                                                 xMax,
                                                 yMin,
                                                 yMax,
                                                 ctrl_mode,
                                                 action_manual,
                                                 Fmin,
                                                 Mmin,
                                                 Fmax,
                                                 Mmax,
                                                 Nruns,
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
        
        t, state, observation, state_full = my_simulator.get_sim_step_data()
        
        action = controllers.ctrl_selector(t, observation, action_manual, my_ctrl_nominal_3wrobot, my_ctrl_benchm, ctrl_mode)
        
        my_3wrobot.receive_action(action)
        my_ctrl_benchm.receive_sys_state(my_3wrobot._state)
        my_ctrl_benchm.upd_icost(observation, action)
        
        xCoord = state_full[0]
        yCoord = state_full[1]
        alpha = state_full[2]
        v = state_full[3]
        omega = state_full[4]
        
        r = my_ctrl_benchm.rcost(observation, action)
        icost = my_ctrl_benchm.icost_val
        
        if is_print_sim_step:
            my_logger.print_sim_step(t, xCoord, yCoord, alpha, v, omega, r, icost, action)
            
        if is_log_data:
            my_logger.log_data_row(datafile, t, xCoord, yCoord, alpha, v, omega, r, icost, action)
        
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
                my_ctrl_benchm.reset(t0)
            else:
                my_ctrl_nominal_3wrobot.reset(t0)
            
            icost = 0  