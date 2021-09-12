#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preset: nonlinear double-tank system.

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
           f"run this script ('{os.path.basename(__file__)}') outside " \
           f"'rcognita/presets'."
else:
    info = f"this script is being run using " \
           f"locally installed rcognita ({rcognita.__version__}). " \
           f"Make sure the versions match."
print("INFO:", info)

import pathlib
    
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

#----------------------------------------Set up dimensions
dim_state = 2
dim_input = 1
dim_output = dim_state
dim_disturb = 1

dim_R1 = dim_output + dim_input
dim_R2 = dim_R1

description = "Agent-environment preset: nonlinear double-tank system."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['manual',
                             'MPC',
                             'RQL',
                             'SQL'],
                    default='MPC',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; ' +
                    '----SQL: stacked Q-learning.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.1,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=100.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['2', '-2'],
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=bool,
                    default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=bool,
                    default=True,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--is_est_model', type=bool,
                    default=False,
                    help='Flag to estimate environment model.')
parser.add_argument('--model_est_stage', type=float,
                    default=1.0,
                    help='Seconds to learn model until benchmarking controller kicks in.')
parser.add_argument('--model_est_period_multiplier', type=float,
                    default=1,
                    help='Model is updated every model_est_period_multiplier times dt seconds.')
parser.add_argument('--model_order', type=int,
                    default=5,
                    help='Order of state-space estimation model.')
parser.add_argument('--prob_noise_pow', type=float,
                    default=False,
                    help='Power of probing (exploration) noise.')
parser.add_argument('--action_manual', type=float,
                    default=[0.5], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=10,
                    help='Horizon length (in steps) for predictive controllers.')
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=2.0,
                    help='Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.')
parser.add_argument('--buffer_size', type=int,
                    default=10,
                    help='Size of the buffer (experience replay) for model estimation, agent learning etc.')
parser.add_argument('--stage_obj_struct', type=str,
                    default='quadratic',
                    choices=['quadratic',
                             'biquadratic'],
                    help='Structure of stage objective function.')
parser.add_argument('--R1_diag', type=float, nargs='+',
                    default=[10, 10, 1],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[10, 10, 1],
                    help='Parameter of stage objective function . Must have proper dimension. ' + 
                    'Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, ' +
                    'where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--Ncritic', type=int,
                    default=4,
                    help='Critic stack size (number of temporal difference terms in critic cost).')
parser.add_argument('--gamma', type=float,
                    default=1.0,
                    help='Discount factor.')
parser.add_argument('--critic_period_multiplier', type=float,
                    default=1.0,
                    help='Critic is updated every critic_period_multiplier times dt seconds.')
parser.add_argument('--critic_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix',
                                                   'quad-mix'],
                    help='Feature structure (critic). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms; ' +
                    '----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).')
parser.add_argument('--actor_struct', type=str,
                    default='quad-nomix', choices=['quad-lin',
                                                   'quadratic',
                                                   'quad-nomix'],
                    help='Feature structure (actor). Currently available: ' +
                    '----quad-lin: quadratic-linear; ' +
                    '----quadratic: quadratic; ' +
                    '----quad-nomix: quadratic, no mixed terms.')

args = parser.parse_args()

#----------------------------------------Post-processing of arguments
# Convert `pi` to a number pi
for k in range(len(args.state_init)):
    args.state_init[k] = eval( args.state_init[k].replace('pi', str(np.pi)) )

args.state_init = np.array(args.state_init)
args.action_manual = np.array(args.action_manual)

pred_step_size = args.dt * args.pred_step_size_multiplier
model_est_period = args.dt * args.model_est_period_multiplier
critic_period = args.dt * args.critic_period_multiplier

R1 = np.diag(np.array(args.R1_diag))
R2 = np.diag(np.array(args.R2_diag))

assert args.t1 > args.dt > 0.0
assert args.state_init.size == dim_state

globals().update(vars(args))

#----------------------------------------(So far) fixed settings
is_disturb = 0
is_dyn_ctrl = 0

t0 = 0

action_init = 0.5 * np.ones(dim_input)

disturb_init = 0 * np.ones(dim_disturb)

# Solver
atol = 1e-5
rtol = 1e-3

# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0

# Control constraints
action_min = 0
action_max = 1
ctrl_bnds=np.array([[action_min], [action_max]]).T

# System parameters
tau1 = 18.4
tau2 = 24.4
K1 = 1.3
K2 = 1
K3 = 0.2

# Target filling of the tanks
observation_target = np.array([0.5, 0.5])

#----------------------------------------Initialization : : system
my_sys = systems.Sys2Tank(sys_type="diff_eqn",
                            dim_state=dim_state,
                            dim_input=dim_input,
                            dim_output=dim_output,
                            dim_disturb=dim_disturb,
                            pars=[tau1, tau2, K1, K2, K3],
                            ctrl_bnds=ctrl_bnds)

observation_init = my_sys.out(state_init)

#----------------------------------------Initialization : : model

#----------------------------------------Initialization : : controller
# Predictive optimal controller
my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input,
                                           dim_output,
                                           ctrl_mode,
                                           ctrl_bnds = ctrl_bnds,
                                           action_init = action_init,
                                           t0 = t0,
                                           sampling_time = dt,
                                           Nactor = Nactor,
                                           pred_step_size = pred_step_size,
                                           sys_rhs = my_sys._state_dyn,
                                           sys_out = my_sys.out,
                                           state_sys = state_init,
                                           prob_noise_pow = prob_noise_pow,
                                           is_est_model = is_est_model,
                                           model_est_stage = model_est_stage,
                                           model_est_period = model_est_period,
                                           buffer_size = buffer_size,
                                           model_order = model_order,
                                           model_est_checks = model_est_checks,
                                           gamma = gamma,
                                           Ncritic = Ncritic,
                                           critic_period = critic_period,
                                           critic_struct = critic_struct,
                                           stage_obj_struct = stage_obj_struct,
                                           stage_obj_pars = [R1],
                                           observation_target = observation_target)

my_ctrl_benchm = my_ctrl_opt_pred
    
#----------------------------------------Initialization : : simulator
my_simulator = simulator.Simulator(sys_type = "diff_eqn",
                                   closed_loop_rhs = my_sys.closed_loop_rhs,
                                   sys_out = my_sys.out,
                                   state_init = state_init,
                                   disturb_init = [],
                                   action_init = action_init,
                                   t0 = t0,
                                   t1 = t1,
                                   dt = dt,
                                   max_step = dt/2,
                                   first_step = 1e-6,
                                   atol = atol,
                                   rtol = rtol,
                                   is_disturb = is_disturb,
                                   is_dyn_ctrl = is_dyn_ctrl)

#----------------------------------------Initialization : : logger
if os.path.basename( os.path.normpath( os.path.abspath(os.getcwd()) ) ) == 'presets':
    data_folder = '../simdata'
else:
    data_folder = 'simdata'

pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True) 

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns

for k in range(0, Nruns):
    datafiles[k] = data_folder + '/' + my_sys.name + '__' + ctrl_mode + '__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
    
    if is_log_data:
        print('Logging data to:    ' + datafiles[k])
            
        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['System', my_sys.name ] )
            writer.writerow(['Controller', ctrl_mode ] )
            writer.writerow(['dt', str(dt) ] )
            writer.writerow(['state_init', str(state_init) ] )
            writer.writerow(['is_est_model', str(is_est_model) ] )
            writer.writerow(['model_est_stage', str(model_est_stage) ] )
            writer.writerow(['model_est_period_multiplier', str(model_est_period_multiplier) ] )
            writer.writerow(['model_order', str(model_order) ] )
            writer.writerow(['prob_noise_pow', str(prob_noise_pow) ] )
            writer.writerow(['Nactor', str(Nactor) ] )
            writer.writerow(['pred_step_size_multiplier', str(pred_step_size_multiplier) ] )
            writer.writerow(['buffer_size', str(buffer_size) ] )
            writer.writerow(['stage_obj_struct', str(stage_obj_struct) ] )
            writer.writerow(['R1_diag', str(R1_diag) ] )
            writer.writerow(['R2_diag', str(R2_diag) ] )
            writer.writerow(['Ncritic', str(Ncritic) ] )
            writer.writerow(['gamma', str(gamma) ] )
            writer.writerow(['critic_period_multiplier', str(critic_period_multiplier) ] )
            writer.writerow(['critic_struct', str(critic_struct) ] )
            writer.writerow(['actor_struct', str(actor_struct) ] )          
            writer.writerow(['t [s]', 'h1', 'h2', 'p', 'stage_obj', 'accum_obj'] )

# Do not display annoying warnings when print is on
if is_print_sim_step:
    warnings.filterwarnings('ignore')
    
my_logger = loggers.Logger2Tank()

#----------------------------------------Main loop
if is_visualization:
    
    state_full_init = my_simulator.state_full
    
    my_animator = visuals.Animator2Tank(objects=(my_simulator,
                                                my_sys,
                                                [],
                                                my_ctrl_benchm,
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
    
    my_animator.get_anm(anm)
    
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
        
        action = controllers.ctrl_selector(t, observation, action_manual, [], my_ctrl_benchm, ctrl_mode)
        
        my_sys.receive_action(action)
        my_ctrl_benchm.receive_sys_state(my_sys._state)
        my_ctrl_benchm.upd_accum_obj(observation, action)
        
        h1 = state_full[0]
        h2 = state_full[1]
        p = action       
        
        stage_obj = my_ctrl_benchm.stage_obj(observation, action)
        accum_obj = my_ctrl_benchm.accum_obj_val
        
        if is_print_sim_step:
            my_logger.print_sim_step(t, h1, h2, p, stage_obj, accum_obj)
            
        if is_log_data:
            my_logger.log_data_row(datafile, t, h1, h2, p, stage_obj, accum_obj)
        
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
            
            if ctrl_mode != 'nominal':
                my_ctrl_benchm.reset(t0)
            else:
                my_ctrl_nominal.reset(t0)
            
            accum_obj = 0  