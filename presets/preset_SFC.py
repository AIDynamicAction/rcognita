"""
Preset: SFC economic model
3-wheel robot with dynamical actuators.
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
#Check dimension
dim_state = 27
dim_input = 1
dim_output = dim_state
dim_disturb = 0


#Initial values
Y_1 = 100
C_1 = 60
I_1= 25
G_1= 15
BD_1 = 45
B_1 = 0
BP_1 =  0.979955
BT_1 = 0
DIV_1 = 20
DIVe_1 = 13.33  
DIVh_1 = 6.66
Vg_1 = 0
E_1 = 3
Ee_1 = 2
Eh_1 = 1
g_1 = 0.0625
Hh_1 = 9.54858
Hb_1 = 2.250225
K_2 = K_1 = 400
L_2 = L_1 = 100
pe_1 = 35
rl_1= 0.02
r_1 = 0.02
rb_1= 0.02
TB_1= 0.393063 
TCB_1 = 0.176982075  
T_1 = 7.47687
UP_1 = 23.6813
Vh_1 = 89.54858  
YHSh_1 = 67.2918
YDh_1 =  67.2918
W_1= 67.652
H_1= 11.798805 
RF_1= 11.798805
pb_1= 50
v0= 0.22382378 
v1= 0.2
v2= 0.2
f0 = 0.09826265506
f1 = 0.2
f2 = 0.6
m2b =  0.005 
#start with interest rate equal to 2.
ib_1 = 0.02

Ve_1=K_1+pe_1*Ee_1-L_1-pe_1*E_1
Vb_1=K_1-Vh_1-Ve_1-Vg_1
CGh_1 = YHSh_1-YDh_1
id_1 = ib_1-m2b
re_1 =pb_1*B_1/(Vh_1)-v0-v1*rb_1+v2*id_1

#from equation 15 
ree_1=(pe_1*Ee_1/(pe_1*Ee_1+K_1)-f0-f2*(UP_1/K_2))/f1


initial_state = [G_1,Y_1,C_1,I_1,B_1, YDh_1,W_1,T_1,CGh_1, YHSh_1,Vg_1,
                         Eh_1,Vh_1,re_1,pe_1,BD_1,K_1,Ee_1, 
            ree_1, L_1, UP_1, E_1, Ve_1, BT_1, RF_1, L_2, K_2]


description = "Agent-environment preset: SFC economic model."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['manual',
                             'nominal',
                             'MPC'],
                             
                    default='manual',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                    
                    '----MPC:model-predictive control; ')


parser.add_argument('--action_manual', type=float,
                    default=[0.02], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')

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



parser.add_argument('--dt', type=float, metavar='dt',
                    default=1,
                    help='Controller sampling time.' )

parser.add_argument('--t1', type=float, metavar='t1',
                    default=10.0,
                    help='Final time of episode.' )
parser.add_argument('--pred_step_size_multiplier', type=float,
                    default=1.0,
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
                    default=[1, 1, 1],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')

parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')

parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=initial_state,
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')

parser.add_argument('--Nactor', type=int,
                    default=5,
                    help='Horizon length (in steps) for predictive controllers.')

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

parser.add_argument('--is_log_data', type=bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')

parser.add_argument('--is_visualization', type=bool,
                    default=True,
                    help='Flag to produce graphical output.')

parser.add_argument('--is_playback', type=bool,
                    default=False,
                    help='Flag to playback') 

parser.add_argument('--stage_obj_init', type=float,
                    default=0.0,
                    help='The initial value of stage objective to plot') 

args = parser.parse_args()

#----------------------------------------Post-processing of arguments
# Convert `pi` to a number pi
# for k in range(len(args.state_init)):
#     args.state_init[k] = eval( args.state_init[k].replace('pi', str(np.pi)) )

args.state_init = np.array(args.state_init)

args.action_manual = np.array(args.action_manual)

assert args.state_init.size == dim_state

globals().update(vars(args))

#----------------------------------------(So far) fixed settings
is_disturb = 0
is_dyn_ctrl = 0
# Control constraints
ctrl_bnds = np.array([0, 1]).reshape(1,-1)
#ctrl_bnds = [[0,1]]
t0 = 0
atol = 1e-5
rtol = 1e-3
# Model estimator stores models in a stack and recall the best of model_est_checks
model_est_checks = 0


action_init = 0.01 * np.ones(dim_input)

# Parametres for model predictive control
pred_step_size = args.dt * args.pred_step_size_multiplier

model_est_period = args.dt * args.model_est_period_multiplier
critic_period = args.dt * args.critic_period_multiplier
#Parametre for loss fucntion
R1 = np.diag(np.array(args.R1_diag))

# Output and inflation objectives
observation_target = np.array([0.01, 0.03])



# System parameters


#----------------------------------------Initialization : : system
#``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
my_sys_eco = systems.SFC_System(sys_type="discr_fnc",
                                     dim_state=dim_state,
                                     dim_input=dim_input,
                                     dim_output=1,
                                     dim_disturb=0,
                                     ctrl_bnds = ctrl_bnds,
                                     is_dyn_ctrl=is_dyn_ctrl,
                                     is_disturb=is_disturb,
                                     pars_disturb=[])

observation_init = my_sys_eco.out(args.state_init)

##init 

#----------------------------------------Initialization : : model

#----------------------------------------Initialization : : controller
#my_ctrl_nominal = controllers.CtrlNominal3WRobot(m, I, ctrl_gain=5, ctrl_bnds=ctrl_bnds, t0=t0, sampling_time=dt)
my_ctrl_nominal=[]
# Predictive optimal controller
#lambda function 
#state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])
#lambda tmp_time, state, action: my_sys_eco._state_dyn(tmp_time, state, action)-state
print('----------------------------------------Initialization : : controllers.CtrlOptPred')
my_ctrl_opt_pred = controllers.CtrlOptPred(dim_input,
                                           dim_output,
                                           ctrl_mode,
                                           ctrl_bnds = ctrl_bnds,
                                           action_init = action_init,
                                           t0 = t0,
                                           sampling_time = dt,
                                           Nactor = Nactor,
                                           pred_step_size = pred_step_size,
                                           sys_rhs = lambda tmp_time, state, action: my_sys_eco._state_dyn(tmp_time, 
                                                                state, action)-state,
                                           sys_out = my_sys_eco.out,
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
print('----------------------------------------Initialization : : simulator')
my_simulator = simulator.Simulator(sys_type = "discr_fnc",
                                   closed_loop_rhs = my_sys_eco.closed_loop_rhs,
                                   sys_out = my_sys_eco.out,
                                   state_init = args.state_init,
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
print('----------------------------------------Initialization : : logger')
if os.path.basename(os.path.normpath( os.path.abspath(os.getcwd()) ) ) == 'presets':
    data_folder = '../simdata'
else:
    data_folder = 'simdata'

pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True) 

date = datetime.now().strftime("%Y-%m-%d")
time = datetime.now().strftime("%Hh%Mm%Ss")
datafiles = [None] * Nruns
print('----------------------------------------datafiles')
for k in range(0, Nruns):
    datafiles[k] = data_folder + '/' + my_sys_eco.name + '__' + ctrl_mode + '__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
    # Logging if necessary
    if is_log_data:
        print('Logging data to:    ' + datafiles[k])
            
        with open(datafiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['System', my_sys_eco.name ] )
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
            writer.writerow(['Ncritic', str(Ncritic) ] )
            writer.writerow(['gamma', str(gamma) ] )
            writer.writerow(['critic_period_multiplier', str(critic_period_multiplier) ] )
            writer.writerow(['critic_struct', str(critic_struct) ] )
            writer.writerow(['actor_struct', str(actor_struct) ] )          
            writer.writerow(['t [s]', 'h1', 'h2', 'p', 'stage_obj', 'accum_obj'] )

            
        
# Do not display annoying warnings when print is on

if is_print_sim_step:
    warnings.filterwarnings('ignore')
    
my_logger = loggers.LoggerSFC()

#----------------------------------------Main loop
if is_visualization:
    
    action_min, action_max = 0, 1
    
    state_full_init = my_simulator.state_full
    #is_playback = False
    #stage_obj_init = 0


    
    my_animator = visuals.AnimatorSFC(objects=(my_simulator,
                                                     my_sys_eco,
                                                     my_ctrl_nominal,
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
                                                  is_print_sim_step, 
                                                  is_log_data, 
                                                  is_playback,
                                                   stage_obj_init))

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
    
    action = controllers.ctrl_selector(t, observation, action_manual, my_ctrl_nominal, 
    my_ctrl_benchm, ctrl_mode)
    

    my_sys_eco.receive_action(action)
    my_ctrl_benchm.receive_sys_state(my_sys_eco._state)
    my_ctrl_benchm.upd_accum_obj(observation, action)
    
    #For drawing main parametres     
    
    stage_obj = my_ctrl_benchm.stage_obj(observation, action)
    accum_obj = my_ctrl_benchm.accum_obj_val
    
    #printing state parametres 


    Y_output = state[1]
    Kapital = state[16]
    Labor = state[19]
    Investment = state[3]
    Consumption = state[2]
    Y_output, inflation = observation

    # Y_output, Labor, Investment, Consumption, inflation,  stage_obj, accum_obj)
    # ?? output growth rate??
    if is_print_sim_step:
        my_logger.print_sim_step(t, Y_output, Labor, Investment, Consumption, inflation, stage_obj, accum_obj, action)
        
    if is_log_data:
        my_logger.log_data_row(datafile, t, Y_output, Labor, Investment, Consumption, inflation, stage_obj, accum_obj, action)
        print('logging  log_data_row')
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
        
        # if ctrl_mode != 'nominal':
        #     my_ctrl_benchm.reset(t0)
        # else:
        #     my_ctrl_nominal.reset(t0)
        
        accum_obj = 0  