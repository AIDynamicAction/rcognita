import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

parser = argparse.ArgumentParser(description=description)

parser.add_argument('--ctrl_mode', metavar='ctrl_mode', type=str,
                    choices=['manual',
                             'nominal',
                             'MPC',
                             'RQL',
                             'SQL',
                             'JACS'],
                    default='nominal',
                    help='Control mode. Currently available: ' +
                    '----manual: manual constant control specified by action_manual; ' +
                    '----nominal: nominal controller, usually used to benchmark optimal controllers;' +                     
                    '----MPC:model-predictive control; ' +
                    '----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; ' +
                    '----SQL: stacked Q-learning; ' + 
                    '----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.')
parser.add_argument('--dt', type=float, metavar='dt',
                    default=0.01,
                    help='Controller sampling time.' )
parser.add_argument('--t1', type=float, metavar='t1',
                    default=6.0,
                    help='Final time of episode.' )
parser.add_argument('--Nruns', type=int,
                    default=1,
                    help='Number of episodes. Learned parameters are not reset after an episode.')
parser.add_argument('--state_init', type=str, nargs="+", metavar='state_init',
                    default=['5', '5', '-5*pi/4'],
                    help='Initial state (as sequence of numbers); ' + 
                    'dimension is environment-specific!')
parser.add_argument('--is_log_data', type=str2bool,
                    default=False,
                    help='Flag to log data into a data file. Data are stored in simdata folder.')
parser.add_argument('--is_visualization', type=str2bool, nargs='?',
                        const=True, default=True,
                    help='Flag to produce graphical output.')
parser.add_argument('--is_print_sim_step', type=str2bool,
                    default=True,
                    help='Flag to print simulation data into terminal.')
parser.add_argument('--is_est_model', type=str2bool,
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
                    default=[-5, -3], nargs='+',
                    help='Manual control action to be fed constant, system-specific!')
parser.add_argument('--Nactor', type=int,
                    default=3,
                    help='Horizon length (in steps) for predictive controllers.')
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
                    default=[1, 10, 1, 0, 0],
                    help='Parameter of stage objective function. Must have proper dimension. ' +
                    'Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.')
parser.add_argument('--R2_diag', type=float, nargs='+',
                    default=[1, 10, 1, 0, 0],
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
