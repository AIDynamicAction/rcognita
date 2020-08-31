# general imports
import os
import pathlib
import warnings
import sys
from collections import namedtuple
import copy
import itertools
import statistics

# scipy
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import basinhopping

# numpy
import numpy as np
from numpy.random import rand
from numpy.random import randn
import numpy.linalg as la
from scipy import signal
import sippy  # Github:CPCLAB-UNIPI/SIPPY

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# rcognita
from . import utilities

# other
from mpldatacursor import datacursor
from tabulate import tabulate


class System(utilities.Generic):
    """
    Class of continuous-time dynamical systems with input and dynamical disturbance for use with ODE solvers.

    In RL, this is considered the *environment*.
    Normally, you should pass `closed_loop`, which represents the right-hand side, to your solver.

    Parameters
    ----------
    dim_state : int
        dimension of state vector
        x_t = [x_c, y_c, alpha, upsilon, omega]

    dim_input : int
        * dimension of action vector
        * u_t = [F, M]

    dim_output : int
        * dimension of output vector
        * x_t+1 = [x_c, y_c, alpha, upsilon, omega]

    dim_disturb : int
        * dimension of disturbance vector
        * actuator disturbance that gets added to F and M

    initial_x : int
        * initial x coordinate of robot

    initial_y : int
        * initial x coordinate of robot

    m : int
        * m = robot's mass

    I : int
        * I = moment of inertia about the vertical axis

    f_min, f_max, m_min, m_max : all int
        * control bounds

    f_man : int

    n_man: int

    is_dyn_ctrl : int
        * is dynamic control?
        * If 1, the controller (a.k.a. agent) is considered as a part of the full state vector

    is_disturb : int
        * use disturbance?
        * If 0, no disturbance is fed into the system

    sigma_q, mu_q, tau_q : int
        * hyperparameters to disturbance
        * Parameters of the disturbance model


    Attributes
    ----------

    alpha : float
        * turning angle

    num_controllers : int
        * number of controllers the environment will interact with

    system_state
        * state of the environment
        * can be a vector or matrix (if there are multiple sub-states, i.e. for multiple controllers)

    u0 : float vector
        * control input vector

    q0 : float vector
        * disturbance vector

    _dim_initial_full_state : int vector
        * dimensions of full state

    full_state : int vector
        * includes the system state vector, control input vector and disturbance vector
        * can be a vector or matrix (if there are multiple sub-states, i.e. for multiple controllers)

    num_controllers : int
        * number of controllers to be used with the environment

    multi_sim : int
        * variable for the closed_loop function
        * specifies that the closed_loop is being executed for a specific controller


    """

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 dim_disturb=2,
                 initial_x=5,
                 initial_y=5,
                 m=10,
                 I=1,
                 f_man=-3,
                 n_man=-1,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 is_dyn_ctrl=0,
                 is_disturb=0,
                 sigma_q=None,
                 mu_q=None,
                 tau_q=None):

        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb

        if is_dyn_ctrl:
            self._dim_initial_full_state = self.dim_state + self.dim_disturb + self.dim_input
        else:
            self._dim_initial_full_state = self.dim_state + self.dim_disturb

        self.m = m
        self.I = I
        self.f_min = f_min
        self.f_max = f_max
        self.m_min = m_min
        self.m_max = m_max
        self.f_man = f_man
        self.n_man = n_man
        self.u_man = np.array([f_man, n_man])
        self.control_bounds = np.array([[f_min, f_max], [m_min, m_max]])
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.num_controllers = 1
        self.q0 = np.zeros(dim_disturb)
        self.u0 = np.zeros(dim_input)
        self.is_dyn_ctrl = is_dyn_ctrl
        self.multi_sim = None

        # initial values of the system's state
        self.initial_alpha = self.alpha = initial_alpha = np.pi / 2
        initial_state = np.zeros(dim_state)
        initial_state[0] = initial_x
        initial_state[1] = initial_y
        initial_state[2] = initial_alpha
        self.system_state = initial_state

        self.full_state = self.create_full_state(
            self.system_state, self.q0, self.u0, is_dyn_ctrl)

        """ disturbance """
        self.is_disturb = is_disturb
        self.sigma_q = sigma_q
        self.mu_q = mu_q
        self.tau_q = tau_q

    def create_full_state(self, system_state, u0, q0=None, is_dyn_ctrl=0):
        if is_dyn_ctrl:
            self.full_state = np.concatenate([self.system_state, u0, q0])
        else:
            self.full_state = np.concatenate([self.system_state, u0])

        return self.full_state

    def add_bots(self, initial_x, initial_y, number=1):
        self.new_state = np.zeros(self.dim_state)
        self.new_state[0] = initial_x
        self.new_state[1] = initial_y
        self.new_state[2] = self.initial_alpha

        self.system_state = np.vstack((self.system_state, self.new_state))

        if self.u0.ndim == 1:
            self.u0 = np.tile(self.u0, (2, 1))

        elif self.u0.ndim > 1:
            new_row = self.u0[0, :]
            self.u0 = np.vstack((self.u0, new_row))

        if self.q0.ndim == 1:
            self.q0 = np.tile(self.q0, (2, 1))

        elif self.q0.ndim > 1:
            new_row = self.u0[0, :]
            self.q0 = np.vstack((self.q0, new_row))

        if self.is_dyn_ctrl:
            self.full_state = np.concatenate(
                (self.system_state, self.q0, self.u0), axis=1)
        else:
            self.full_state = np.concatenate(
                (self.system_state, self.q0), axis=1)

        self.alpha = self.system_state[:, 2]

        self.num_controllers += 1

    @staticmethod
    def get_system_dynamics(t, x, u, q, m, I, dim_state, is_disturb):
        """ get system internal dynamics

            Generalized derivative of: x_t+1 = f(x_t, u_t, q_t)

        where:
            x -- state
            u -- input
            q -- disturbance


        System description
        ------------------

        Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

        Variables:
            * x_с -- x-coordinate [m]
            * y_с -- y-coordinate [m]
            * \\alpha : turning angle [rad]
            * v -- speed [m/s]
            * \\omega : revolution speed [rad/s]
            * F -- pushing force [N]
            * M -- steering torque [Nm]
            * m -- robot mass [kg]
            * I -- robot moment of inertia around vertical axis [kg m^2]
            * q -- actuator disturbance (see System._add_disturbance). Is zero if is_disturb = 0

            x = [x_c, y_c, \\alpha, v, \\omega]`
            u = [F, M]
            pars = [m, I]

        References
        ----------
        .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
            nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

        """

        # define vars
        F = u[0]
        M = u[1]
        alpha = x[2]
        v = x[3]
        omega = x[4]

        # create state
        system_dynamics = np.zeros(dim_state)

        # compute new values
        x = v * np.cos(alpha)
        y = v * np.sin(alpha)
        alpha = omega

        if is_disturb:
            v = 1 / m * (F + q[0])
            omega = 1 / I * (M + q[1])
        else:
            v = 1 / m * F
            omega = 1 / I * M

        # assign next state
        system_dynamics[0] = x
        system_dynamics[1] = y
        system_dynamics[2] = alpha
        system_dynamics[3] = v
        system_dynamics[4] = omega

        return system_dynamics

    @staticmethod
    def get_curr_state(x, u=[]):
        """ Return current state of system """
        y = x
        return y

    def _add_disturbance(self, t, q):
        """ Dynamical disturbance model """

        Dq = np.zeros(self.dim_disturb)

        if self.is_disturb:
            for k in range(0, self.dim_disturb):
                Dq[k] = - tau_q[k] * (q[k] + sigma_q[k] * (randn() + mu_q[k]))

        return Dq

    def _create_dyn_controller(t, u, y):
        """
        Dynamical controller.

        When `is_dyn_ctrl=0`, the controller is considered static, which is to say that the control actions are computed immediately from the system's output.

        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.

        Currently, left for future implementation

        """

        Du = np.zeros(self.dim_input)

        return Du

    def set_latest_action(self, u, mid=None):
        if self.num_controllers > 1:
            self.u0[mid] = u
        else:
            self.u0 = u

    def set_multi_sim(self, mid):
        self.multi_sim = mid

    def closed_loop(self, t, full_state):
        """
        Closed loop of the system.
        This function is designed for use with ODE solvers.
        Normally, you shouldn't change it

        Examples
        --------
        Assuming `sys` is a `system`-object, `t0, t1` - start and stop times, and `full_state` - a properly defined initial condition:

        >>> import scipy as sp
        >>> simulator = sp.integrate.RK45(sys.closed_loop, t0, full_state, t1)
        >>> while t < t1:
                simulator.step()
                t = simulator.t
                full_state = simulator.y
                x = full_state[0:sys.dim_state]
                y = sys.get_curr_state(x)
                u = myController(y)
                sys.set_latest_action(u)

        """
        # environment + disturbance
        if self.multi_sim is not None:
            mid = self.multi_sim

        new_full_state = np.zeros(self._dim_initial_full_state)

        x = full_state[0:self.dim_state]
        q = full_state[self.dim_state:]

        if self.is_dyn_ctrl:
            u = full_state[-self.dim_input:]
            new_full_state[-self.dim_input:
                           ] = self._create_dyn_controller(t, u, y)
        else:
            # Fetch the control action stored in the system
            if self.multi_sim is not None:
                u = self.u0[mid]
            else:
                u = self.u0

        if self.control_bounds.any():
            for k in range(self.dim_input):
                u[k] = np.clip(u[k], self.control_bounds[k, 0],
                               self.control_bounds[k, 1])

        new_full_state[0:self.dim_state] = self.get_system_dynamics(
            t, x, u, q, self.m, self.I, self.dim_state, self.is_disturb)

        if self.is_disturb:
            new_full_state[self.dim_state:] = self._add_disturbance(t, q)

        if self.multi_sim is not None:
            self.system_state[mid, :] = x
        else:
            self.system_state = x

        return new_full_state

class Controller(utilities.Generic):
    """
    Optimal controller (a.k.a. agent) class.

    Parameters and descriptions of instance attributes
    --------------------------------------------------

    system : object of type `System` class
        object of type System (class)

    initial_x : int
        * starting x coordinate of controller

    initial_y : int
        * starting y coordinate of controller

    t0 : int
        * Initial value of the controller's internal clock

    t1 : int
        * End value of controller's internal clock

    n_actor : int
        Number of prediction steps. n_actor=1 means the controller is purely data-driven and doesn't use prediction.

    n_critic : int
        Critic stack size. The critic optimizes the temporal error, a.k.a. the value (of state) function. The temporal errors are stacked up using the said buffer.

    buffer_size : int
        The size of the buffer to store data for model estimation. The bigger the buffer, the more accurate the estimation may be achieved. Using a larger buffer results in better model estimation at the expense of computational cost.

    ctrl_mode : int
        Modes with online model estimation are experimental
        * 0     - manual constant control (only for basic testing)
        * -1    - nominal parking controller (for benchmarking optimal controllers)
        * 1     - model-predictive control (MPC). Prediction via discretized true model
        * 2     - adaptive MPC. Prediction via estimated model
        * 3     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via discretized true model
        * 4     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via estimated model
        * 5     - RL: stacked Q-learning. Prediction via discretized true model
        * 6     - RL: stacked Q-learning. Prediction via estimated model

        * Modes 1, 3, 5 use model for prediction, passed into class exogenously. This could be, for instance, a true system model
        * Modes 2, 4, 6 use an estimated online

    critic_mode : int
        Choice of the structure of the critic's feature vector
        * 1 - Quadratic-linear
        * 2 - Quadratic
        * 3 - Quadratic, no mixed terms
        * 4 - Quadratic, no mixed terms in input and output

    critic_update_time : float
        * Time between critic updates

    r_cost_struct : int
        * Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.
        * 1 - quadratic chi.T @ R1 @ chi
        * 2 - 4th order chi**2.T @ R2 @ chi**2 + chi.T @ R2 @ chi

    sample_time : int or float
        Controller's sampling time (in seconds). The system itself is continuous as a physical process while the controller is digital.
        * the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
        * smaller sampling times lead to higher computation times
        * especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with sample_time and try achieve a trade-off between stability and computational performance

    pred_step_size : float
        * Prediction step size in `J` (in seconds). Is the time between the computation of control inputs and outputs J. Should be a multiple of `sample_time`.

    estimator_buffer_fill : int
        * Initial phase to fill the estimator's buffer before applying optimal control (in seconds)

    estimator_buffer_power : int
        * Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control

    estimator_update_time : float
        * In seconds, the time between model estimate updates. This constant determines how often the estimated parameters are updated. The more often the model is updated, the higher the computational burden is. On the other hand, more frequent updates help keep the model actual.

    stacked_model_params : int
        * Estimated model parameters can be stored in stacks and the best among the `stacked_model_params` last ones is picked.
        * May improve the prediction quality somewhat

    model_order : int
        * The order of the state-space estimation model. We are interested in adequate predictions of y under given u's. The higher the model order, the better estimation results may be achieved, but be aware of overfitting.

    gamma : float
        * Discounting factor
        * number in (0, 1]
        * Characterizes fading of running costs along horizon

    Attributes
    ----------

    A, B, C, D : float vectors
        * vectors denoting model parameters

    my_model : object of type `_model` class

    R1, R2 : float vectors
        * running cost parameters

    u_min, u_max : float vectors
        * denoting the min and max control action values

    u_buffer : float vector
        * buffer of previous controls

    y_buffer : float vector
        * buffer of previous outputs

    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155

    """

    def __init__(self,
                 system,
                 t0=0,
                 t1=15,
                 n_actor=6,
                 n_critic=50,
                 buffer_size=50,
                 ctrl_mode=3,
                 critic_mode=1,
                 critic_update_time=0.1,
                 r_cost_struct=1,
                 sample_time=0.1,
                 estimator_update_time=0.1,
                 estimator_buffer_fill=6,
                 estimator_buffer_power=2,
                 stacked_model_params=0,
                 pred_step_size=2,
                 model_order=3,
                 gamma=1):
        """

        SYSTEM-RELATED ATTRIBUTES

        """
        self.dim_state = system.dim_state
        self.dim_input = system.dim_input
        self.dim_output = system.dim_output
        self.m = system.m
        self.I = system.I
        self.is_disturb = system.is_disturb
        self.system_state = system.system_state
        self.ctrl_bnds = system.control_bounds
        self.sys_rhs = system.get_system_dynamics
        self.sys_out = system.get_curr_state

        """

        CONTROLLER-RELATED ATTRIBUTES

        """
        self.t0 = t0
        self.t1 = t1
        self.est_clock = t0
        self.is_prob_noise = 1
        self.estimator_buffer_power = estimator_buffer_power
        self.estimator_buffer_fill = estimator_buffer_fill
        self.estimator_update_time = estimator_update_time
        self.stacked_model_params = stacked_model_params
        self.buffer_size = buffer_size
        self.model_order = model_order

        # model params
        A = np.zeros([self.model_order, self.model_order])
        B = np.zeros([self.model_order, self.dim_input])
        C = np.zeros([self.dim_output, self.model_order])
        D = np.zeros([self.dim_output, self.dim_input])
        x0_est = np.zeros(self.model_order)

        self.my_model = utilities._model(A, B, C, D, x0_est)
        self.model_stack = []

        for k in range(self.stacked_model_params):
            self.model_stack.append(self.my_model)

        # number of prediction steps
        self.n_actor = n_actor

        # time between critic updates
        self.critic_update_time = critic_update_time

        # choice of running cost structure
        self.r_cost_struct = r_cost_struct

        # running cost parameters
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        self.r_cost_pars = [self.R1, self.R2]

        # integrated cost
        self.i_cost_val = 0
        self.critic_mode = critic_mode
        self.critic_clock = t0
        self.n_critic = n_critic
        self.n_critic = np.min([self.n_critic, self.buffer_size - 1])

        # control mode
        self.ctrl_mode = ctrl_mode
        self.ctrl_clock = t0

        self.sample_time = sample_time
        self.pred_step_size = pred_step_size * sample_time

        self.min_bounds = np.array(self.ctrl_bnds[:, 0])
        self.max_bounds = np.array(self.ctrl_bnds[:, 1])
        self.u_min = utilities._repMat(self.min_bounds, 1, n_actor)
        self.u_max = utilities._repMat(self.max_bounds, 1, n_actor)
        self.u_curr = self.min_bounds / 10
        self.u_init = utilities._repMat(self.min_bounds / 10, 1, self.n_actor)

        # buffer of previous controls
        self.u_buffer = np.zeros([buffer_size, self.dim_input])

        # buffer of previous outputs
        self.y_buffer = np.zeros([buffer_size, self.dim_output])

        # discount factor
        self.gamma = gamma

        # critic weights conditional logic
        if self.critic_mode == 1:
            self.dim_crit = int(((self.dim_output + self.dim_input) + 1) * (
                self.dim_output + self.dim_input) / 2 + (self.dim_output + self.dim_input))

            self.w_min = -1e3 * np.ones(self.dim_crit)
            self.w_max = 1e3 * np.ones(self.dim_crit)

        elif self.critic_mode == 2:
            self.dim_crit = int(((self.dim_output + self.dim_input) + 1)
                                * (self.dim_output + self.dim_input) / 2)
            self.w_min = np.zeros(self.dim_crit)
            self.w_max = 1e3 * np.ones(self.dim_crit)

        elif self.critic_mode == 3:
            self.dim_crit = int(self.dim_output + self.dim_input)
            self.w_min = np.zeros(self.dim_crit)
            self.w_max = 1e3 * np.ones(self.dim_crit)

        elif self.critic_mode == 4:
            self.dim_crit = int(
                self.dim_output + self.dim_output * self.dim_input + self.dim_input)
            self.w_min = -1e3 * np.ones(self.dim_crit)
            self.w_max = 1e3 * np.ones(self.dim_crit)

        self.Wprev = np.ones(int(self.dim_crit))

        self.Winit = self.Wprev

    def record_sys_state(self, system_state):
        self.system_state = system_state

    def running_cost(self, y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)
        """
        chi = np.concatenate([y, u])

        r = 0

        if self.r_cost_struct == 1:
            R1 = self.r_cost_pars[0]
            r = chi @ R1 @ chi

        elif self.r_cost_struct == 2:
            R1 = self.r_cost_pars[0]
            R2 = self.r_cost_pars[1]

            r = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi

        return r

    def update_icost(self, y, u):
        """
        Sample-to-sample integrated running cost. This can be handy to evaluate the performance of the agent.

        If the agent succeeded to stabilize the system, `icost` would converge to a finite value which is the performance mark.

        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)

        """
        self.i_cost_val += self.running_cost(y, u) * self.sample_time

    def _estimate_model(self, t, y):
        """
        Estimate model parameters by accumulating data buffers `u_buffer` and `y_buffer`

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample
            # Update buffers when using RL or requiring estimated model
            if self.ctrl_mode in (2, 3, 4, 5, 6):
                time_in_est_period = t - self.est_clock

                # Estimate model if required by ctrlStatMode
                if (time_in_est_period >= estimator_update_time) and (self.ctrl_mode in (2, 4, 6)):
                    # Update model estimator's internal clock
                    self.est_clock = t

                    try:
                        SSest = sippy.system_identification(self.y_buffer,
                                                            self.u_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sample_time)

                        self.my_model.updatePars(
                            SSest.A, SSest.B, SSest.C, SSest.D)

                    except:
                        print('Model estimation problem')
                        self.my_model.updatePars(np.zeros([self.model_order, self.model_order]),
                                                 np.zeros(
                                                     [self.model_order, self.dim_input]),
                                                 np.zeros(
                                                     [self.dim_output, self.model_order]),
                                                 np.zeros([self.dim_output, self.dim_input]))

                    # Model checks
                    if self.stacked_model_params > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        totAbsErrCurr = 1e8
                        for k in range(self.stacked_model_params):
                            A, B, C, D = self.model_stack[k].A, self.model_stack[
                                k].B, self.model_stack[k].C, self.model_stack[k].D
                            x0_est, _, _, _ = np.linalg.lstsq(C, y)
                            y_est, _ = self._dss_sim(
                                A, B, C, D, self.u_buffer, x0_est, y)
                            meanErr = np.mean(y_est - self.y_buffer, axis=0)

                            totAbsErr = np.sum(np.abs(meanErr))
                            if totAbsErr <= totAbsErrCurr:
                                totAbsErrCurr = totAbsErr
                                self.my_model.updatePars(
                                    SSest.A, SSest.B, SSest.C, SSest.D)

            # Update initial state estimate
            x0_est, _, _, _ = np.linalg.lstsq(self.my_model.C, y)
            self.my_model.updateIC(x0_est)

            if t >= self.estimator_buffer_fill:
                    # Drop probing noise
                self.is_prob_noise = 0

    def _actor(self, y, u_init, N, W, pred_step_size, ctrl_mode):
        """
        This method normally should not be altered. The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            # 'disp': True, 'verbose': 2}
            actor_opt_options = {'maxiter': 300, 'disp': False}
        else:
            # actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False,
            #                      'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}
            actor_opt_options = {'maxiter': 300, 'disp': False, 'ftol': 1e-7}

        isGlobOpt = 0

        myu_init = np.reshape(u_init, [N * self.dim_input, ])

        bnds = sp.optimize.Bounds(self.u_min, self.u_max, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {
                    'method': actor_opt_method,
                    'bounds': bnds,
                    'tol': 1e-7,
                    'options': actor_opt_options
                }

                U = basinhopping(lambda U: self._get_actor_cost(U, y, N, W, pred_step_size, ctrl_mode),
                                 myu_init,
                                 minimizer_kwargs=minimizer_kwargs,
                                 niter=10).x

            else:
                warnings.filterwarnings('ignore')
                U = minimize(lambda U: self._get_actor_cost(U, y, N, W, pred_step_size, ctrl_mode),
                             myu_init,
                             method=actor_opt_method,
                             tol=1e-7,
                             bounds=bnds,
                             options=actor_opt_options).x
        except ValueError:
            print("Actor's optimizer failed. Returning default action")
            U = myu_init

        return U[:self.dim_input]    # Return first action

    def _get_actor_cost(self, U, y, N, W, pred_step_size, ctrl_mode):
        myU = np.reshape(U, [N, self.dim_input])
        Y = np.zeros([N, self.dim_output])

        # System output prediction
        if (ctrl_mode == 1) or (ctrl_mode == 3) or (ctrl_mode == 5):
            # Via exogenously passed model
            Y[0, :] = y
            x = self.system_state

            for k in range(1, self.n_actor):
                # Euler scheme
                x = x + pred_step_size * \
                    self.sys_rhs([], x, myU[k - 1, :], [], self.m,
                                 self.I, self.dim_state, self.is_disturb)

                Y[k, :] = self.sys_out(x)

        elif (ctrl_mode == 2) or (ctrl_mode == 4) or (ctrl_mode == 6):
            # Via estimated model
            myU_upsampled = myU.repeat(
                int(pred_step_size / self.sample_time), axis=0)
            Yupsampled, _ = self._dss_sim(
                self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0_est, y)
            Y = Yupsampled[::int(pred_step_size / self.sample_time)]

        J = 0

        if (ctrl_mode == 1) or (ctrl_mode == 2):
            for k in range(N):
                J += self.gamma**k * self.running_cost(Y[k, :], myU[k, :])

        # RL: Q-learning with n_critic-1 roll-outs of running cost
        elif (ctrl_mode == 3) or (ctrl_mode == 4):
            for k in range(N - 1):
                J += self.gamma**k * self.running_cost(Y[k, :], myU[k, :])
            J += W @ self._phi(Y[-1, :], myU[-1, :])

        elif (ctrl_mode == 5) or (ctrl_mode == 6):
            # RL: (normalized) stacked Q-learning
            for k in range(N):
                Q = W @ self._phi(Y[k, :], myU[k, :])
                J += 1 / N * Q

        return J

    def _critic(self, Wprev, Winit, U, Y):
        """ Critic

        Customization
        -------------

        This method normally should not be altered, adjust `controller._get_critic_cost` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            # 'disp': True, 'verbose': 2}
            critic_opt_options = {'maxiter': 200, 'disp': False}
        else:
            warnings.filterwarnings('ignore')
            # critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False,'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}
            critic_opt_options = {'maxiter': 200, 'disp': False, 'ftol': 1e-7}

        bnds = sp.optimize.Bounds(self.w_min, self.w_max, keep_feasible=True)

        W = minimize(lambda W: self._get_critic_cost(W, U, Y), Winit,
                     method=critic_opt_method,
                     tol=1e-7,
                     bounds=bnds,
                     options=critic_opt_options).x

        return W

    def _get_critic_cost(self, W, U, Y):
        """ Cost function of the critic

        Currently uses value-iteration-like method

        Customization
        -------------

        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation

        """
        Jc = 0

        for k in range(self.dim_crit, 0, -1):
            y_prev = Y[k - 1, :]
            y_next = Y[k, :]
            u_prev = U[k - 1, :]
            u_next = U[k, :]

            # Temporal difference
            e = W @ self._phi(y_prev, u_prev) - self.gamma * self.Wprev @ self._phi(y_next, u_next) - self.running_cost(y_prev, u_prev)

            Jc += 1 / 2 * e**2

        return Jc

    def compute_action(self, t, y):
        """ Main method. """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            if self.ctrl_mode in (1, 2):

                # Apply control when model estimation phase is over
                if self.is_prob_noise and (self.ctrl_mode == 2):
                    return self.estimator_buffer_power * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and (self.ctrl_mode == 2):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    [], self.pred_step_size, self.ctrl_mode)

                elif (self.ctrl_mode == 1):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    [], self.pred_step_size, self.ctrl_mode)

            elif self.ctrl_mode in (3, 4, 5, 6):
                # Critic
                time_in_critic_update_time = t - self.critic_clock

                # Update data buffers
                self.u_buffer = utilities._pushVec(self.u_buffer, self.u_curr)
                self.y_buffer = utilities._pushVec(self.y_buffer, y)

                if time_in_critic_update_time >= self.critic_update_time:
                    # Update critic's internal clock
                    self.critic_clock = t

                    W = self._critic(
                        self.Wprev, self.Winit, self.u_buffer[-self.n_critic:, :], self.y_buffer[-self.n_critic:, :])
                    self.Wprev = W

                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.Winit = self.Wprev

                else:
                    W = self.Wprev

                # Actor. Apply control when model estimation phase is over
                if self.is_prob_noise and (self.ctrl_mode in (4, 6)):
                    u = self.estimator_buffer_power * \
                        (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and (self.ctrl_mode in (4, 6)):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    W, self.pred_step_size, self.ctrl_mode)

                elif self.ctrl_mode in (3, 5):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    W, self.pred_step_size, self.ctrl_mode)

            self.u_curr = u

            return u

        else:
            return self.u_curr

    def _dss_sim(self, A, B, C, D, uSqn, x0, y0):
        """
        Simulate output response of a discrete-time state-space model
        """
        if uSqn.ndim == 1:
            return y0, x0
        else:
            ySqn = np.zeros([uSqn.shape[0], C.shape[0]])
            xSqn = np.zeros([uSqn.shape[0], A.shape[0]])
            x = x0
            ySqn[0, :] = y0
            xSqn[0, :] = x0

            for k in range(1, uSqn.shape[0]):
                x = A @ x + B @ uSqn[k - 1, :]
                xSqn[k, :] = x
                ySqn[k, :] = C @ x + D @ uSqn[k - 1, :]

            return ySqn, xSqn

    def _phi(self, y, u):
        """
        Feature vector of critic

        In Q-learning mode, it uses both `y` and `u`. In value function approximation mode, it should use just `y`

        Customization
        -------------

        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like `W @ self._phi( y, u )`
        within `controller._get_critic_cost`

        """
        chi = np.concatenate([y, u])

        if self.critic_mode == 1:
            return np.concatenate([_uptria2vec(np.kron(chi, chi)), chi])

        elif self.critic_mode == 2:
            return np.concatenate([_uptria2vec(np.kron(chi, chi))])

        elif self.critic_mode == 3:
            return chi * chi

        elif self.critic_mode == 4:
            return np.concatenate([y**2, np.kron(y, u), u**2])

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        All the learned parameters are retained
        """
        self.ctrl_clock = t0
        self.u_curr = self.min_bounds / 10


class NominalController(utilities.Generic):
    """
    This is a class of nominal controllers used for benchmarking of optimal controllers.
    Specification should be provided for each individual case (system)

    The controller is sampled.

    For a three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Parameters
    ----------
    m : int
        * mass of robot

    I : int
        * Inertia around vertical axis of the robot

    ctrl_gain : int
        * Controller gain

    t0 : int
        * Initial value of the controller's internal clock

    sample_time : int or float
        * Controller's sampling time (in seconds)

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self,
                 t0=0,
                 m=10,
                 I=1,
                 ctrl_gain=10,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 sample_time=0.1):

        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = np.array([[f_min, f_max], [m_min, m_max]])
        self.ctrl_clock = t0
        self.sample_time = sample_time
        self.u_curr = np.zeros(2)
        self.m = m
        self.I = I

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrl_clock = t0
        self.u_curr = np.zeros(2)

    def _zeta(self, x_ni, theta):
        """
        utilities.Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)

        """

        sigma_tilde = x_ni[0] * np.cos(theta) + x_ni[1] * \
            np.sin(theta) + np.sqrt(np.abs(x_ni[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4 * x_ni[0]**3 - 2 * \
            np.abs(x_ni[2])**3 * np.cos(theta) / sigma_tilde**3

        nablaF[1] = 4 * x_ni[1]**3 - 2 * \
            np.abs(x_ni[2])**3 * np.sin(theta) / sigma_tilde**3

        nablaF[2] = (3 * x_ni[0] * np.cos(theta) + 3 * x_ni[1] * np.sin(theta) + 2 *
                     np.sqrt(np.abs(x_ni[2]))) * x_ni[2]**2 * np.sign(x_ni[2]) / sigma_tilde**3

        return nablaF

    def _kappa(self, x_ni, theta):
        """
        Stabilizing controller for NI-part

        """
        kappa_val = np.zeros(2)

        G = np.zeros([3, 2])
        G[:, 0] = np.array([1, 0, x_ni[1]])
        G[:, 1] = np.array([0, 1, -x_ni[0]])

        zeta_val = self._zeta(x_ni, theta)

        kappa_val[0] = - np.abs(np.dot(zeta_val, G[:, 0])
                                )**(1 / 3) * np.sign(np.dot(zeta_val, G[:, 0]))
        kappa_val[1] = - np.abs(np.dot(zeta_val, G[:, 1])
                                )**(1 / 3) * np.sign(np.dot(zeta_val, G[:, 1]))

        return kappa_val

    def _Fc(self, x_ni, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """

        sigma_tilde = x_ni[0] * np.cos(theta) + x_ni[1] * \
            np.sin(theta) + np.sqrt(np.abs(x_ni[2]))

        F = x_ni[0]**4 + x_ni[1]**4 + np.abs(x_ni[2])**3 / sigma_tilde

        z = eta - self._kappa(x_ni, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _theta_minimizer(self, x_ni, eta):
        theta_init = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        theta_val = minimize(lambda theta: self._Fc(x_ni, eta, theta), theta_init,
                             method='trust-constr', tol=1e-6, bounds=bnds, options=options).x

        return theta_val

    def _cart_to_nh(self, cart_coords):
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: `\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        x_ni = np.zeros(3)
        eta = np.zeros(2)

        xc = cart_coords[0]
        yc = cart_coords[1]
        alpha = cart_coords[2]
        v = cart_coords[3]
        omega = cart_coords[4]

        x_ni[0] = alpha
        x_ni[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        x_ni[2] = - 2 * (yc * np.cos(alpha) - xc * np.sin(alpha)) - \
            alpha * (xc * np.cos(alpha) + yc * np.sin(alpha))

        eta[0] = omega
        eta[1] = (yc * np.cos(alpha) - xc * np.sin(alpha)) * omega + v

        return [x_ni, eta]

    def _nh_to_cartctrl(self, x_ni, eta, u_ni):
        """
        Get control for Cartesian NI from NH coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: `\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)


        """

        uCart = np.zeros(2)

        uCart[0] = self.m * (u_ni[1] + x_ni[1] * eta[0]**2 +
                             1 / 2 * (x_ni[0] * x_ni[1] * u_ni[0] + u_ni[0] * x_ni[2]))
        uCart[1] = self.I * u_ni[0]

        return uCart

    def compute_action(self, t, y):
        """
        See algorithm description in [[1]_], [[2]_]

        **This algorithm needs full-state measurement of the robot**

        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample

            # This controller needs full-state measurement
            x_ni, eta = self._cart_to_nh(y)
            theta_star = self._theta_minimizer(x_ni, eta)
            kappa_val = self._kappa(x_ni, theta_star)
            z = eta - kappa_val
            u_ni = - self.ctrl_gain * z
            u = self._nh_to_cartctrl(x_ni, eta, u_ni)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0],
                                   self.ctrl_bnds[k, 1])

            self.u_curr = u

            return u

        else:
            return self.u_curr


class Simulation(utilities.Generic):
    """class to create and run simulation.

    Parameters
    ----------

    system : object of type `System` class

    controller : object of type `Controller` class

    nominal_ctrl : object of type `NominalController` class

    a_tol : float
        * ODE solver sensitivity hyperparameter

    r_tol : float
        * ODE solver sensitivity hyperparameter

    x_min : int
        * minimum x limit of graph

    x_max : int
        * maximum x limit of graph

    y_min : int
        * minimum y limit of graph

    y_max : int
        * minimum y limit of graph

    """

    def __init__(self,
                 system,
                 controller,
                 nominal_ctrl,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10):
        """

        CONTROL FLOW LOGIC: IGNORE

        """

        if hasattr(controller, '__len__') and hasattr(nominal_ctrl, '__len__'):
            self.num_controllers = len(controller)
            self.num_nom_controllers = len(nominal_ctrl)

            if self.num_nom_controllers > 1 and self.num_controllers == 1 and system.num_controllers == 1:
                self.nominal_ctrl = nominal_ctrl[0]
                self.controller = controller[0]

            elif self.num_nom_controllers > 1 and self.num_controllers > 1 and system.num_controllers == 1:
                self.error_message = "You forgot to call function `add_bots` on System object."

            elif self.num_nom_controllers > 1 and self.num_controllers > 1 and system.num_controllers > 1:

                if self.num_nom_controllers == self.num_controllers == system.num_controllers > 1:
                    self.nominal_ctrlers = nominal_ctrl
                    self.controllers = controller

                else:
                    self.error_message = "Number of controllers, nominal controllers, and registered system controllers should be equal."

            elif self.num_nom_controllers == 1 and self.num_controllers > 1 and system.num_controllers == 1:
                self.controller = controller[0]
                self.nominal_ctrl = nominal_ctrl[0]

        elif hasattr(controller, '__len__') or hasattr(nominal_ctrl, '__len__'):
            self.error_message = "Number of controllers and nominal controllers need to be identical."

        else:
            self.controller = controller
            self.nominal_ctrl = nominal_ctrl
            self.num_controllers = 1

            if system.num_controllers > 1:
                self.error_message = "Warning: you called system.add_bots() but did not pass the same number of Controller objects when instantiating the Simulation class."

        if hasattr(self, 'error_message'):
            print(self.error_message)
            return None

        else:
            """

            VISUALIZATION PARAMS

            """

            # x and y limits of scatter plot. Used so far rather for visualization
            # only, but may be integrated into the actor as constraints
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

            """

            CONTROLLER AND SYSTEM PARAMS

            """
            self.system = system

            # control constraints
            self.dim_input = system.dim_input
            self.f_min = system.f_min
            self.f_max = system.f_max
            self.f_man = system.f_man
            self.n_man = system.n_man
            self.m_min = system.m_min
            self.m_max = system.m_max

            self.control_bounds = system.control_bounds

            closed_loop = system.closed_loop

            if self.num_controllers == 1:
                self.sample_time, self.ctrl_mode, self.t1, self.t0 = self._get_controller_info(
                    self.controller)

                self.system_state, self.full_state, self.alpha, self.initial_x, self.initial_y = self._get_system_info(
                    system)

                self.simulator = sp.integrate.RK45(closed_loop,
                                                   self.t0,
                                                   self.full_state,
                                                   self.t1,
                                                   max_step=self.sample_time / 2,
                                                   first_step=1e-6,
                                                   atol=a_tol,
                                                   rtol=r_tol)

            elif self.num_controllers > 1:
                self.sample_times, self.ctrl_modes, self.t1s, self.t0 = self._get_controller_info(
                    self.controllers, multi=True)

                self.system_states, self.full_states, self.alphas, self.initial_xs, self.initial_ys, self.u0s = self._get_system_info(
                    system, multi=True)

                self.simulators = []

                for i in range(self.num_controllers):
                    self.system.set_multi_sim(i)

                    simulator = sp.integrate.RK45(closed_loop,
                                                  self.t0,
                                                  self.full_states[i],
                                                  self.t1s[i],
                                                  max_step=self.sample_times[
                                                      i] / 2,
                                                  first_step=1e-6,
                                                  atol=a_tol,
                                                  rtol=r_tol)

                    self.simulators.append(simulator)

    def _get_controller_info(self, controller, multi=False):
        # if we have a single controller
        if multi is False:
            sample_time = controller.sample_time
            ctrl_mode = controller.ctrl_mode
            t1 = controller.t1

            return sample_time, ctrl_mode, t1, 0

        # if we have multiple controllers
        else:
            controllers = controller
            num_controllers = len(controllers)

            sample_times = []
            ctrl_modes = []
            t1s = []

            for controller in controllers:
                sample_times.append(controller.sample_time)
                ctrl_modes.append(controller.ctrl_mode)
                t1s.append(controller.t1)

            return sample_times, ctrl_modes, t1s, 0

    def _get_system_info(self, system, multi=False):
        if multi is False:
            system_state = system.system_state
            full_state = system.full_state
            alpha = system.initial_alpha
            initial_x = system.initial_x
            initial_y = system.initial_y

            return system_state, full_state, alpha, initial_x, initial_y

        else:
            system_states = system.system_state
            full_states = system.full_state
            alphas = system.alpha
            initial_xs = system.system_state[:, 0]
            initial_ys = system.system_state[:, 1]
            u0s = system.u0
            try:
                q0s = system.q0
            except:
                pass

            return system_states, full_states, alphas, initial_xs, initial_ys, u0s

    def _ctrl_selector(self, t, y, uMan, nominal_ctrl, controller, mode):
        """
        Main interface for different agents

        """

        if mode == 0:  # Manual control
            u = uMan
        elif mode == -1:  # Nominal controller
            u = nominal_ctrl.compute_action(t, y)
        elif mode > 0:  # Optimal controller
            u = controller.compute_action(t, y)

        return u

    def _create_figure_plots(self, system, controller, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        y0 = System.get_curr_state(self.system_state)
        self.alpha = self.alpha / 2 / np.pi

        plt.close('all')

        self.sim_fig = plt.figure(figsize=(fig_width, fig_height))

        """

        Simulation subplot

        """
        self.xy_plane_axes = self.sim_fig.add_subplot(221,
                                                      autoscale_on=False,
                                                      xlim=(self.x_min,
                                                            self.x_max),
                                                      ylim=(self.y_min,
                                                            self.y_max),
                                                      xlabel='x [m]',
                                                      ylabel='y [m]',
                                                      title=' Simulation: \n Pause - space, q - quit, click - data cursor')

        self.xy_plane_axes.set_aspect('equal', adjustable='box')

        self.xy_plane_axes.plot([self.x_min, self.x_max], [
            0, 0], 'k--', lw=0.75)   # x-axis

        self.xy_plane_axes.plot([0, 0], [self.y_min, self.y_max],
                                'k--', lw=0.75)   # y-axis

        self.traj_line, = self.xy_plane_axes.plot(
            self.initial_x, self.initial_y, 'b--', lw=0.5)

        self.robot_marker = utilities._pltMarker(angle=self.alpha)

        text_time = 't = {time:2.3f}'.format(time=self.t0)

        self.text_time_handle = self.xy_plane_axes.text(0.05, 0.95,
                                                        text_time,
                                                        horizontalalignment='left',
                                                        verticalalignment='center',
                                                        transform=self.xy_plane_axes.transAxes)

        self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Proximity subplot

        """
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')

        self.sol_axes.title.set_text('Proximity-to-Target')

        self.sol_axes.plot([self.t0, self.t1], [0, 0],
                           'k--', lw=0.75)   # Help line

        self.norm_line, = self.sol_axes.plot(self.t0, la.norm(
            [self.initial_x, self.initial_y]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')

        self.alpha_line, = self.sol_axes.plot(
            self.t0, self.alpha, 'r-', lw=0.5, label=r'$\alpha$ [rad]')

        self.sol_axes.legend(fancybox=True, loc='upper right')

        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Cost subplot

        """
        self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            0, 1e4 * controller.running_cost(y0, system.u0)), yscale='symlog', xlabel='t [s]')

        self.cost_axes.title.set_text('Cost')

        r = controller.running_cost(y0, system.u0)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=0)

        self.text_icost_handle = self.sim_fig.text(
            0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')

        self.r_cost_line, = self.cost_axes.plot(
            self.t0, r, 'r-', lw=0.5, label='r')

        self.i_cost_line, = self.cost_axes.plot(
            self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')

        self.cost_axes.legend(fancybox=True, loc='upper right')

        """

        Control subplot

        """
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([system.f_min, system.m_min]), 1.1 * np.max([system.f_max, system.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line

        self.ctrl_lines = self.ctrlAxs.plot(
            self.t0, utilities._toColVec(system.u0).T, lw=0.5)

        self.ctrlAxs.legend(
            iter(self.ctrl_lines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        # Pack all lines together
        self.lines = [self.traj_line, self.norm_line, self.alpha_line,
                      self.r_cost_line, self.i_cost_line, self.ctrl_lines]

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

        return self.sim_fig

    def _create_figure_plots_multi(self, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        self.colors = ['b', 'r', 'g', 'o']
        self.color_pairs = [['b', 'g'], ['r', 'm'], ['g', 'y'], ['o', 'teal']]

        y0_list = []

        for system_state in self.system_states:
            y0 = System.get_curr_state(system_state)
            y0_list.append(y0)

        self.alphas = self.alphas / 2 / np.pi

        plt.close('all')

        self.sim_fig = plt.figure(figsize=(fig_width, fig_height))

        """

        Simulation subplot

        """
        self.xy_plane_axes = self.sim_fig.add_subplot(221,
                                                      autoscale_on=False,
                                                      xlim=(self.x_min,
                                                            self.x_max),
                                                      ylim=(self.y_min,
                                                            self.y_max),
                                                      xlabel='x [m]',
                                                      ylabel='y [m]',
                                                      title=' Simulation: \n Pause - space, q - quit, click - data cursor')

        self.xy_plane_axes.set_aspect('equal', adjustable='box')
        self.xy_plane_axes.plot([self.x_min, self.x_max], [
                                0, 0], 'k--', lw=0.75)   # x-axis
        self.xy_plane_axes.plot(
            [0, 0], [self.y_min, self.y_max], 'k--', lw=0.75)   # y-axis

        self.traj_lines = []
        self.robot_markers = []
        self.text_time_handles = []
        self.run_handles = []
        text_time = 't = {time:2.3f}'.format(time=self.t0)
        time_positions = [[0.05, 0.95], [0.70, 0.95], [0.05, 0.10], [0.70, 0.10]]
        run_positions = [[0.15, 0.90], [0.80, 0.90], [0.15, 0.13], [0.80, 0.13]]

        for i in range(self.num_controllers):
            self.traj_line, = self.xy_plane_axes.plot(self.initial_xs[i], self.initial_ys[i], f'{self.colors[i]}--', lw=0.5, c=self.colors[i])

            self.robot_marker = utilities._pltMarker(angle=self.alphas[i])

            self.run_handle = self.xy_plane_axes.text(run_positions[i][0], run_positions[i][1], f"Run: 0", horizontalalignment='center', transform=self.xy_plane_axes.transAxes)

            self.text_time_handle = self.xy_plane_axes.text(time_positions[i][0], time_positions[i][
                                                            1], text_time, horizontalalignment='left', verticalalignment='center', transform=self.xy_plane_axes.transAxes)

            self.traj_lines.append(self.traj_line)
            self.robot_markers.append(self.robot_marker)
            self.text_time_handles.append(self.text_time_handle)
            self.run_handles.append(self.run_handle)

        self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Proximity subplot

        """
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')

        self.sol_axes.title.set_text('Proximity-to-Target')

        self.sol_axes.plot([self.t0, max(self.t1s)], [0, 0],
                           'k--', lw=0.75)   # Help line

        # logic for multiple controllers
        self.norm_lines = []
        self.alpha_lines = []

        for i in range(self.num_controllers):
            self.norm_line, = self.sol_axes.plot(self.t0, la.norm([self.initial_xs[i], self.initial_ys[i]]), f'{self.color_pairs[i][0]}--', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')

            self.alpha_line, = self.sol_axes.plot(self.t0, self.alphas[i], f'{self.color_pairs[i][1]}--', lw=0.5, label=r'$\alpha$ [rad]')

            self.norm_lines.append(self.norm_line)
            self.alpha_lines.append(self.alpha_line)

        self.sol_axes.legend(fancybox=True, loc='upper right')

        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Cost subplot

        """

        self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            0, 1e4 * self.controllers[0].running_cost(y0_list[0], self.u0s[0])), yscale='symlog', xlabel='t [s]')

        self.cost_axes.title.set_text('Cost')

        self.text_icost_handles = []
        self.r_cost_lines = []
        self.i_cost_lines = []

        text_positions = [[0.05, 0.50], [
            0.05, 0.48], [0.50, 0.50], [0.50, 0.48]]

        for i in range(self.num_controllers):
            r = self.controllers[i].running_cost(y0_list[i], self.u0s[i])
            text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(
                icost=0)

            self.text_icost_handle = self.sim_fig.text(text_positions[i][0], text_positions[
                                                       i][1], text_icost, horizontalalignment='left', verticalalignment='center')

            self.r_cost_line, = self.cost_axes.plot(
                self.t0, r, f'{self.color_pairs[i][0]}-', lw=0.5, label='r')

            self.i_cost_line, = self.cost_axes.plot(
                self.t0, 0, f'{self.color_pairs[i][1]}-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')

            self.text_icost_handles.append(self.text_icost_handle)
            self.r_cost_lines.append(self.r_cost_line)
            self.i_cost_lines.append(self.i_cost_line)

            self.cost_axes.legend(fancybox=True, loc='upper right')

        """

        Control subplot

        """
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, max(self.t1s)], [0, 0],
                          'k--', lw=0.75)   # Help line

        # logic for multiple controllers
        self.all_ctrl_lines = []

        clabels = ['F [N]', 'M [Nm]']

        for i in range(self.num_controllers):
            u = np.expand_dims(self.u0s[i], axis=0)
            self.ctrl_lines = self.ctrlAxs.plot(
                self.t0, u, lw=0.5, label=clabels)

            self.all_ctrl_lines.append(self.ctrl_lines)

        handles, labels = self.ctrlAxs.get_legend_handles_labels()

        # clabels = clabels[::-1]
        new_labels = [clabels] * self.num_controllers
        new_labels = list(itertools.chain.from_iterable(new_labels))

        labels = new_labels

        self.ctrlAxs.legend(handles, labels, fancybox=True, loc='upper right')

        self.all_lines = [[] for i in range(self.num_controllers)]

        for i in range(self.num_controllers):
            self.all_lines[i].extend([self.traj_lines[i], self.norm_lines[i], self.alpha_lines[
                                     i], self.r_cost_lines[i], self.i_cost_lines[i], self.all_ctrl_lines[i]])

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for line in self.all_lines:
            for item in line:
                if isinstance(item, list):
                    for subitem in item:
                        datacursor(subitem)
                else:
                    datacursor(item)

        return self.sim_fig

    def _graceful_exit(self, plt_close=True):
        if plt_close is True:
            plt.close('all')

        # graceful exit from Jupyter notebook
        try:
            __IPYTHON__
            return None

        # graceful exit from terminal
        except NameError:
            if plt_close is True:
                print("Program exit")
                sys.exit()
            else:
                return None

    def _initialize_figure(self):
        self.scatter_plots = []

        if self.num_controllers > 1:
            self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs, self.initial_ys, s=400, c=self.colors[
                                                          :self.num_controllers], marker=self.robot_marker.marker)
            self.scatter_plots.append(self.sol_scatter)

            if self.show_annotations:
                self.annotations = []
                for i in range(self.num_controllers):
                    self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')
                    self.annotations.append(self.annotation)

        else:
            self.sol_scatter = self.xy_plane_axes.scatter(
                self.initial_x, self.initial_y, marker=self.robot_marker.marker, s=400, c='b')

        return self.sol_scatter,

    def _log_data_row(self, dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
        with open(dataFile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, xCoord, yCoord, alpha,
                             v, omega, icost, u[0], u[1]])

    def _log_data(self, n_runs, save=False):
        dataFiles = [None] * n_runs

        if save:
            cwd = os.getcwd()
            datafolder = '/data'
            dataFolder_path = cwd + datafolder

            # create data dir
            pathlib.Path(dataFolder_path).mkdir(parents=True, exist_ok=True)

            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%Hh%Mm%Ss")
            dataFiles = [None] * n_runs
            for k in range(0, n_runs):
                dataFiles[k] = dataFolder_path + '/RLsim__' + date + \
                    '__' + time + '__run{run:02d}.csv'.format(run=k + 1)
                with open(dataFiles[k], 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]',
                                     'v [m/s]', 'omega [rad/s]', 'int r sample_time', 'F [N]', 'M [N m]'])

        return dataFiles

    def _on_key_press(self, event, anm, args):
        if event.key == ' ':
            if anm.running is True:
                anm.event_source.stop()
                anm.running = False

            elif anm.running is False:
                anm.event_source.start()
                anm.running = True

        elif event.key == 'q':
            _, controller, nominal_ctrl, simulator, _ = args

            if self.num_controllers > 1:
                for i in range(self.num_controllers):
                    self._reset_sim(
                        controller[i], nominal_ctrl, simulator[i], i)

            else:
                self._reset_sim(controller, nominal_ctrl, simulator)

            self._graceful_exit()

    def print_simu_stats(self):
        self.mean_rc = []  # mean rc
        self.var_rc = []  # var rc
        self.mean_velocity = []  # mean vel
        self.var_velocity = []  # var vel
        self.var_alpha = []  # var alpha

        if self.num_controllers > 1:
            n_runs = self.n_runs[0]

        for mid in range(self.num_controllers):
            self.mean_rc.append(round(statistics.mean(
                self.statistics['running_cost'][mid]), 2))
            self.var_rc.append(round(statistics.variance(
                self.statistics['running_cost'][mid]), 2))
            self.mean_velocity.append(
                round(statistics.mean(self.statistics['velocity'][mid]), 2))
            self.var_velocity.append(
                round(statistics.variance(self.statistics['velocity'][mid]), 2))
            self.var_alpha.append(
                round(statistics.variance(self.statistics['alpha'][mid]), 2))

        print(f"Total runs: {n_runs}")
        for mid in range(self.num_controllers):
            print(f"""Statistics for controller {mid+1}:
            - Mean of running cost: {self.mean_rc[mid]}
            - Variance of running cost: {self.var_rc[mid]}
            - Mean of velocity: {self.mean_velocity[mid]}
            - Variance of velocity: {self.var_velocity[mid]}
            - Variance of alpha: {self.var_alpha[mid]}
                """)

        return None

    def _print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180

        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]',
                     'v [m/s]', 'omega [rad/s]', 'int r sample_time', 'F [N]', 'M [N m]']
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f',
                     '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat,
                         headers='firstrow', tablefmt='grid')

        print(table)

    def _reset_all_lines(self, lines):
        for line in lines:
            if isinstance(line, list) is False:
                self._reset_line(line)
            else:
                self._reset_all_lines(line)

    def _reset_line(self, line):
        line.set_data([], [])

    def _reset_sim(self, controller, nominal_ctrl, simulator, mid=None):
        if self.print_statistics_at_step:
            if mid is not None:
                print(f'........Controller {mid+1}: Run #{self.current_runs[mid]} done........')
            else:
                print(f'........Run {self.current_run} done........')

        if self.is_log_data:
            self.current_data_file = self.data_files[self.current_run - 1]

        # Reset simulator
        simulator.status = 'running'
        simulator.t = self.t0

        if mid is not None:
            simulator.y = self.full_states[mid]

        else:
            simulator.y = self.full_state

        # Reset controller
        if controller.ctrl_mode > 0:
            controller.reset(self.t0)
        else:
            nominal_ctrl.reset(self.t0)

        if self.current_runs[mid] <= self.n_runs[mid]:
            if mid is None:
                self._reset_all_lines(self.lines)

            else:
                self._reset_all_lines(self.all_lines[mid])

    def _run_animation(self, system, controller, nominal_ctrl, simulator, fig_width, fig_height, multi_controllers=False):
        animate = True
        self.exit_animation = False

        if multi_controllers is True:
            controllers = controller
            simulators = simulator
            nominal_ctrlers = nominal_ctrl

            self.sim_fig = self._create_figure_plots_multi(
                fig_width, fig_height)
            fargs = (system, controllers, nominal_ctrlers, simulators, animate)

            self.anm = animation.FuncAnimation(self.sim_fig,
                                               self._wrapper_take_steps_multi,
                                               fargs=fargs,
                                               init_func=self._initialize_figure,
                                               interval=1,
                                               blit=False)

        else:
            self.sim_fig = self._create_figure_plots(
                system, controller, fig_width, fig_height)
            fargs = (system, controller, nominal_ctrl, simulator, animate)

            self.anm = animation.FuncAnimation(self.sim_fig,
                                               self._wrapper_take_steps,
                                               fargs=fargs,
                                               init_func=self._initialize_figure,
                                               interval=1,
                                               blit=False)

        self.anm.running = True
        self.sim_fig.canvas.mpl_connect(
            'key_press_event', lambda event: self._on_key_press(event, self.anm, fargs))
        self.sim_fig.tight_layout()
        plt.show()

    def run_simulation(self, n_runs=1, fig_width=8, fig_height=8, close_plt_on_finish=True, show_annotations=False, print_summary_stats=False, is_log_data=False, is_visualization=True, print_statistics_at_step=False):
        """
        is_log_data : bool
            * log data to local drive?

        is_visualization : bool
            * visual simulation?

        print_statistics_at_step : bool
            * print results of simulation?
        """
        if hasattr(self, 'error_message') is False:
            self.is_log_data = is_log_data
            self.is_visualization = is_visualization
            self.print_statistics_at_step = print_statistics_at_step
            self.print_summary_stats = print_summary_stats
            self.statistics = {'running_cost': {}, 'velocity': {}, 'alpha': {}}

            if self.print_statistics_at_step:
                warnings.filterwarnings('ignore')

            for i in range(self.num_controllers):
                self.statistics['running_cost'].setdefault(i, [])
                self.statistics['velocity'].setdefault(i, [])
                self.statistics['alpha'].setdefault(i, [])

            if self.num_controllers > 1:
                self.current_runs = np.ones(self.num_controllers, dtype=np.int64)
                self.n_runs = np.array([n_runs] * self.num_controllers)
                self.show_annotations = show_annotations
                self.keep_stepping = np.ones((self.num_controllers), dtype=bool)

            else:
                self.current_run = 1
                self.n_runs = n_runs

            self.close_plt_on_finish = close_plt_on_finish
            self.data_files = self._log_data(n_runs, save=self.is_log_data)

            # CODE IN THIS CONDITIONAL BLOCK IS IN DEVELOPMENT NEEDS TO BE UPDATED
            if self.is_visualization is False:
                self.current_data_file = data_files[0]

                t = self.simulator.t

                while self.current_run <= self.n_runs:
                    while t < self.t1:
                        self._take_step(self.system, self.controller,
                                        self.nominal_ctrl, self.simulator)
                        t += 1

                    else:
                        self._reset_sim(self.controller,
                                        self.nominal_ctrl, self.simulator)
                        icost = 0

                        for line in self.all_lines:
                            for item in line:
                                if item != self.traj_line:
                                    if isinstance(item, list):
                                        for subitem in item:
                                            self._reset_line(subitem)
                                    else:
                                        self._reset_line(item)

                        self._update_line(self.traj_line, np.nan, np.nan)
                    self.current_run += 1
                else:
                    self._graceful_exit()

            else:
                if self.num_controllers > 1:
                    self._run_animation(self.system,
                                        self.controllers,
                                        self.nominal_ctrlers,
                                        self.simulators,
                                        fig_width,
                                        fig_height,
                                        multi_controllers=True)
                else:
                    self._run_animation(
                        self.system, self.controller, self.nominal_ctrl, self.simulator, fig_width, fig_height)
        else:
            pass

    def _take_step(self, system, controller, nominal_ctrl, simulator, animate=False):
        simulator.step()

        t = simulator.t
        full_state = simulator.y

        system_state = system.system_state
        y = system.get_curr_state(system_state)

        u = self._ctrl_selector(
            t, y, system.u_man, nominal_ctrl, controller, controller.ctrl_mode)

        system.set_latest_action(u)

        controller.record_sys_state(system_state)
        controller.update_icost(y, u)

        x_coord = full_state[0]
        y_coord = full_state[1]
        alpha = full_state[2]
        v = full_state[3]
        omega = full_state[4]

        icost = controller.i_cost_val
        alpha_deg = alpha / np.pi * 180
        r = controller.running_cost(y, u)
        text_time = 't = {time:2.3f}'.format(time=t)

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        self._collect_print_statistics(t, x_coord, y_coord, alpha, v, omega, icost, r, u, self.l2_norm)

        if self.is_log_data:
            self._log_data_row(self.current_data_file, t, x_coord,
                               y_coord, alpha, v, omega, icost.val, u)

        if animate:
            self._update_all_lines(
                text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u)

        return t

    def _take_step_multi(self, mid, system, controller, nominal_ctrl, simulator, animate=False):
        system.set_multi_sim(mid)
        simulator.step()

        t = simulator.t
        full_state = simulator.y

        system_state = self.system_states[mid]

        y = system.get_curr_state(system_state)

        u = self._ctrl_selector(
            t, y, system.u_man, nominal_ctrl, controller, controller.ctrl_mode)

        system.set_latest_action(u, mid)

        controller.record_sys_state(system_state)
        controller.update_icost(y, u)

        x_coord = full_state[0]
        y_coord = full_state[1]
        alpha = full_state[2]
        v = full_state[3]
        omega = full_state[4]

        icost = controller.i_cost_val

        alpha_deg = alpha / np.pi * 180
        r = controller.running_cost(y, u)
        text_time = 't = {time:2.3f}'.format(time=t)

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        self._collect_print_statistics(t, x_coord, y_coord, alpha, v, omega, icost, r, u, self.l2_norm, mid)
        
        if self.is_log_data:
            self._log_data_row(self.current_data_file, t, x_coord,
                               y_coord, alpha, v, omega, icost.val, u)

        if animate:
            self._update_all_lines_multi(text_time, full_state, alpha_deg,
                                         x_coord, y_coord, t, alpha, r, icost, u, self.l2_norm, mid)

        return t, x_coord, y_coord

    def _collect_print_statistics(self, t, x_coord, y_coord, alpha, v, omega, icost, r, u, l2_norm, mid=None):
        self.statistics['running_cost'][0].append(r)
        self.statistics['velocity'][0].append(v)
        self.statistics['alpha'][0].append(alpha)

        if self.print_statistics_at_step:
            print(f"Controller\t{mid}")
            self._print_sim_step(t, x_coord, y_coord, alpha, v, omega, icost, u)

    def _update_line(self, line, new_x, new_y):
        line.set_xdata(np.append(line.get_xdata(), new_x))
        line.set_ydata(np.append(line.get_ydata(), new_y))

    def _update_text(self, text_handle, new_text):
        text_handle.set_text(new_text)

    def _update_all_lines(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, l2_norm):
        """
        Update lines on all scatter plots
        """
        self._update_text(self.text_time_handle, text_time)

        # Update the robot's track on the plot
        self._update_line(self.traj_line, *full_state[:2])

        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot
        self.sol_scatter.remove()
        self.sol_scatter = self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_marker.marker, s=400, c='b')

        # Solution
        self._update_line(self.norm_line, t, l2_norm)
        self._update_line(self.alpha_line, t, alpha)

        # Cost
        self._update_line(self.r_cost_line, t, r)
        self._update_line(self.i_cost_line, t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handle, text_icost)

        # Control
        for (line, uSingle) in zip(self.ctrl_lines, u):
            self._update_line(line, t, uSingle)

    def _update_all_lines_multi(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, l2_norm, mid):
        """
        Update lines on all scatter plots
        """
        self._update_text(self.text_time_handles[mid], text_time)
        self._update_text(self.run_handles[mid], f"A{mid+1}, run: {self.current_runs[mid]}")

        # Update the robot's track on the plot
        self._update_line(self.traj_lines[mid], x_coord, y_coord)

        # Rotate the robot on the plot
        self.robot_markers[mid].rotate(alpha_deg)

        self.scatter_plots.append(self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_markers[mid].marker, s=400, c=self.colors[mid]))

        # Solution
        self._update_line(self.norm_lines[mid], t, l2_norm)
        self._update_line(self.alpha_lines[mid], t, alpha)

        # Cost
        self._update_line(self.r_cost_lines[mid], t, r)
        self._update_line(self.i_cost_lines[mid], t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handles[mid], text_icost)

        # Control
        for (line, uSingle) in zip(self.all_ctrl_lines[mid], u):
            self._update_line(line, t, uSingle)

    def _wrapper_take_steps(self, k, *args):
        _, controller, nominal_ctrl, simulator, _ = args
        t = simulator.t

        if self.current_run <= self.n_runs:
            if t < self.t1:
                t = self._take_step(*args)

            else:
                self.current_run += 1
                self._reset_sim(controller, nominal_ctrl, simulator)

        elif self.current_run > self.n_runs and self.exit_animation is False:
            if self.print_summary_stats is True:
                self.print_simu_stats()

            self.anm.running = False
            self.exit_animation = True

        else:
            if self.close_plt_on_finish is True:
                self._graceful_exit()

            elif self.close_plt_on_finish is False:
                self._graceful_exit(plt_close=False)

    def _wrapper_take_steps_multi(self, k, *args):
        system, controllers, nominal_ctrlers, simulators, animate = args

        for i in range(self.num_controllers):
            if self.current_runs[i] <= self.n_runs[i]:
                self.keep_stepping[i] = True
            else:
                self.keep_stepping[i] = False

        if self.keep_stepping.any() == True:
            for scat in self.scatter_plots:
                scat.remove()

            self.scatter_plots = []

            if self.show_annotations:
                for ann in self.annotations:
                    ann.remove()

                self.annotations = []

            for i in range(self.num_controllers):
                if self.keep_stepping[i] == True:
                    t = simulators[i].t

                    if t < self.t1s[i]:
                        t, x_coord, y_coord = self._take_step_multi(
                            i, system, controllers[i], nominal_ctrlers[i], simulators[i], animate)

                        if self.show_annotations:
                            self.annotations.append(self.xy_plane_axes.annotate(f'{i+1}', xy=(x_coord + 0.5, y_coord + 0.5), color='k'))

                    else:
                        self.current_runs[i] += 1
                        self._reset_sim(controllers[i], nominal_ctrlers[
                                        i], simulators[i], i)
                else:
                    self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs[i], self.initial_ys[
                                                                  i], s=400, c=self.colors[i], marker=self.robot_markers[i].marker)

                    if self.show_annotations:
                        self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')
                    continue

        elif self.keep_stepping.all() == False and self.exit_animation is False:

            for i in range(self.num_controllers):
                self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs[i], self.initial_ys[
                                                              i], s=400, c=self.colors[i], marker=self.robot_markers[i].marker)

                if self.show_annotations:
                    self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')

            if self.print_summary_stats:
                self.print_simu_stats()

            self.anm.running = False
            self.exit_animation = True

        else:
            self.t_elapsed = self.t1s
            if self.close_plt_on_finish is True:
                self._graceful_exit()

            elif self.close_plt_on_finish is False:
                self._graceful_exit(plt_close=False)

            return None
