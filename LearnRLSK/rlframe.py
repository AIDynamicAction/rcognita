"""
Created on Tue Apr 21 15:54:06 2020

@author: Pavel Osinenko
"""


"""
=============================================================================
Reinforcement learning frame

This is a skeleton for reinforcement learning (RL) methods in Python ready for implementation of custom setups, 
e.g., value iteration, policy iteration, dual etc.

=============================================================================

Remark: 

All vectors are trated as of type [n,]
All buffers are trated as of type [L, n] where each row is a vector
Buffers are updated from bottom
"""

# imports
import os
import pathlib
import inspect
import warnings
import sys
from collections import namedtuple

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

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# LearnRLSK
from . import utilities

# other
from mpldatacursor import datacursor
from tabulate import tabulate



class System:
    """
    Class of continuous-time dynamical systems with input and dynamical disturbance for use with ODE solvers.
    In RL, this is considered the *environment*.
    Normally, you should pass `closed_loop`, which represents the right-hand side, to your solver.

    Parameters
    ----------
    dim_state, dim_input, dim_output, dim_disturb -- 
        * System dimensions
    
    m, I --
        * m = robot's mass
        * I = moment of inertia about the vertical axis

    f_min, f_max, m_min, m_max -- 
        * control bounds
    
    is_dyn_ctrl -- 
        * 0 or 1
        * If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    
    is_disturb --
        * 0 or 1
        * If 0, no disturbance is fed into the system
    
    sigma_q, mu_q, tau_q --
        * Parameters of the disturbance model

    """

    @classmethod
    def print_docstring(cls):
        print(cls.__doc__)

    @classmethod
    def print_init_params(cls):
        signature = inspect.signature(cls.__init__)
        for i, param in enumerate(signature.parameters.values()):
            if i == 0:
                pass
            else:
                print(param)

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 dim_disturb=2,
                 m=10,
                 I=1,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 is_dyn_ctrl=0,
                 is_disturb=0,
                 sigma_q=None,
                 mu_q=None,
                 tau_q=None):
        """system """
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb
        self.m = m
        self.I = I
        self.control_bounds = np.array([[f_min, f_max], [m_min, m_max]])

        """disturbance"""
        self.is_disturb = is_disturb
        self.sigma_q = sigma_q
        self.mu_q = mu_q
        self.tau_q = tau_q

        # Track system's state
        self._x = np.zeros(dim_state)

        # Current input (a.k.a. action)
        self.u = np.zeros(dim_input)

        # Static or dynamic controller
        self.is_dyn_ctrl = is_dyn_ctrl

        if is_dyn_ctrl:
            self._dim_full_state = self.dim_state + self.dim_disturb + self.dim_input
        else:
            self._dim_full_state = self.dim_state + self.dim_disturb

    @staticmethod
    def get_next_state(t, x, u, q, m, I, dim_state, is_disturb):
        """
        Right-hand side of the system internal dynamics

            x_t+1 = f(x_t, u_t, q_t)

        where:

            `x` : state
            `u` : input
            `q` : disturbance


        System description
        ------------------

        Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

        Variables:
            `x_с` : x-coordinate [m]
            `y_с` : y-coordinate [m]
            `\\alpha` : turning angle [rad]
            `v` : speed [m/s]
            `\\omega` : revolution speed [rad/s]
            `F` : pushing force [N]          
            `M` : steering torque [Nm]
            `m` : robot mass [kg]
            `I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
            `q` : actuator disturbance (see `System._add_disturbance`). Is zero if `is_disturb = 0`

            `x = [x_c, y_c, \\alpha, v, \\omega]`

            `u = [F, M]`

            `pars` = `[m, I]`

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
        next_state = np.zeros(dim_state)

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
        next_state[0] = x
        next_state[1] = y
        next_state[2] = alpha
        next_state[3] = v
        next_state[4] = omega

        return next_state

    def _add_disturbance(self, t, q):
        """
        Dynamical disturbance model:

            q = rho(q)


        System description
        ------------------ 

        `sigma_q, mu_q, tau_q`, with each being an array of shape `[dim_disturb, ]`

        """

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

    @staticmethod
    def get_curr_state(x, u=[]):
        """
        Return current state of system

        """
        y = x 
        return y

    def receive_action(self, u):
        """
        Receive control action from agent. 

        Parameters
        ----------
        u : array of shape `[dim_input, ]`

        """
        self.u = u

    def closed_loop(self, t, ksi):
        """
        Closed loop of the system.
        This function is designed for use with ODE solvers.
        Normally, you shouldn't change it

        Examples
        --------
        Assuming `sys` is a `system`-object, `t0, t1` - start and stop times, and `ksi0` - a properly defined initial condition:

        >>> import scipy as sp
        >>> simulator = sp.integrate.RK45(sys.closed_loop, t0, ksi0, t1)
        >>> while t < t1:
                simulator.step()
                t = simulator.t
                ksi = simulator.y
                x = ksi[0:sys.dim_state]
                y = sys.get_curr_state(x)
                u = myController(y)
                sys.receive_action(u)

        """

        full_state = np.zeros(self._dim_full_state)

        x = ksi[0:self.dim_state]
        q = ksi[self.dim_state:]

        if self.is_dyn_ctrl:
            u = ksi[-self.dim_input:]
            full_state[-self.dim_input:] = self._create_dyn_controller(t, u, y)
        else:
            # Fetch the control action stored in the system
            u = self.u

        if self.control_bounds.any():
            for k in range(self.dim_input):
                u[k] = np.clip(u[k], self.control_bounds[k, 0], self.control_bounds[k, 1])

        full_state[0:self.dim_state] = System.get_next_state(
            t, x, u, q, self.m, self.I, self.dim_state, self.is_disturb)

        if self.is_disturb:
            full_state[self.dim_state:] = self._add_disturbance(t, q)

        # Track system's state
        self._x = x

        return full_state


class Controller:
    """
    Optimal controller (a.k.a. agent) class.

    Parameters
    ----------
    dim_input, dim_output --
        * Dimension of input and output which should comply with the system-to-be-controlled

    t0 --
        * default = 0
        * Initial value of the controller's internal clock
    
    sample_time --
        * Controller's sampling time (in seconds)
    
    sys_rhs, sys_out --      
        * Functions that represents the right-hand side, resp., the output of the exogenously passed model.
        * The latter could be, for instance, the true model of the system.
        * In turn, `system_state` represents the (true) current state of the system and should be updated accordingly.
        * Parameters `sys_rhs, sys_out, system_state` are used in controller modes which rely on them.
    
    prob_noise_pow -- 
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control      
    
    mod_est_phase -- 
        * Initial phase to fill the estimator's buffer before applying optimal control (in seconds)      
    
    mod_est_period -- 
        * In seconds, the time between model estimate updates. This constant
        determines how often the estimated parameters are updated. The more
        often the model is updated, the higher the computational burden is.
        On the other hand, more frequent updates help keep the model actual. 
    
    buffer_size -- 
        * The size of the buffer to store data for model estimation. The bigger
        the buffer, the more accurate the estimation may be achieved. For
        successful model estimation, the system must be sufficiently excited.
        Using bigger buffers is a way to achieve this. 
    
    model_order --
        * The order of the state-space estimation model. We are interested in
        adequate predictions of y under given u's. The higher the model
        order, the better estimation results may be achieved, but be aware of
        overfitting         

        **See** `controller._estimate_model` . **This is just a particular model estimator.
        When customizing,** `controller._estimate_model`
        **may be changed and in turn the parameter** `model_order` **also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers
        of neurons, in which case** `model_order` **could be substituted for, say,** `Nlayers`, `Nneurons` 
    
    mod_est_checks --
        * Estimated model parameters can be stored in stacks and the best among the `mod_est_checks` last ones is picked.
        * May improve the prediction quality somewhat
    
    gamma --
        * number in (0, 1]
        * Discounting factor.
        * Characterizes fading of running costs along horizon
    
    n_actor --
        * Number of prediction steps. n_actor=1 means the controller is purely data-driven and doesn't use prediction.

    n_critic --
        * Critic stack size `N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer
    
    critic_period --
        * # Time between critic updates

    pred_step_size --
        * Prediction step size in `J` as defined above (in seconds). Should be a multiple of `sample_time`. Commonly, equals it, but here left adjustable for
        * convenience. Larger prediction step size leads to longer factual horizon
        
    r_cost_struct --
        * Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.
            Notation: chi = [y, u]
            1 - quadratic chi.T @ R1 @ chi
            2 - 4th order chi**2.T @ R2 @ chi**2 + chi.T @ R2 @ chi
            R1, R2 must be positive-definite

    critic_struct -- 
        * Choice of the structure of the critic's feature vector
           * 1 - Quadratic-linear
           * 2 - Quadratic
           * 3 - Quadratic, no mixed terms
           * 4 - Quadratic, no mixed terms in input and output

    n_critic --
        * Should not greater than buffer_size. The critic optimizes the temporal error which is a measure of critic's ability to capture the optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer. The principle here is pretty much the same as with the model estimation: accuracy against performance

    ctrl_mode --
        Modes with online model estimation are experimental
            
            0     - manual constant control (only for basic testing)
            -1    - nominal parking controller (for benchmarking optimal controllers)
            1     - model-predictive control (MPC). Prediction via discretized true model
            2     - adaptive MPC. Prediction via estimated model
            3     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via discretized true model
            4     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via estimated model
            5     - RL: stacked Q-learning. Prediction via discretized true model
            6     - RL: stacked Q-learning. Prediction via estimated model
            
        * Modes 1, 3, 5 use model for prediction, passed into class exogenously. This could be, for instance, a true system model
        * Modes 2, 4, 6 use am estimated online, see `controller.estimateModel` 

    f_min, f_max, m_min, m_max (ctrl_bnds) -- 
        * control bounds
        * Box control constraints. First element in each row is the lower bound, the second - the upper bound. If empty, control is unconstrained (default)

    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        

    """

    @classmethod
    def print_docstring(cls):
        print(cls.__doc__)

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 ctrl_mode=1,
                 initial_x=5,
                 initial_y=5,
                 m=10,
                 I=1,
                 t0=0,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 n_actor=1,
                 n_critic=4,
                 buffer_size=20,
                 critic_period=0.1,
                 critic_struct=1,
                 r_cost_struct=1,
                 sample_time=0.1,
                 mod_est_phase=1,
                 mod_est_period=0.1,
                 mod_est_checks=0,
                 model_order=3,
                 prob_noise_pow=1,
                 pred_step_size=0.1,
                 gamma=1,
                 is_disturb=0):
        """ system vars """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_state = dim_state
        self.m = m
        self.I = I

        """ disturbance """
        self.is_disturb = is_disturb

        # initial values of the system's state
        alpha = np.pi / 2
        initial_state = np.zeros(dim_state)
        initial_state[0] = initial_x
        initial_state[1] = initial_y
        initial_state[2] = alpha
        self.system_state = initial_state

        """ model estimator """
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.mod_est_phase = mod_est_phase
        self.mod_est_period = mod_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.mod_est_checks = mod_est_checks

        A = np.zeros([self.model_order, self.model_order])
        B = np.zeros([self.model_order, self.dim_input])
        C = np.zeros([self.dim_output, self.model_order])
        D = np.zeros([self.dim_output, self.dim_input])
        x0_est = np.zeros(self.model_order)

        self.my_model = utilities._model(A, B, C, D, x0_est)
        self.model_stack = []

        for k in range(self.mod_est_checks):
            self.model_stack.append(self.my_model)

        """ Controller  """

        self.n_actor = n_actor
        self.critic_period = critic_period
        self.pred_step_size = pred_step_size

        """ RL elements """
        self.r_cost_struct = r_cost_struct
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        self.r_cost_pars = [self.R1, self.R2]
        self.i_cost_val = 0
        self.critic_struct = critic_struct
        self.critic_clock = t0
        self.n_critic = n_critic
        self.n_critic = np.min([self.n_critic, self.buffer_size - 1])

        """control mode"""
        self.ctrl_mode = ctrl_mode
        self.ctrl_clock = t0
        self.sample_time = sample_time

        # manual control
        self.ctrl_bnds = np.array([[f_min, f_max], [m_min, m_max]])
        self.min_bounds = np.array(self.ctrl_bnds[:, 0])
        self.max_bounds = np.array(self.ctrl_bnds[:, 1])
        self.u_min = utilities._repMat(self.min_bounds, 1, n_actor)
        self.u_max = utilities._repMat(self.max_bounds, 1, n_actor)
        self.u_curr = self.min_bounds / 10
        self.u_init = utilities._repMat(self.min_bounds / 10, 1, self.n_actor)

        self.u_buffer = np.zeros([buffer_size, dim_input])
        self.y_buffer = np.zeros([buffer_size, dim_output])

        """ other """
        self.sys_rhs = System.get_next_state
        self.sys_out = System.get_curr_state

        # discount factor
        self.gamma = gamma

        if self.critic_struct == 1:
            self.dim_crit = ((self.dim_output + self.dim_input) + 1) * \
                (self.dim_output + self.dim_input) / \
                2 + (self.dim_output + self.dim_input)

            self.w_min = -1e3 * np.ones(int(self.dim_crit))
            self.w_max = 1e3 * np.ones(int(self.dim_crit))

        elif self.critic_struct == 2:
            self.dim_crit = ((self.dim_output + self.dim_input) + 1) * \
                (self.dim_output + self.dim_input) / 2
            self.w_min = np.zeros(self.dim_crit)
            self.w_max = 1e3 * np.ones(int(self.dim_crit))

        elif self.critic_struct == 3:
            self.dim_crit = self.dim_output + self.dim_input
            self.w_min = np.zeros(self.dim_crit)
            self.w_max = 1e3 * np.ones(int(self.dim_crit))

        elif self.critic_struct == 4:
            self.dim_crit = self.dim_output + self.dim_output * self.dim_input + self.dim_input
            self.w_min = -1e3 * np.ones(int(self.dim_crit))
            self.w_max = 1e3 * np.ones(int(self.dim_crit))

        self.Wprev = np.ones(int(self.dim_crit))

        self.Winit = self.Wprev

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained

        """
        self.ctrl_clock = t0
        self.u_curr = self.min_bounds / 10

    def receive_sys_state(self, x):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation

        """
        self.system_state = x

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

    def rcost(self, y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)

        See class documentation
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
        self.i_cost_val += self.rcost(y, u) * self.sample_time

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
                if (time_in_est_period >= mod_est_period) and (self.ctrl_mode in (2, 4, 6)):
                    # Update model estimator's internal clock
                    self.est_clock = t

                    try:
                        # Using ssid from Githug:AndyLamperski/pyN4SID
                        # Aid, Bid, Cid, Did, _ ,_ = ssid.N4SID(serf.u_buffer.T,  self.y_buffer.T,
                        #                                       NumRows = self.dim_input + self.model_order,
                        #                                       NumCols = self.buffer_size - (self.dim_input + self.model_order)*2,
                        #                                       NSig = self.model_order,
                        #                                       require_stable=False)
                        # self.my_model.updatePars(Aid, Bid, Cid, Did)

                        # Using Github:CPCLAB-UNIPI/SIPPY
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S,
                        # PARSIM-K
                        SSest = sippy.system_identification(self.y_buffer,
                                                            self.u_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.buffer_size/12),
                                                            # SS_p=int(self.buffer_size/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sample_time)

                        self.my_model.updatePars(
                            SSest.A, SSest.B, SSest.C, SSest.D)

                        # [EXPERIMENTAL] Using MATLAB's system identification toolbox
                        # us_ml = eng.transpose(matlab.double(self.u_buffer.tolist()))
                        # ys_ml = eng.transpose(matlab.double(self.y_buffer.tolist()))

                        # Aml, Bml, Cml, Dml = eng.mySSest_simple(ys_ml, us_ml, dt, model_order, nargout=4)

                        # self.my_model.updatePars(np.asarray(Aml), np.asarray(Bml), np.asarray(Cml), np.asarray(Dml) )

                    except:
                        print('Model estimation problem')
                        self.my_model.updatePars(np.zeros([self.model_order, self.model_order]),
                                                 np.zeros(
                            [self.model_order, self.dim_input]),
                            np.zeros(
                            [self.dim_output, self.model_order]),
                            np.zeros([self.dim_output, self.dim_input]))

                    # Model checks
                    if self.mod_est_checks > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        totAbsErrCurr = 1e8
                        for k in range(self.mod_est_checks):
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

            if t >= self.mod_est_phase:
                    # Drop probing noise
                self.is_prob_noise = 0

    def _phi(self, y, u):
        """
        Feature vector of critic

        In Q-learning mode, it uses both `y` and `u`. In value function approximation mode, it should use just `y`

        Customization
        -------------

        Adjust this method if you still sitck with a linearly parametrized approximator for Q-function, value function etc.
        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like `W @ self._phi( y, u )` 
        within `controller._critic_cost`

        """
        chi = np.concatenate([y, u])

        if self.critic_struct == 1:
            return np.concatenate([_uptria2vec(np.kron(chi, chi)), chi])

        elif self.critic_struct == 2:
            return np.concatenate([_uptria2vec(np.kron(chi, chi))])

        elif self.critic_struct == 3:
            return chi * chi

        elif self.critic_struct == 4:
            return np.concatenate([y**2, np.kron(y, u), u**2])

    def _critic_cost(self, W, U, Y):
        """
        Cost function of the critic

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
            e = W @ self._phi(y_prev, u_prev) - self.gamma * self.Wprev @ self._phi(y_next, u_next) - self.rcost(y_prev, u_prev)

            Jc += 1 / 2 * e**2

        return Jc

    def _critic(self, Wprev, Winit, U, Y):
        """
        See class documentation. Parameter `delta` here is a shorthand for `pred_step_size`

        Customization
        -------------

        This method normally should not be altered, adjust `controller._critic_cost` instead.
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
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False,
                                  'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}  # 'disp': True, 'verbose': 2}

        bnds = sp.optimize.Bounds(self.w_min, self.w_max, keep_feasible=True)

        W = minimize(lambda W: self._critic_cost(W, U, Y), Winit,
                     method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x


        return W

    def _actor_cost(self, U, y, N, W, delta, ctrl_mode):
        """
        See class documentation. Parameter `delta` here is a shorthand for `pred_step_size`

        Customization
        -------------        

        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """

        myU = np.reshape(U, [N, self.dim_input])

        Y = np.zeros([N, self.dim_output])

        # System output prediction
        if (ctrl_mode == 1) or (ctrl_mode == 3) or (ctrl_mode == 5):    # Via exogenously passed model
            Y[0, :] = y
            x = self.system_state
            for k in range(1, self.n_actor):
                # Euler scheme
                x = x + delta * \
                    self.sys_rhs([], x, myU[k - 1, :], [], self.m,
                                 self.I, self.dim_state, self.is_disturb)
                Y[k, :] = self.sys_out(x)

        elif (ctrl_mode == 2) or (ctrl_mode == 4) or (ctrl_mode == 6):    # Via estimated model
            myU_upsampled = myU.repeat(int(delta / self.sample_time), axis=0)
            Yupsampled, _ = self._dss_sim(
                self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0_est, y)
            Y = Yupsampled[::int(delta / self.sample_time)]

        J = 0
        if (ctrl_mode == 1) or (ctrl_mode == 2):     # MPC
            for k in range(N):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
        # RL: Q-learning with n_critic-1 roll-outs of running cost
        elif (ctrl_mode == 3) or (ctrl_mode == 4):
            for k in range(N - 1):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
            J += W @ self._phi(Y[-1, :], myU[-1, :])
        elif (ctrl_mode == 5) or (ctrl_mode == 6):     # RL: (normalized) stacked Q-learning
            for k in range(N):
                Q = W @ self._phi(Y[k, :], myU[k, :])
                J += 1 / N * Q

        return J

    def _actor(self, y, u_init, N, W, delta, ctrl_mode):
        """
        See class documentation. Parameter `delta` here is a shorthand for `pred_step_size`

        Customization
        -------------         

        This method normally should not be altered, adjust `controller._actor_cost`, `controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            # 'disp': True, 'verbose': 2}
            actor_opt_options = {'maxiter': 300, 'disp': False}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False,
                                 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}  # 'disp': True, 'verbose': 2}

        isGlobOpt = 0

        myu_init = np.reshape(u_init, [N * self.dim_input, ])

        bnds = sp.optimize.Bounds(self.u_min, self.u_max, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {
                    'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-7, 'options': actor_opt_options}
                U = basinhopping(lambda U: self._actor_cost(
                    U, y, N, W, delta, ctrl_mode), myu_init, minimizer_kwargs=minimizer_kwargs, niter=10).x
            else:
                U = minimize(lambda U: self._actor_cost(U, y, N, W, delta, ctrl_mode), myu_init,
                             method=actor_opt_method, tol=1e-7, bounds=bnds, options=actor_opt_options).x
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myu_init


        return U[:self.dim_input]    # Return first action

    def compute_action(self, t, y):
        """
        Main method. See class documentation

        Customization
        -------------         

        Add your modes, that you introduced in `controller._actor_cost`, here

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            if self.ctrl_mode in (1, 2):

                # Apply control when model estimation phase is over
                if self.is_prob_noise and (self.ctrl_mode == 2):
                    return self.prob_noise_pow * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and (self.ctrl_mode == 2):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    [], self.pred_step_size, self.ctrl_mode)

                elif (self.ctrl_mode == 1):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    [], self.pred_step_size, self.ctrl_mode)

            elif self.ctrl_mode in (3, 4, 5, 6):
                # Critic
                time_in_critic_period = t - self.critic_clock

                # Update data buffers
                self.u_buffer = utilities._pushVec(self.u_buffer, self.u_curr)
                self.y_buffer = utilities._pushVec(self.y_buffer, y)

                if time_in_critic_period >= self.critic_period:
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
                    u = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and (self.ctrl_mode in (4, 6)):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    W, self.pred_step_size, self.mode)


                elif self.ctrl_mode in (3, 5):
                    u = self._actor(y, self.u_init, self.n_actor,
                                    W, self.pred_step_size, self.ctrl_mode)

            self.u_curr = u

            return u

        else:
            return self.u_curr


class NominalController:
    """
    This is a class of nominal controllers used for benchmarking of optimal controllers.
    Specification should be provided for each individual case (system)

    The controller is sampled.

    For a three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Parameters
    ----------
    m, I --
        * Mass and moment of inertia around vertical axis of the robot
    ctrl_gain --
        * Controller gain       
    t0 --
        * Initial value of the controller's internal clock
    sample_time --
        * Controller's sampling time (in seconds)        

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self, m=10, I=1, ctrl_gain=10, f_min=-5, f_max=5, m_min=-1, m_max=1, t0=0, sample_time=0.1):
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
        Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)

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
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

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


class Simulation:
    """class to create and run simulation."""

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dimDisturb=2,
                 initial_x=5,
                 initial_y=5,
                 t0=0,
                 t1=100,
                 n_runs=1,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10,
                 dt=0.05,
                 f_man=-3,
                 n_man=-1,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 is_log_data=0,
                 is_visualization=1,
                 is_print_sim_step=1,
                 is_dyn_ctrl=0,
                 ctrl_mode=5):
        """system """
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dimDisturb = dimDisturb

        """simulation"""
        # start time of episode
        self.t0 = t0

        # stop time of episode
        self.t1 = t1

        # number of episodes
        self.n_runs = n_runs

        self.initial_x = initial_x
        self.initial_y = initial_y
        self.alpha = np.pi / 2

        # initial values of state
        initial_state = np.zeros(dim_state)
        initial_state[0] = initial_x
        initial_state[1] = initial_y
        initial_state[2] = self.alpha
        self.system_state = initial_state

        # initial value of control
        self.u0 = np.zeros(dim_input)

        # initial value of disturbance
        self.q0 = np.zeros(dimDisturb)

        # sensitivity of the solver. The lower the values, the more accurate the simulation results are
        self.a_tol = a_tol
        self.r_tol = r_tol

        # x and y limits of scatter plot. Used so far rather for visualization only, but may be integrated into the actor as constraints
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        """ controller sampling time.
        The system itself is continuous as a physical process while the controller is digital.

            Things to note:
                * the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
                * smaller sampling times lead to higher computation times
                * especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with dt and try achieve a trade-off between stability and computational performance
        """
        self.dt = dt

        """control mode
        
            Modes with online model estimation are experimental
            
            0     - manual constant control (only for basic testing)
            -1    - nominal parking controller (for benchmarking optimal controllers)
            1     - model-predictive control (MPC). Prediction via discretized true model
            2     - adaptive MPC. Prediction via estimated model
            3     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via discretized true model
            4     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via estimated model
            5     - RL: stacked Q-learning. Prediction via discretized true model
            6     - RL: stacked Q-learning. Prediction via estimated model
        """
        self.ctrl_mode = ctrl_mode

        # manual control
        self.f_man = f_man
        self.n_man = n_man
        self.u_man = np.array([f_man, n_man])

        # control constraints
        self.f_min = f_min
        self.f_max = f_max
        self.m_min = m_min
        self.m_max = m_max

        self.control_bounds = np.array(
            [[self.f_min, self.f_max], [self.m_min, self.m_max]])

        """Other"""
        #%% User settings: main switches
        self.is_log_data = is_log_data
        self.is_visualization = is_visualization
        self.is_print_sim_step = is_print_sim_step

        # Static or dynamic controller
        self.is_dyn_ctrl = is_dyn_ctrl

        if self.is_dyn_ctrl:
            self.ksi0 = np.concatenate([self.system_state, self.q0, self.u0])
        else:
            self.ksi0 = np.concatenate([self.system_state, self.q0])

        # extras
        self.data_files = self._logdata(self.n_runs, save=self.is_log_data)

        if self.is_print_sim_step:
            warnings.filterwarnings('ignore')

    def _ctrlSelector(self, t, y, uMan, nominalCtrl, agent, mode):
        """
        Main interface for different agents

        """
        
        if mode==0: # Manual control
            u = uMan
        elif mode==-1: # Nominal controller
            u = nominalCtrl.compute_action(t, y)
        elif mode > 0: # Optimal controller
            u = agent.compute_action(t, y)
            
        return u

    def create_simulator(self, closed_loop):
        simulator = sp.integrate.RK45(closed_loop,
                                      self.t0,
                                      self.ksi0,
                                      self.t1,
                                      max_step=self.dt / 2,
                                      first_step=1e-6,
                                      atol=self.a_tol,
                                      rtol=self.r_tol)
        return simulator

    def _create_figure(self, agent):
        y0 = System.get_curr_state(self.system_state)
        alpha_deg0 = self.alpha / 2 / np.pi

        plt.close('all')

        self.sim_fig = plt.figure(figsize=(10, 10))

        # xy plane
        self.xy_plane_axes = self.sim_fig.add_subplot(221,
                                                      autoscale_on=False,
                                                      xlim=(self.x_min,
                                                            self.x_max),
                                                      ylim=(self.y_min,
                                                            self.y_max),
                                                      xlabel='x [m]',
                                                      ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')

        self.xy_plane_axes.set_aspect('equal', adjustable='box')
        self.xy_plane_axes.plot([self.x_min, self.x_max], [
            0, 0], 'k--', lw=0.75)   # Help line
        self.xy_plane_axes.plot([0, 0], [self.y_min, self.y_max],
                                'k--', lw=0.75)   # Help line
        self.traj_line, = self.xy_plane_axes.plot(
            self.initial_x, self.initial_y, 'b--', lw=0.5)
        self.robot_marker = utilities._pltMarker(angle=alpha_deg0)

        text_time = 't = {time:2.3f}'.format(time=self.t0)

        self.text_time_handle = self.xy_plane_axes.text(0.05, 0.95,
                                                        text_time,
                                                        horizontalalignment='left',
                                                        verticalalignment='center',
                                                        transform=self.xy_plane_axes.transAxes)

        self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        # Solution
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')
        self.sol_axes.plot([self.t0, self.t1], [0, 0],
                           'k--', lw=0.75)   # Help line
        self.norm_line, = self.sol_axes.plot(self.t0, la.norm(
            [self.initial_x, self.initial_y]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.alpha_line, = self.sol_axes.plot(
            self.t0, self.alpha, 'r-', lw=0.5, label=r'$\alpha$ [rad]')
        self.sol_axes.legend(fancybox=True, loc='upper right')
        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        # Cost
        self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            0, 1e4 * agent.rcost(y0, self.u0)), yscale='symlog', xlabel='t [s]')

        r = agent.rcost(y0, self.u0)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=0)
        self.text_icost_handle = self.sim_fig.text(
            0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')
        self.r_cost_line, = self.cost_axes.plot(
            self.t0, r, 'r-', lw=0.5, label='r')
        self.i_cost_line, = self.cost_axes.plot(
            self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.cost_axes.legend(fancybox=True, loc='upper right')

        # Control
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')
        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line
        self.ctrl_lines = self.ctrlAxs.plot(
            self.t0, utilities._toColVec(self.u0).T, lw=0.5)
        self.ctrlAxs.legend(
            iter(self.ctrl_lines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        # Pack all lines together
        cLines = namedtuple('lines', [
                            'traj_line', 'norm_line', 'alpha_line', 'r_cost_line', 'i_cost_line', 'ctrl_lines'])
        self.lines = cLines(traj_line=self.traj_line,
                            norm_line=self.norm_line,
                            alpha_line=self.alpha_line,
                            r_cost_line=self.r_cost_line,
                            i_cost_line=self.i_cost_line,
                            ctrl_lines=self.ctrl_lines)

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

        return self.sim_fig

    def _initialize_figure(self):
        self.sol_scatter = self.xy_plane_axes.scatter(
            self.initial_x, self.initial_y, marker=self.robot_marker.marker, s=400, c='b')
        self.current_run = 1

        return self.sol_scatter

    def _update_line(self, line, newX, newY):
        line.set_xdata(np.append(line.get_xdata(), newX))
        line.set_ydata(np.append(line.get_ydata(), newY))

    def _reset_line(self, line):
        line.set_data([], [])

    # def _update_scatter(self, scatter, newX, newY):
    #     scatter.set_offsets(
    #         np.vstack([scatter.get_offsets().data, np.c_[newX, newY]]))

    def _update_text(self, text_handle, newText):
        text_handle.set_text(newText)

    def _update_scatter(self, text_time, ksi, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u):
        self._update_text(self.text_time_handle, text_time)
        # Update the robot's track on the plot
        self._update_line(self.traj_line, *ksi[:2])

        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot
        self.sol_scatter.remove()
        self.sol_scatter = self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_marker.marker, s=400, c='b')

        # Solution
        self._update_line(self.norm_line, t, la.norm([x_coord, y_coord]))
        self._update_line(self.alpha_line, t, alpha)

        # Cost
        self._update_line(self.r_cost_line, t, r)
        self._update_line(self.i_cost_line, t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handle, text_icost)
        # Control
        for (line, uSingle) in zip(self.ctrl_lines, u):
            self._update_line(line, t, uSingle)

    def _reset_sim(self, agent, nominal_ctrl, simulator):
        if self.is_print_sim_step:
            print('.....................................Run {run:2d} done.....................................'.format(
                run=self.current_run))

        self.current_run += 1

        if self.current_run > self.Nruns:
            return

        if self.is_log_data:
            self.current_data_file = self.data_files[self.current_run - 1]

        # Reset simulator
        simulator.status = 'running'
        simulator.t = self.t0
        simulator.y = self.ksi0

        # Reset controller
        if self.ctrl_mode > 0:
            agent.reset(self.t0)
        else:
            nominal_ctrl.reset(self.t0)

    def _onKeyPress(self, event, anm):
        if event.key == ' ':
            if anm.running is True:
                anm.event_source.stop()
                anm.running = False
                
            elif anm.running is False:
                anm.event_source.start()
                anm.running = True
            
        elif event.key == 'q':
            plt.close('all')
            print("Program exit")
            os._exit(1)

    def _logDataRow(self, dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
        with open(dataFile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]])

    def _logdata(self, Nruns, save=False):
        dataFiles = [None] * Nruns

        if save:
            cwd = os.getcwd()
            datafolder = '/data'
            dataFolder_path = cwd + datafolder
            
            # create data dir
            pathlib.Path(dataFolder_path).mkdir(parents=True, exist_ok=True) 

            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%Hh%Mm%Ss")
            dataFiles = [None] * Nruns
            for k in range(0, Nruns):
                dataFiles[k] = dataFolder_path + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
                with open(dataFiles[k], 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]'] )

        return dataFiles

    def _printSimStep(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180      
        
        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]']  
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]  
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
        
        print(table)

    def _wrapper_take_steps(self, k, *args):
        return self._take_step(*args)

    def _take_step(self, sys, agent, nominal_ctrl, simulator, animate=False):
        # take step
        simulator.step()

        t = simulator.t
        ksi = simulator.y

        x = ksi[0:self.dim_state]
        y = sys.get_curr_state(x)

        u = self._ctrlSelector(
            t, y, self.u_man, nominal_ctrl, agent, self.ctrl_mode)

        sys.receive_action(u)
        agent.receive_sys_state(sys._x)
        agent.update_icost(y, u)

        x_coord = ksi[0]
        y_coord = ksi[1]
        alpha = ksi[2]
        v = ksi[3]
        omega = ksi[4]
        icost = agent.i_cost_val

        if self.is_print_sim_step:
            self._printSimStep(t, x_coord, y_coord, alpha, v, omega, icost, u)

        if self.is_log_data:
            self._logDataRow(self.current_data_file, t, x_coord,
                       y_coord, alpha, v, omega, icost.val, u)

        if animate == True:
            alpha_deg = alpha / np.pi * 180
            r = agent.rcost(y, u)
            text_time = 't = {time:2.3f}'.format(time=t)
            self._update_scatter(text_time, ksi, alpha_deg,
                                 x_coord, y_coord, t, alpha, r, icost, u)

        # Run done
        if t >= self.t1:
            self._reset_sim(agent, nominal_ctrl, simulator)
            icost = 0

            for item in self.lines:
                if item != self.traj_line:
                    if isinstance(item, list):
                        for subitem in item:
                            self._reset_line(subitem)
                    else:
                        self._reset_line(item)

            self._update_line(self.traj_line, np.nan, np.nan)

        if animate == True:
            return self.sol_scatter

    def run_simulation(self, sys, agent, nominal_ctrl, simulator):
        if self.is_visualization == 0:
            self.current_run = 1
            self.current_data_file = data_files[0]

            while True:
                self._take_step(sys, agent, nominal_ctrl, simulator)

        else:
            self.sim_fig = self._create_figure(agent)

            animate = True
            fargs = (sys, agent, nominal_ctrl, simulator, animate)
            anm = animation.FuncAnimation(self.sim_fig,
                                          self._wrapper_take_steps,
                                          fargs=fargs,
                                          init_func=self._initialize_figure,
                                          interval=1)

            anm.running = True
            self.sim_fig.canvas.mpl_connect(
                'key_press_event', lambda event: self._onKeyPress(event, anm))
            self.sim_fig.tight_layout()
            plt.show()
