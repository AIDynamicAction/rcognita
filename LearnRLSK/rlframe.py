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

# scipy
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import basinhopping

# numpy
import numpy as np
from numpy.random import rand
from numpy.random import randn
from scipy import signal
import sippy  # Github:CPCLAB-UNIPI/SIPPY

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# LearnRLSK
from .utilities import *

# other
import warnings
import sys
from collections import namedtuple
from mpldatacursor import datacursor


class System:
    """
    Class of continuous-time dynamical systems with exogenous input and dynamical disturbance for use with ODE solvers.
    In RL, this is considered the *environment*.
    Normally, you should pass ``closedLoop``, which represents the right-hand side, to your solver.

    Attributes
    ----------
    dim_state, dim_input, dim_output, dim_disturb : : integer
        System dimensions 
    pars : : list
        List of fixed parameters of the system
    ctrlBnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    is_dyn_ctrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    is_disturb : : 0 or 1
        If 0, no disturbance is fed into the system
    sigma_q, mu_q, tau_q : : list
        Parameters of the disturbance model

    Customization
    -------------        

    Change specification of ``stateDyn, out, disturbDyn``.
    Set up dimensions properly and use the parameters ``pars`` and ``parsDisturb`` in accordance with your system specification

    """

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
        """system - needs description"""
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb
        self.m = m
        self.I = I
        self.ctrlBnds = np.array([[f_min, f_max], [m_min, m_max]])

        """disturbance - needs description"""
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
    def stateDyn(t, x, u, q, m, I, dim_state, is_disturb):
        """
        Right-hand side of the system internal dynamics

        .. math:: \mathcal D x = f(x, u, q),

        where:

        | :math:`x` : state
        | :math:`u` : input
        | :math:`q` : disturbance

        The time variable ``t`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is
        non-autonomous.
        For the latter case, however, you already have the input and disturbance at your disposal

        Parameters of the system are contained in ``pars`` attribute.
        Make a proper use of them here

        Normally, you should not call this method directly, but rather :func:`~RLframe.system.closedLoop` from your ODE solver and, respectively,
        :func:`~RLframe.system.sysOut` from your controller

        System description
        ------------------
        **Describe your system specification here**

        Three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_]

        .. math::
            \\begin{array}{ll}
                        \dot x_с & = v \cos \\alpha \\newline
                        \dot y_с & = v \sin \\alpha \\newline
                        \dot \\alpha & = \\omega \\newline
                        \dot v & = \\left( \\frac 1 m F + q_1 \\right) \\newline
                        \dot \\omega & = \\left( \\frac 1 I M + q_2 \\right)
            \\end{array}

        **Variables**

        | :math:`x_с` : x-coordinate [m]
        | :math:`y_с` : y-coordinate [m]
        | :math:`\\alpha` : turning angle [rad]
        | :math:`v` : speed [m/s]
        | :math:`\\omega` : revolution speed [rad/s]
        | :math:`F` : pushing force [N]          
        | :math:`M` : steering torque [Nm]
        | :math:`m` : robot mass [kg]
        | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
        | :math:`q` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``

        :math:`x = [x_c, y_c, \\alpha, v, \\omega]`

        :math:`u = [F, M]`

        ``pars`` = :math:`[m, I]`

        References
        ----------
        .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
            nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

        """

        Dx = np.zeros(dim_state)
        Dx[0] = x[3] * np.cos(x[2])
        Dx[1] = x[3] * np.sin(x[2])
        Dx[2] = x[4]

        if is_disturb:
            Dx[3] = 1 / m * (u[0] + q[0])
            Dx[4] = 1 / I * (u[1] + q[1])
        else:
            Dx[3] = 1 / m * u[0]
            Dx[4] = 1 / I * u[1]

        return Dx

    def _disturbDyn(self, t, q):
        """
        Dynamical disturbance model:

        .. math:: \mathcal D q = \\rho(q),    


        System description
        ------------------ 
        **Describe your system specification here**

        We use here a 1st-order stochastic linear system of the type

        .. math:: \mathrm d Q_t = - \\frac{1}{\\tau_q} \\left( Q_t \\mathrm d t + \\sigma_q ( \\mathrm d B_t + \\mu_q ) \\right) ,

        where :math:`B` is the standard Brownian motion, :math:`Q` is the stochastic process whose realization is :math:`q`, and
        :math:`\\tau_q, \\sigma_q, \\mu_q` are the time constant, standard deviation and mean, resp.

        ``sigma_q, mu_q, tau_q``, with each being an array of shape ``[dim_disturb, ]``

        """

        Dq = np.zeros(self.dim_disturb)

        if self.is_disturb:
            for k in range(0, self.dim_disturb):
                Dq[k] = - tau_q[k] * (q[k] + sigma_q[k] * (randn() + mu_q[k]))

        return Dq

    def _ctrlDyn(t, u, y):
        """
        Dynamical controller. When ``is_dyn_ctrl=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.

        Controller description
        ---------------------- 
        **Provide your specification of a dynamical controller here**

        Currently, left for future implementation    

        """

        Du = np.zeros(self.dim_input)

        return Du

    @staticmethod
    def out(x, u=[]):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``x`` since no physical processes transmit input to output instantly

        System description
        ------------------ 
        **Describe your system specification here**

        In a three-wheel robot specified here, we measure the full state vector, which means the system be equipped with position sensors along with
        force and torque sensors

        See also
        --------
        :func:`~RLframe.system.stateDyn`

        """
        # y = x[:3] + measNoise # <-- Measure only position and orientation
        y = x  # <-- Position, force and torque sensors on
        return y

    def receiveAction(self, u):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~RLframe.system.sysOut` 

        Parameters
        ----------
        u : : array of shape ``[dim_input, ]``
            Action

        Examples
        --------
        Assuming ``sys`` is a ``system``-object, ``t0, t1`` - start and stop times, and ``ksi0`` - a properly defined initial condition:

        >>> import scipy as sp
        >>> simulator = sp.integrate.RK45(sys.closedLoop, t0, ksi0, t1)
        >>> while t < t1:
                simulator.step()
                t = simulator.t
                ksi = simulator.y
                x = ksi[0:sys.dim_state]
                y = sys.out(x)
                u = myController(y)
                sys.receiveAction(u)

        """
        self.u = u

    def closedLoop(self, t, ksi):
        """
        Closed loop of the system.
        This function is designed for use with ODE solvers.
        Normally, you shouldn't change it

        Examples
        --------
        Assuming ``sys`` is a ``system``-object, ``t0, t1`` - start and stop times, and ``ksi0`` - a properly defined initial condition:

        >>> import scipy as sp
        >>> simulator = sp.integrate.RK45(sys.closedLoop, t0, ksi0, t1)
        >>> while t < t1:
                simulator.step()
                t = simulator.t
                ksi = simulator.y
                x = ksi[0:sys.dim_state]
                y = sys.out(x)
                u = myController(y)
                sys.receiveAction(u)

        """

        # DEBUG ===============================================================
        # print('INTERNAL t = {time:2.3f}'.format(time=t))
        # /DEBUG ==============================================================

        DfullState = np.zeros(self._dim_full_state)

        x = ksi[0:self.dim_state]
        q = ksi[self.dim_state:]

        if self.is_dyn_ctrl:
            u = ksi[-self.dim_input:]
            DfullState[-self.dim_input:] = self._ctrlDyn(t, u, y)
        else:
            # Fetch the control action stored in the system
            u = self.u

        if self.ctrlBnds.any():
            for k in range(self.dim_input):
                u[k] = np.clip(u[k], self.ctrlBnds[k, 0], self.ctrlBnds[k, 1])

        DfullState[0:self.dim_state] = System.stateDyn(
            t, x, u, q, self.m, self.I, self.dim_state, self.is_disturb)

        if self.is_disturb:
            DfullState[self.dim_state:] = self._disturbDyn(t, q)

        # Track system's state
        self._x = x

        return DfullState


class Controller:
    """
    Optimal controller (a.k.a. agent) class.

    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled
    mode : : natural number
        Controller mode. Currently available (:math:`r` is the running cost, :math:`\\gamma` is the discounting factor):

        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1

           * - Mode
             - Cost function
           * - 1, 2 - Model-predictive control (MPC)
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right)=\\sum_{k=1}^{N_a} \\gamma^{k-1} r(y_k, u_k)`
           * - 3, 4 - RL/ADP via :math:`N_a-1` roll-outs of :math:`r`
             - :math:`J \\left( y_1, \\{u\}_{1}^{N_a}\\right) =\\sum_{k=1}^{N_a-1} \\gamma^{k-1} r(y_k, u_k) + \\hat Q(y_{N_a}, u_{N_a})` 
           * - 5, 6 - RL/ADP via normalized stacked Q-learning [[1]_]
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right) =\\frac{1}{N_a} \\sum_{k=1}^{N_a-1} \\hat Q(y_{N_a}, u_{N_a})`               

        Modes 1, 3, 5 use model for prediction, passed into class exogenously. This could be, for instance, a true system model

        Modes 2, 4, 6 use am estimated online, see :func:`~RLframe.controller.estimateModel` 

        **Add your specification into the table when customizing the agent**    

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    t0 : : number
        Initial value of the controller's internal clock
    sampl_time : : number
        Controller's sampling time (in seconds)
    n_actor : : natural number
        Size of prediction horizon :math:`N_a` 
    pred_step_size : : number
        Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampl_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon
    sys_rhs, sys_out : : functions        
        Functions that represents the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``xSys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, xSys`` are used in controller modes which rely on them.
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control      
    mod_est_phase : : number
        Initial phase to fill the estimator's buffer before applying optimal control (in seconds)      
    mod_est_period : : number
        Time between model estimate updates (in seconds)
    buffer_size : : natural number
        Size of the buffer to store data
    model_order : : natural number
        Order of the state-space estimation model

        .. math::
            \\begin{array}{ll}
                \\hat x^+ & = A \\hat x + B u \\newline
                y^+  & = C \\hat x + D u,
            \\end{array}             

        **See** :func:`~RLframe.controller._estimateModel` . **This is just a particular model estimator.
        When customizing,** :func:`~RLframe.controller._estimateModel`
        **may be changed and in turn the parameter** ``model_order`` **also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers
        of neurons, in which case** ``model_order`` **could be substituted for, say,** ``Nlayers``, ``Nneurons`` 
    mod_est_checks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``mod_est_checks`` last ones is picked.
        May improve the prediction quality somewhat
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of running costs along horizon
    n_critic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer
    critic_period : : number
        The same meaning as ``mod_est_period`` 
    critic_struct : : natural number
        Choice of the structure of the critic's feature vector

        Currently available:

        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1

           * - Mode
             - Structure
           * - 1
             - Quadratic-linear
           * - 2
             - Quadratic
           * - 3
             - Quadratic, no mixed terms
           * - 4
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector

        **Add your specification into the table when customizing the critic** 
    r_cost_struct : : natural number
        Choice of the running cost structure.

        Currently available:

        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1

           * - Mode
             - Structure
           * - 1
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``r_cost_pars`` should be ``[R1]``
           * - 2
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``r_cost_pars``
               should be ``[R1, R2]``           

        **Pass correct running cost parameters in** ``r_cost_pars`` **(as a list)**

        **When customizing the running cost, add your specification into the table above**

    Examples
    ----------

    Assuming ``sys`` is a ``system``-object, ``t0, t1`` - start and stop times, and ``ksi0`` - a properly defined initial condition:

    >>> import scipy as sp
    >>> simulator = sp.integrate.RK45(sys.closedLoop, t0, ksi0, t1)
    >>> agent = controller(sys.dim_input, sys.dim_output)

    >>> while t < t1:
            simulator.step()
            t = simulator.t
            ksi = simulator.y
            x = ksi[0:sys.dim_state]
            y = sys.out(x)
            u = agent.computeAction(t, y)
            sys.receiveAction(u)
            agent.update_icost(y, u)

    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        

    """

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 ctrl_mode=1,
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
                 sampl_time=0.1,
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

        # initial values of state
        self.x0 = np.zeros(dim_state)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2
        self.xSys = self.x0

        """ model estimator """
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow

        # In seconds, an initial phase to fill the estimator's buffer before
        # applying optimal control
        self.mod_est_phase = mod_est_phase

        # In seconds, the time between model estimate updates. This constant
        # determines how often the estimated parameters are updated. The more
        # often the model is updated, the higher the computational burden is.
        # On the other hand, more frequent updates help keep the model actual.
        self.mod_est_period = mod_est_period

        # The size of the buffer to store data for model estimation. The bigger
        # the buffer, the more accurate the estimation may be achieved. For
        # successful model estimation, the system must be sufficiently excited.
        # Using bigger buffers is a way to achieve this.
        self.buffer_size = buffer_size

        # The order of the state-space estimation model. We are interested in
        # adequate predictions of y under given u's. The higher the model
        # order, the better estimation results may be achieved, but be aware of
        # overfitting
        self.model_order = model_order

        # Estimated model parameters can be stored in stacks and the best among
        # the mod_est_checks last ones is picked
        self.mod_est_checks = mod_est_checks

        A = np.zeros([self.model_order, self.model_order])
        B = np.zeros([self.model_order, self.dim_input])
        C = np.zeros([self.dim_output, self.model_order])
        D = np.zeros([self.dim_output, self.dim_input])
        x0_est = np.zeros(self.model_order)
        self.my_model = model(A, B, C, D, x0_est)
        self.model_stack = []

        for k in range(self.mod_est_checks):
            self.model_stack.append(self.my_model)

        """ Controller 

            # u[0]: Pushing force F [N]
            # u[1]: Steering torque M [N m]
        """

        # Number of prediction steps. n_actor=1 means the controller is purely
        # data-driven and doesn't use prediction.
        self.n_actor = n_actor

        # Time between critic updates
        self.critic_period = critic_period

        # In seconds. Should be a multiple of dt
        self.pred_step_size = pred_step_size

        """ RL elements

            Running cost. 

            Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.

            Notation: chi = [y, u]
            1 - quadratic chi.T R1 chi
            2 - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
            R1, R2 must be positive-definite
        """
        self.r_cost_struct = r_cost_struct
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        # R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
        # R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
        # R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1],
        # [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        # R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
        # R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1],
        # [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi
        self.r_cost_pars = [self.R1, self.R2]
        self.i_cost_val = 0

        """critic structure

            1 - quadratic-linear
            2 - quadratic
            3 - quadratic, no mixed terms
            4 - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...]
            # u[0]^2 + ...
        """
        self.critic_struct = critic_struct
        self.critic_clock = t0

        """Critic stack size.

            Should not greater than buffer_size. The critic optimizes the temporal error which is a measure of critic's ability to capture the optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer. The principle here is pretty much the same as with the model estimation: accuracy against performance

        """
        self.n_critic = n_critic
        
        # Clip critic buffer size
        self.n_critic = np.min([self.n_critic, self.buffer_size - 1])

        """control mode
        
            Modes with online model estimation are experimental
            
            0     - manual constant control (only for basic testing)
            -1    - nominal parking controller (for benchmarking optimal controllers)
            1     - model-predictive control (MPC). Prediction via discretized true model
            2     - adaptive MPC. Prediction via estimated model
            3     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via discretized true model
            4     - RL: Q-learning with n_critic roll-outs of running cost. Prediction via estimated model
            5     - RL: stacked Q-learning. Prediction via discretized true model
            6     - RL: stacked Q-learning. Prediction via estimated model
        """
        self.ctrl_mode = ctrl_mode
        self.ctrl_clock = t0
        self.sampl_time = sampl_time

        # manual control
        self.ctrl_bnds = np.array([[f_min, f_max], [m_min, m_max]])
        self.min_bounds = np.array(self.ctrl_bnds[:, 0])
        self.max_bounds = np.array(self.ctrl_bnds[:, 1])
        self.u_min = repMat(self.min_bounds, 1, n_actor)
        self.u_max = repMat(self.max_bounds, 1, n_actor)
        self.u_curr = self.min_bounds / 10
        self.u_init = repMat(self.min_bounds / 10, 1, self.n_actor)
        
        self.u_buffer = np.zeros([buffer_size, dim_input])
        self.y_buffer = np.zeros([buffer_size, dim_output])


        """ other """
        self.sys_rhs = System.stateDyn
        self.sys_out = System.out

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

    def receiveSysState(self, x):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation

        """
        self.xSys = x

    def _dssSim(self, A, B, C, D, uSqn, x0, y0):
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
        If the agent succeeded to stabilize the system, ``icost`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)

        """
        self.i_cost_val += self.rcost(y, u) * self.sampl_time

    def _estimateModel(self, t, y):
        """
        Estimate model parameters by accumulating data buffers ``u_buffer`` and ``y_buffer``

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampl_time:  # New sample
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
                                                            tsample=self.sampl_time)

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
                            y_est, _ = self._dssSim(
                                A, B, C, D, self.u_buffer, x0_est, y)
                            meanErr = np.mean(y_est - self.y_buffer, axis=0)

                            # DEBUG ===========================================
                            # ================================Interm output of model prediction quality
                            # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                            # dataRow = []
                            # for k in range(dim_output):
                            #     dataRow.append( meanErr[k] )
                            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                            # print( table )
                            # /DEBUG ==========================================

                            totAbsErr = np.sum(np.abs(meanErr))
                            if totAbsErr <= totAbsErrCurr:
                                totAbsErrCurr = totAbsErr
                                self.my_model.updatePars(
                                    SSest.A, SSest.B, SSest.C, SSest.D)

                        # DEBUG ===============================================
                        # ==========================================Print quality of the best model
                        # R  = '\033[31m'
                        # Bl  = '\033[30m'
                        # x0_est,_,_,_ = np.linalg.lstsq(ctrlStat.C, y)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.u_buffer, x0_est, y)
                        # meanErr = np.mean(Yest - ctrlStat.y_buffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                        # dataRow = []
                        # for k in range(dim_output):
                        #     dataRow.append( meanErr[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                        # print(R+table+Bl)
                        # /DEBUG ==============================================

            # Update initial state estimate
            x0_est, _, _, _ = np.linalg.lstsq(self.my_model.C, y)
            self.my_model.updateIC(x0_est)

            if t >= self.mod_est_phase:
                    # Drop probing noise
                self.is_prob_noise = 0

    def _Phi(self, y, u):
        """
        Feature vector of critic

        In Q-learning mode, it uses both ``y`` and ``u``. In value function approximation mode, it should use just ``y``

        Customization
        -------------

        Adjust this method if you still sitck with a linearly parametrized approximator for Q-function, value function etc.
        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like ``W @ self._Phi( y, u )`` 
        within :func:`~RLframe.controller._criticCost`

        """
        chi = np.concatenate([y, u])

        if self.critic_struct == 1:
            return np.concatenate([uptria2vec(np.kron(chi, chi)), chi])
        
        elif self.critic_struct == 2:
            return np.concatenate([uptria2vec(np.kron(chi, chi))])
        
        elif self.critic_struct == 3:
            return chi * chi
        
        elif self.critic_struct == 4:
            return np.concatenate([y**2, np.kron(y, u), u**2])

    def _criticCost(self, W, U, Y):
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
            e = W @ self._Phi(y_prev, u_prev) - self.gamma * self.Wprev @ self._Phi(y_next, u_next) - self.rcost(y_prev, u_prev)

            Jc += 1 / 2 * e**2

        return Jc

    def _critic(self, Wprev, Winit, U, Y):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~RLframe.controller._criticCost` instead.
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

        W = minimize(lambda W: self._criticCost(W, U, Y), Winit,
                     method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x

        # DEBUG ===============================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( W )
        # /DEBUG ==============================================================

        return W

    def _actorCost(self, U, y, N, W, delta, ctrl_mode):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------        

        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """

        myU = np.reshape(U, [N, self.dim_input])

        Y = np.zeros([N, self.dim_output])

        # System output prediction
        if (ctrl_mode == 1) or (ctrl_mode == 3) or (ctrl_mode == 5):    # Via exogenously passed model
            Y[0, :] = y
            x = self.xSys
            for k in range(1, self.n_actor):
                # Euler scheme
                x = x + delta * \
                    self.sys_rhs([], x, myU[k - 1, :], [], self.m,
                                self.I, self.dim_state, self.is_disturb)
                Y[k, :] = self.sys_out(x)

        elif (ctrl_mode == 2) or (ctrl_mode == 4) or (ctrl_mode == 6):    # Via estimated model
            myU_upsampled = myU.repeat(int(delta / self.sampl_time), axis=0)
            Yupsampled, _ = self._dssSim(
                self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0_est, y)
            Y = Yupsampled[::int(delta / self.sampl_time)]

        J = 0
        if (ctrl_mode == 1) or (ctrl_mode == 2):     # MPC
            for k in range(N):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
        # RL: Q-learning with n_critic-1 roll-outs of running cost
        elif (ctrl_mode == 3) or (ctrl_mode == 4):
            for k in range(N - 1):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
            J += W @ self._Phi(Y[-1, :], myU[-1, :])
        elif (ctrl_mode == 5) or (ctrl_mode == 6):     # RL: (normalized) stacked Q-learning
            for k in range(N):
                Q = W @ self._Phi(Y[k, :], myU[k, :])
                J += 1 / N * Q

        return J

    def _actor(self, y, u_init, N, W, delta, ctrl_mode):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``pred_step_size``

        Customization
        -------------         

        This method normally should not be altered, adjust :func:`~RLframe.controller._actorCost`, :func:`~RLframe.controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        actorOptMethod = 'SLSQP'
        if actorOptMethod == 'trust-constr':
            # 'disp': True, 'verbose': 2}
            actorOptOptions = {'maxiter': 300, 'disp': False}
        else:
            actorOptOptions = {'maxiter': 300, 'maxfev': 5000, 'disp': False,
                               'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7}  # 'disp': True, 'verbose': 2}

        isGlobOpt = 0

        myu_init = np.reshape(u_init, [N * self.dim_input, ])

        bnds = sp.optimize.Bounds(self.u_min, self.u_max, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {
                    'method': actorOptMethod, 'bounds': bnds, 'tol': 1e-7, 'options': actorOptOptions}
                U = basinhopping(lambda U: self._actorCost(
                    U, y, N, W, delta, ctrl_mode), myu_init, minimizer_kwargs=minimizer_kwargs, niter=10).x
            else:
                U = minimize(lambda U: self._actorCost(U, y, N, W, delta, ctrl_mode), myu_init,
                             method=actorOptMethod, tol=1e-7, bounds=bnds, options=actorOptOptions).x
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myu_init

        # DEBUG ===============================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # myU = np.reshape(U, [N, self.dim_input])
        # myU_upsampled = myU.repeat(int(delta/self.sampl_time), axis=0)
        # Yupsampled, _ = self._dssSim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0_est, y)
        # Y = Yupsampled[::int(delta/self.sampl_time)]
        # Yt = np.zeros([N, self.dim_output])
        # Yt[0, :] = y
        # x = self.xSys
        # for k in range(1, n_actor):
        #     x = x + delta * self.sys_rhs([], x, myU[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sys_out(x)
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
        # dataRow = []
        # for k in range(dim_output):
        #     dataRow.append( np.mean(Y[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
        # print(R+table+Bl)
        # /DEBUG ==============================================================

        return U[:self.dim_input]    # Return first action

    def computeAction(self, t, y):
        """
        Main method. See class documentation

        Customization
        -------------         

        Add your modes, that you introduced in :func:`~RLframe.controller._actorCost`, here

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampl_time:  # New sample
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
                self.u_buffer = pushVec(self.u_buffer, self.u_curr)
                self.y_buffer = pushVec(self.y_buffer, y)

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

                    # [EXPERIMENTAL] Call MATLAB's actor
                    # R1 = self.r_cost_pars[0]
                    # u = eng.optCtrl(eng.transpose(matlab.double(y.tolist())), eng.transpose(matlab.double(self.u_init.tolist())),
                    #                                   matlab.double(R1[:dim_output,:dim_output].tolist()), matlab.double(R1[dim_output:,dim_output:].tolist()), self.gamma,
                    #                                   self.n_actor,
                    #                                   eng.transpose(matlab.double(W.tolist())),
                    #                                   matlab.double(self.my_model.A.tolist()),
                    #                                   matlab.double(self.my_model.B.tolist()),
                    #                                   matlab.double(self.my_model.C.tolist()),
                    #                                   matlab.double(self.my_model.D.tolist()),
                    #                                   eng.transpose(matlab.double(self.my_model.x0_est.tolist())),
                    #                                   self.mode,
                    #                                   eng.transpose(matlab.double(self.u_min.tolist())),
                    #                                   eng.transpose(matlab.double(self.u_max.tolist())),
                    #                                   dt, matlab.double(self.trueModelPars), self.critic_struct, nargout=1)
                    # u = np.squeeze(np.asarray(u)

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

    For a three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot
    ctrlGain : : number
        Controller gain       
    t0 : : number
        Initial value of the controller's internal clock
    samplTime : : number
        Controller's sampling time (in seconds)        

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self, m=10, I=1, ctrlGain=10, f_min=-5, f_max=5, m_min=-1, m_max=1, t0=0, samplTime=0.1):
        self.m = m
        self.I = I
        self.ctrlGain = ctrlGain
        self.ctrlBnds = np.array([[f_min, f_max], [m_min, m_max]])
        self.ctrlClock = t0
        self.samplTime = samplTime
        self.uCurr = np.zeros(2)

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrlClock = t0
        self.uCurr = np.zeros(2)

    def _zeta(self, xNI, theta):
        """
        Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)

        """

        #                                 3
        #                             |x |
        #         4     4             | 3|
        # V(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta
        #                     /     / 2   2 \             \ 2
        #                    | sqrt| x + x   | + sqrt|x |  |
        #                     \     \ 1   2 /        | 3| /
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                         3
        #                                     |x |
        #            4     4                     | 3|
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #            1     2
        #                        /                                     \ 2
        #                        | x cos theta + x sin theta + sqrt|x | |
        #                        \ 1             2                | 3| /
        #                           \_______________  ______________/
        #                                            \/
        #                                            sigma~

        sigmaTilde = xNI[0] * np.cos(theta) + xNI[1] * \
            np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4 * xNI[0]**3 - 2 * \
            np.abs(xNI[2])**3 * np.cos(theta) / sigmaTilde**3

        nablaF[1] = 4 * xNI[1]**3 - 2 * \
            np.abs(xNI[2])**3 * np.sin(theta) / sigmaTilde**3

        nablaF[2] = (3 * xNI[0] * np.cos(theta) + 3 * xNI[1] * np.sin(theta) + 2 *
                     np.sqrt(np.abs(xNI[2]))) * xNI[2]**2 * np.sign(xNI[2]) / sigmaTilde**3

        return nablaF

    def _kappa(self, xNI, theta):
        """
        Stabilizing controller for NI-part

        """
        kappaVal = np.zeros(2)

        G = np.zeros([3, 2])
        G[:, 0] = np.array([1, 0, xNI[1]])
        G[:, 1] = np.array([0, 1, -xNI[0]])

        zetaVal = self._zeta(xNI, theta)

        kappaVal[0] = - np.abs(np.dot(zetaVal, G[:, 0])
                               )**(1 / 3) * np.sign(np.dot(zetaVal, G[:, 0]))
        kappaVal[1] = - np.abs(np.dot(zetaVal, G[:, 1])
                               )**(1 / 3) * np.sign(np.dot(zetaVal, G[:, 1]))

        return kappaVal

    def _Fc(self, xNI, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """

        sigmaTilde = xNI[0] * np.cos(theta) + xNI[1] * \
            np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        F = xNI[0]**4 + xNI[1]**4 + np.abs(xNI[2])**3 / sigmaTilde

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _thetaMinimizer(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        thetaVal = minimize(lambda theta: self._Fc(xNI, eta, theta), thetaInit,
                            method='trust-constr', tol=1e-6, bounds=bnds, options=options).x

        return thetaVal

    def _Cart2NH(self, CartCoords):
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        xNI = np.zeros(3)
        eta = np.zeros(2)

        xc = CartCoords[0]
        yc = CartCoords[1]
        alpha = CartCoords[2]
        v = CartCoords[3]
        omega = CartCoords[4]

        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * (yc * np.cos(alpha) - xc * np.sin(alpha)) - \
            alpha * (xc * np.cos(alpha) + yc * np.sin(alpha))

        eta[0] = omega
        eta[1] = (yc * np.cos(alpha) - xc * np.sin(alpha)) * omega + v

        return [xNI, eta]

    def _NH2CartCtrl(self, xNI, eta, uNI):
        """
        Get control for Cartesian NI from NH coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)


        """

        uCart = np.zeros(2)

        uCart[0] = self.m * (uNI[1] + xNI[1] * eta[0]**2 +
                             1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2]))
        uCart[1] = self.I * uNI[0]

        return uCart

    def computeAction(self, t, y):
        """
        See algorithm description in [[1]_], [[2]_]

        **This algorithm needs full-state measurement of the robot**

        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

        """

        timeInSample = t - self.ctrlClock

        if timeInSample >= self.samplTime:  # New sample

            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH(y)
            thetaStar = self._thetaMinimizer(xNI, eta)
            kappaVal = self._kappa(xNI, thetaStar)
            z = eta - kappaVal
            uNI = - self.ctrlGain * z
            u = self._NH2CartCtrl(xNI, eta, uNI)

            if self.ctrlBnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrlBnds[k, 0],
                                   self.ctrlBnds[k, 1])

            self.uCurr = u

            return u

        else:
            return self.uCurr


class Simulation:
    """class to create simulation and run simulation."""

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dimDisturb=2,
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
        """system - needs description"""
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

        # initial values of state
        self.x0 = np.zeros(dim_state)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2

        # initial value of control
        self.u0 = np.zeros(dim_input)

        # initial value of disturbance
        self.q0 = np.zeros(dimDisturb)

        # sensitivity of the solver. The lower the values, the more accurate
        # the simulation results are
        self.a_tol = a_tol
        self.r_tol = r_tol

        # x and y limits of scatter plot. Used so far rather for visualization
        # only, but may be integrated into the actor as constraints
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
        self.uMan = np.array([f_man, n_man])

        # control constraints
        self.f_min = f_min
        self.f_max = f_max
        self.m_min = m_min
        self.m_max = m_max

        self.ctrlBnds = np.array(
            [[self.f_min, self.f_max], [self.m_min, self.m_max]])

        """Other"""
        #%% User settings: main switches
        self.is_log_data = is_log_data
        self.is_visualization = is_visualization
        self.is_print_sim_step = is_print_sim_step

        # Static or dynamic controller
        self.is_dyn_ctrl = is_dyn_ctrl

        if self.is_dyn_ctrl:
            self.ksi0 = np.concatenate([self.x0, self.q0, self.u0])
        else:
            self.ksi0 = np.concatenate([self.x0, self.q0])

        # extras
        self.dataFiles = logdata(self.n_runs, save=self.is_log_data)

        if self.is_print_sim_step:
            warnings.filterwarnings('ignore')

    def create_simulator(self, closedLoop):
        simulator = sp.integrate.RK45(closedLoop,
                                      self.t0,
                                      self.ksi0,
                                      self.t1,
                                      max_step=self.dt / 2,
                                      first_step=1e-6,
                                      atol=self.a_tol,
                                      rtol=self.r_tol)
        return simulator

    def _create_figure(self, agent):
        y0 = System.out(self.x0)
        xCoord0 = self.x0[0]
        yCoord0 = self.x0[1]
        alpha0 = self.x0[2]
        alphaDeg0 = alpha0 / 2 / np.pi

        plt.close('all')

        self.simFig = plt.figure(figsize=(10, 10))

        # xy plane
        self.xyPlaneAxs = self.simFig.add_subplot(221,
                                                  autoscale_on=False,
                                                  xlim=(self.x_min,
                                                        self.x_max),
                                                  ylim=(self.y_min,
                                                        self.y_max),
                                                  xlabel='x [m]',
                                                  ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')

        self.xyPlaneAxs.set_aspect('equal', adjustable='box')
        self.xyPlaneAxs.plot([self.x_min, self.x_max], [
                             0, 0], 'k--', lw=0.75)   # Help line
        self.xyPlaneAxs.plot([0, 0], [self.y_min, self.y_max],
                             'k--', lw=0.75)   # Help line
        self.trajLine, = self.xyPlaneAxs.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robotMarker = pltMarker(angle=alphaDeg0)

        textTime = 't = {time:2.3f}'.format(time=self.t0)

        self.textTimeHandle = self.xyPlaneAxs.text(0.05, 0.95,
                                                   textTime,
                                                   horizontalalignment='left',
                                                   verticalalignment='center',
                                                   transform=self.xyPlaneAxs.transAxes)

        self.xyPlaneAxs.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        # Solution
        self.solAxs = self.simFig.add_subplot(222, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')
        self.solAxs.plot([self.t0, self.t1], [0, 0],
                         'k--', lw=0.75)   # Help line
        self.normLine, = self.solAxs.plot(self.t0, la.norm(
            [xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.alphaLine, = self.solAxs.plot(
            self.t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]')
        self.solAxs.legend(fancybox=True, loc='upper right')
        self.solAxs.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        # Cost
        self.costAxs = self.simFig.add_subplot(223, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            0, 1e4 * agent.rcost(y0, self.u0)), yscale='symlog', xlabel='t [s]')

        r = agent.rcost(y0, self.u0)
        textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=0)
        self.textIcostHandle = self.simFig.text(
            0.05, 0.5, textIcost, horizontalalignment='left', verticalalignment='center')
        self.rcostLine, = self.costAxs.plot(
            self.t0, r, 'r-', lw=0.5, label='r')
        self.icostLine, = self.costAxs.plot(
            self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.costAxs.legend(fancybox=True, loc='upper right')

        # Control
        self.ctrlAxs = self.simFig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')
        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line
        self.ctrlLines = self.ctrlAxs.plot(
            self.t0, toColVec(self.u0).T, lw=0.5)
        self.ctrlAxs.legend(
            iter(self.ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        # Pack all lines together
        cLines = namedtuple('lines', [
                            'trajLine', 'normLine', 'alphaLine', 'rcostLine', 'icostLine', 'ctrlLines'])
        self.lines = cLines(trajLine=self.trajLine,
                            normLine=self.normLine,
                            alphaLine=self.alphaLine,
                            rcostLine=self.rcostLine,
                            icostLine=self.icostLine,
                            ctrlLines=self.ctrlLines)

        self.currDataFile = self.dataFiles[0]

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

        return self.simFig

    def _initialize_figure(self):
        xCoord0 = self.x0[0]
        yCoord0 = self.x0[1]

        self.solScatter = self.xyPlaneAxs.scatter(
            xCoord0, yCoord0, marker=self.robotMarker.marker, s=400, c='b')
        self.currRun = 1

        return self.solScatter

    def _updateLine(self, line, newX, newY):
        line.set_xdata(np.append(line.get_xdata(), newX))
        line.set_ydata(np.append(line.get_ydata(), newY))

    def _resetLine(self, line):
        line.set_data([], [])

    # def _updateScatter(self, scatter, newX, newY):
    #     scatter.set_offsets(
    #         np.vstack([scatter.get_offsets().data, np.c_[newX, newY]]))

    def _updateText(self, textHandle, newText):
        textHandle.set_text(newText)

    def _updateScatter(self, textTime, ksi, alphaDeg, xCoord, yCoord, t, alpha, r, icost, u):
        self._updateText(self.textTimeHandle, textTime)
        # Update the robot's track on the plot
        self._updateLine(self.trajLine, *ksi[:2])

        self.robotMarker.rotate(alphaDeg)    # Rotate the robot on the plot
        self.solScatter.remove()
        self.solScatter = self.xyPlaneAxs.scatter(
            xCoord, yCoord, marker=self.robotMarker.marker, s=400, c='b')

        # Solution
        self._updateLine(self.normLine, t, la.norm([xCoord, yCoord]))
        self._updateLine(self.alphaLine, t, alpha)

        # Cost
        self._updateLine(self.rcostLine, t, r)
        self._updateLine(self.icostLine, t, icost)
        textIcost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._updateText(self.textIcostHandle, textIcost)
        # Control
        for (line, uSingle) in zip(self.ctrlLines, u):
            self._updateLine(line, t, uSingle)

    def _reset_sim(self, agent, nominalCtrl, simulator):
        if self.is_print_sim_step:
            print('.....................................Run {run:2d} done.....................................'.format(
                run=self.currRun))

        self.currRun += 1

        if self.currRun > self.Nruns:
            return

        if self.isLogData:
            self.currDataFile = self.dataFiles[self.currRun - 1]

        # Reset simulator
        simulator.status = 'running'
        simulator.t = self.t0
        simulator.y = self.ksi0

        # Reset controller
        if self.ctrl_mode > 0:
            agent.reset(self.t0)
        else:
            nominalCtrl.reset(self.t0)

    def _wrapper_take_steps(self, k, *args):
        return self._take_step(*args)

    def _take_step(self, sys, agent, nominalCtrl, simulator, animate=False):
        # take step
        simulator.step()

        t = simulator.t
        ksi = simulator.y

        x = ksi[0:self.dim_state]
        y = sys.out(x)

        u = ctrlSelector(
            t, y, self.uMan, nominalCtrl, agent, self.ctrl_mode)

        sys.receiveAction(u)
        agent.receiveSysState(sys._x)
        agent.update_icost(y, u)

        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        v = ksi[3]
        omega = ksi[4]
        icost = agent.i_cost_val

        if self.is_print_sim_step:
            printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u)

        if self.is_log_data:
            logDataRow(self.currDataFile, t, xCoord,
                       yCoord, alpha, v, omega, icost.val, u)

        if animate == True:
            alphaDeg = alpha / np.pi * 180
            r = agent.rcost(y, u)
            textTime = 't = {time:2.3f}'.format(time=t)
            self._updateScatter(textTime, ksi, alphaDeg,
                                xCoord, yCoord, t, alpha, r, icost, u)

        # Run done
        if t >= self.t1:
            self._reset_sim(agent, nominalCtrl, simulator)
            # icost = 0

            # for item in self.lines:
            #     if item != self.trajLine:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self._resetLine(subitem)
            #         else:
            #             self._resetLine(item)

            # self._updateLine(self.trajLine, np.nan, np.nan)

        if animate == True:
            return self.solScatter

    def run_simulation(self, sys, agent, nominalCtrl, simulator):
        if self.is_visualization == 0:
            self.currRun = 1
            self.currDataFile = dataFiles[0]

            while True:
                self._take_step(sys, agent, nominalCtrl, simulator)

        else:
            self.simFig = self._create_figure(agent)

            animate = True
            fargs = (sys, agent, nominalCtrl, simulator, animate)
            anm = animation.FuncAnimation(self.simFig,
                                          self._wrapper_take_steps,
                                          fargs=fargs,
                                          init_func=self._initialize_figure,
                                          interval=1)

            anm.running = True
            self.simFig.canvas.mpl_connect(
                'key_press_event', lambda event: onKeyPress(event, anm))
            self.simFig.tight_layout()
            plt.show()
