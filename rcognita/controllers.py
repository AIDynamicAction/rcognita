#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 13:39:34 2021

@author: Pavel Osinenko
"""

"""
=============================================================================
rcognita

https://github.com/AIDynamicAction/rcognita

Python framework for hybrid simulation of predictive reinforcement learning agents and classical controllers

=============================================================================

This module:

controllers

=============================================================================

Remark:

All vectors are treated as of type [n,]
All buffers are treated as of type [L, n] where each row is a vector
Buffers are updated from bottom to top
"""

from rcognita.utilities import dss_sim, rep_mat, uptria2vec, push_vec
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping


import torch
import torch.nn as nn
from rcognita.models import model_SS, model_NN


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# System identification packages
# import ssid  # Github:OsinenkoP/pyN4SID, fork of Githug:AndyLamperski/pyN4SID, with some errors fixed
# import sippy  # Github:CPCLAB-UNIPI/SIPPY

def ctrl_selector(t, y, uMan, ctrl_nominal, ctrl_benchmarking, mode):
    """
    Main interface for different controllers

    Parameters
    ----------
    mode : : integer
        Controller mode, see ``user settings`` section

    Returns
    -------
    u : : array of shape ``[dim_input, ]``
        Control action

    Customization
    -------
    Include your controller modes in this method

    """

    if mode==0: # Manual control
        u = uMan
    elif mode==-1: # Nominal controller
        u = ctrl_nominal.compute_action(t, y)
    elif mode > 0: # Controller for benchmakring
        u = ctrl_benchmarking.compute_action(t, y)

    return u

class ctrl_RL_pred:
    """
    Class of predictive reinforcement learning and model-predictive controllers. Multi-modal: can switch between different RL and MPC modes.

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
           * - 1 - Model-predictive control (MPC)
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right)=\\sum_{k=1}^{N_a} \\gamma^{k-1} r(y_k, u_k)`
           * - 2  - RL/ADP via :math:`N_a-1` roll-outs of :math:`r`
             - :math:`J \\left( y_1, \\{u\}_{1}^{N_a}\\right) =\\sum_{k=1}^{N_a-1} \\gamma^{k-1} r(y_k, u_k) + \\hat Q(y_{N_a}, u_{N_a})`
           * - 3  - RL/ADP via stacked Q-learning [[1]_]
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat Q(y_{N_a}, u_{N_a})`
           * - 4  - RL/ADP via normalized stacked Q-learning with terminal value function
             - :math:`J \\left( x_1, \\{u\\}_1^{N_a} \\right) = \\sum_{i=1}^{N-1} \\hat Q (\hat x_{i|k}, u_{i|k}, \\vartheta^*_{k-1}) + \\hat V(\\hat x_{N|k}; \\lambda^*_{k-1}) \\right)`


        **Add your specification into the table when customizing the agent**

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    t0 : : number
        Initial value of the controller's internal clock
    sampling_time : : number
        Controller's sampling time (in seconds)
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`
    pred_step_size : : number
        Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon
    sys_rhs, sys_out : : functions
        Functions that represents the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``x_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, x_sys`` are used in controller modes which rely on them.
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control
    model_est_stage : : number
        Initial time segment to fill the estimator's buffer before applying optimal control (in seconds)
    model_est_period : : number
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
    model_est_checks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
        May improve the prediction quality somewhat
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of running costs along horizon
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer
    critic_period : : number
        The same meaning as ``model_est_period``
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
    rcost_struct : : natural number
        Choice of the running cost structure.

        Currently available:

        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1

           * - Mode
             - Structure
           * - 1
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcost_pars`` should be ``[R1]``
           * - 2
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcost_pars``
               should be ``[R1, R2]``

        **Pass correct running cost parameters in** ``rcost_pars`` **(as a list)**

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
            x = ksi[0:sys.dimState]
            y = sys.out(x)
            u = agent.compute_action(t, y)
            sys.receiveAction(u)
            agent.update_icost(y, u)

    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155

    """

    def __init__(self, dim_input, dim_output, mode=1, ctrl_bnds=[], ctrl_mode=[], t0=0, sampling_time=0.1, Nactor=1, pred_step_size=0.1,
                 sys_rhs=[], sys_out=[], x_sys=[], is_prob_noise = 0, prob_noise_pow = 1, model_est_stage=1, model_est_period=0.1, buffer_size=20, model_order=3, model_est_checks=0,
                 gamma=1, Ncritic=4, critic_period=0.1, critic_struct_Q=1, critic_struct_V=1, rcost_struct=1, model=None, optimizer=None, criterion=None, is_estimate_model=1, is_use_offline_model=0, rcost_pars=[],
                 lr = None, feature_size = None, output_shape = None, layers = None, hidden_size = None, epochs = None, y_target=[]):

        self.dim_input = dim_input
        self.dim_output = dim_output

        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        # Controller: common
        self.Nactor = Nactor
        self.pred_step_size = pred_step_size

        self.uMin = np.array( ctrl_bnds[:,0] )
        self.uMax = np.array( ctrl_bnds[:,1] )
        self.Umin = rep_mat(self.uMin, 1, Nactor)
        self.Umax = rep_mat(self.uMax, 1, Nactor)

        self.uCurr = self.uMin/10

        self.Uinit = rep_mat( self.uMin/10 , 1, self.Nactor)

        self.ubuffer = np.zeros( [buffer_size, dim_input] )
        self.ybuffer = np.zeros( [buffer_size, dim_output] )

        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.x_sys = x_sys

        # Model estimator's things
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.model_est_stage = model_est_stage
        self.model_est_period = model_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.model_est_checks = model_est_checks

        A = np.zeros( [self.model_order, self.model_order] )
        B = np.zeros( [self.model_order, self.dim_input] )
        C = np.zeros( [self.dim_output, self.model_order] )
        D = np.zeros( [self.dim_output, self.dim_input] )
        x0est = np.zeros( self.model_order )

        self.my_model = model_SS(A, B, C, D, x0est)

        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)

        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct_Q = critic_struct_Q
        self.critic_struct_V = critic_struct_V
        self.rcost_struct = rcost_struct
        self.rcost_pars = rcost_pars
        self.y_target = y_target

        self.icost_val = 0

        if self.critic_struct_Q == 1:
            self.dim_crit_Q = ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input)
            self.Wq_min = -1e3*np.ones(self.dim_crit_Q)
            self.Wq_max = 1e3*np.ones(self.dim_crit_Q)
        elif self.critic_struct_Q == 2:
            self.dim_crit_Q = ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2
            self.Wq_min = np.zeros(self.dim_crit_Q_Q)
            self.Wq_max = 1e3*np.ones(self.dim_crit_Q)
        elif self.critic_struct_Q == 3:
            self.dim_crit_Q = self.dim_output + self.dim_input
            self.Wq_min = np.zeros(self.dim_crit_Q)
            self.Wq_max = 1e3*np.ones(self.dim_crit_Q)
        elif self.critic_struct_Q == 4:
            self.dim_crit_Q = self.dim_output + self.dim_output * self.dim_input + self.dim_input
            self.Wq_min = -1e3*np.ones(self.dim_crit_Q)
            self.Wq_max = 1e3*np.ones(self.dim_crit_Q)

        self.Wq_prev = np.ones(self.dim_crit_Q)

        self.Wq_init = self.Wq_prev

        if self.critic_struct_V == 1:
            self.dim_crit_V = ( ( self.dim_output) + 1 ) * ( self.dim_output )/2 + (self.dim_output)
            self.Wv_min = -1e3*np.ones(self.dim_crit_V)
            self.Wv_max = 1e3*np.ones(self.dim_crit_V)
        elif self.critic_struct_V == 2:
            self.dim_crit_V = ( ( self.dim_output ) + 1 ) * ( self.dim_output)/2
            self.Wv_min = np.zeros(self.dim_crit_V)
            self.Wv_max = 1e3*np.ones(self.dim_crit_V)
        elif self.critic_struct_V == 3:
            self.dim_crit_V = self.dim_output
            self.Wv_min = np.zeros(self.dim_crit_V)
            self.Wv_max = 1e3*np.ones(self.dim_crit_V)
        elif self.critic_struct_V == 4:
            self.dim_crit_V = self.dim_output + self.dim_output
            self.Wv_min = -1e3*np.ones(self.dim_crit_V)
            self.Wv_max = 1e3*np.ones(self.dim_crit_V)

        self.Wv_prev = np.ones(self.dim_crit_V)

        self.Wv_init = self.Wv_prev
################################################################
        self.model = model
        self.is_estimate_model = is_estimate_model
        self.is_use_offline_model = is_use_offline_model
        self.lr = lr
        self.optimizer = optimizer
        self.criterion = criterion
        self.feature_size = feature_size
        self.output_shape = output_shape
        self.layers = layers
        self.hidden_size = hidden_size
        self.epochs = epochs

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained

        """
        self.ctrl_clock = t0
        self.uCurr = self.uMin/10

    def receive_sys_state(self, x):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation

        """
        self.x_sys = x

    def rcost(self, y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)

        See class documentation
        """
        if self.y_target == []:
            chi = np.concatenate([y, u])
        else:
            chi = np.concatenate([y - self.y_target, u])

        r = 0

        if self.rcost_struct == 1:
            R1 = self.rcost_pars[0]
            r = chi @ R1 @ chi
        elif self.rcost_struct == 2:
            R1 = self.rcost_pars[0]
            R2 = self.rcost_pars[1]

            r = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi

        return r

    def upd_icost(self, y, u):
        """
        Sample-to-sample integrated running cost. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``icost`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)

        """
        self.icost_val += self.rcost(y, u)*self.sampling_time

    def _estimate_model(self, t, y):
        """
        Estimate model parameters by accumulating data buffers ``ubuffer`` and ``ybuffer``

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time: # New sample
            # Update buffers when using RL or requiring estimated model
                time_in_est_period = t - self.est_clock

                # Estimate model if required by ctrlStatMode
                if (time_in_est_period >= self.model_est_period) and self.is_estimate_model:
                    self.est_clock = t


            # Update initial state estimate
                elif self.is_use_offline_model:
                    pass

                if t >= self.model_est_stage:
                        # Drop probing noise
                        self.is_prob_noise = 0

    def _phi_Q(self, y, u):
        """
        Feature vector of critic

        In Q-learning mode, it uses both ``y`` and ``u``. In value function approximation mode, it should use just ``y``

        Customization
        -------------

        Adjust this method if you still sitck with a linearly parametrized approximator for Q-function, value function etc.
        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like ``W @ self._phi_Q( y, u )``
        within :func:`~RLframe.controller._critic_cost`

        """
        if self.y_target == []:
            chi = np.concatenate([y, u])
        else:
            chi = np.concatenate([y - self.y_target, u])

        if self.critic_struct_Q == 1:
            return np.concatenate([ uptria2vec( np.kron(chi, chi) ), chi ])
        elif self.critic_struct_Q == 2:
            return np.concatenate([ uptria2vec( np.kron(chi, chi) ) ])
        elif self.critic_struct_Q == 3:
            return chi * chi
        elif self.critic_struct_Q == 4:
            return np.concatenate([ y**2, np.kron(y, u), u**2 ])

    def _phi_V(self, y):
        """
        Feature vector of critic

        In value function approximation learning, it uses ``y``

        """
        if self.critic_struct_V == 1:
            return np.concatenate([ uptria2vec( np.kron(y, y) ), y ])
        elif self.critic_struct_V == 2:
            return np.concatenate([ uptria2vec( np.kron(y, y) ) ])
        elif self.critic_struct_V == 3:
            return y * y

    def _critic_cost(self, W, U, Y):
        """
        Cost function of the critic

        Currently uses value-iteration-like method

        Customization
        -------------

        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation

        """
        Jc = 0

        for k in range(self.Ncritic-1, 0, -1):
            yPrev = Y[k-1, :]
            yNext = Y[k, :]
            uPrev = U[k-1, :]
            uNext = U[k, :]

            # Temporal difference
            if self.mode == 4:

                Wq = W[:(self.dim_crit_Q)]
                Wv = W[-self.dim_crit_V:]
                params = W

                e_wq = np.dot(Wq, self._phi_Q(yPrev, uPrev)) - self.gamma * np.dot(self.Wq_prev, self._phi_Q( yNext, uNext )) - self.rcost(yPrev, uPrev)
                e_wv = np.dot(Wv, self._phi_V(yPrev)) - self.gamma * np.dot(self.Wv_prev, self._phi_V(yNext)) - self.rcost(yPrev, uPrev)

                Jc += 0.5 * e_wq**2  + 0.5 * e_wv**2
            else:
                e = np.dot(W, self._phi_Q( yPrev, uPrev )) - self.gamma * np.dot(self.Wq_prev, self._phi_Q( yNext, uNext )) - self.rcost(yPrev, uPrev)

                Jc += 0.5 * e**2

        return Jc


    def _critic(self, Wqprev, Wqinit, Wvprev, Wvinit, U, Y):
        """
        See class documentation.

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~RLframe.controller._critic_cost` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}

        if self.mode == 4:
            bnds_min = np.concatenate([self.Wq_min, self.Wv_min])
            bnds_max = np.concatenate([self.Wq_max, self.Wv_max])

            bnds = sp.optimize.Bounds(bnds_min, bnds_max, keep_feasible=True)

            wqv_init = np.concatenate([self.Wq_init, self.Wv_init])

            try:
                w_lmbd = minimize(lambda w_lmbd: self._critic_cost(w_lmbd, U, Y), wqv_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x
                Wq = w_lmbd[:self.dim_crit_Q]
                Wv = w_lmbd[-self.dim_crit_V:]
            except:
                print('Critic''s optimizer failed. Returning default parameters.')
                Wq = self.Wq_init
                Wv = self.Wv_init

        else:
            bnds = sp.optimize.Bounds(self.Wq_min, self.Wq_max, keep_feasible=True)

            Wq = minimize(lambda W: self._critic_cost(W, U, Y), self.Wq_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x
            Wv = None


        return Wq, Wv

    def _actor_cost(self, U, y, Wq, Wv):
        """
        See class documentation.

        Customization
        -------------

        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """

        myU = np.reshape(U, [self.Nactor, self.dim_input])

        Y = np.zeros([self.Nactor, self.dim_output])

        # System output prediction
        if (self.mode==1) or (self.mode==2) or (self.mode==3) or (self.mode==4):    # Via exogenously passed model
            Y[0, :] = y
            x = self.x_sys
            for k in range(1, self.Nactor-1):
                if self.is_use_offline_model:
                    a = np.concatenate([x.reshape(3,), myU[k-1,:]])
                    x =  self.model.prediction(a.reshape(5))[0]  # need to define trained_NN with saved weights get_next_state

                else:
                    x = x + self.pred_step_size * self.sys_rhs([], x, myU[k-1, :], [])  # Euler scheme
                Y[k, :] = self.sys_out(x)

        J = 0
        if self.mode==1:     # MPC
            for k in range(self.Nactor):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
        elif self.mode==2:     # RL: Q-learning with Ncritic-1 roll-outs of running cost
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
             J += np.dot(Wq, self._phi_Q( Y[-1, :], myU[-1, :] ))
        elif self.mode==3:     # RL: stacked Q-learning
             for k in range(self.Nactor):
                Q = np.dot(Wq, self._phi_Q( Y[k, :], myU[k, :] ))
                # J += 1/N * Q # normalization
                J += Q
        elif self.mode==4:
             for k in range(self.Nactor-1):
                Q = np.dot(Wq, self._phi_Q( Y[k, :], myU[k, :] ))
                # J += 1/N * Q
                J += Q
             J += np.dot(Wv, self._phi_V( Y[-1, :]))
        return J

    def _actor(self, y, Wq, Wv):
        """
        See class documentation.

        Customization
        -------------

        This method normally should not be altered, adjust :func:`~RLframe.controller._actor_cost`, :func:`~RLframe.controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}

        isGlobOpt = 0

        myUinit = np.reshape(self.Uinit, [self.Nactor*self.dim_input,])

        bnds = sp.optimize.Bounds(self.Umin, self.Umax, keep_feasible=True)

        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 'tol': 1e-7, 'options': actor_opt_options}
                U = basinhopping(lambda U: self._actor_cost(U, y, Wq, Wv), myUinit, minimizer_kwargs=minimizer_kwargs, niter = 10).x
            else:
                U = minimize(lambda U: self._actor_cost(U, y, Wq, Wv), myUinit, method=actor_opt_method, tol=1e-7, bounds=bnds, options=actor_opt_options).x
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myUinit

        return U[:self.dim_input]     # Return first action

    def compute_action(self, t, y):
        """
        Main method. See class documentation

        Customization
        -------------

        Add your modes, that you introduced in :func:`~RLframe.controller._actor_cost`, here

        """

        if self.is_estimate_model:
            self._estimate_model(t,y)

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            if self.mode==1:
                if self.is_prob_noise and self.is_estimate_model:
                    return self.prob_noise_pow * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and self.is_estimate_model:
                    u = self._actor(y, [],[])

                else:
                    u = self._actor(y, [], [])

            elif self.mode in (2, 4, 3):
                # Critic
                timeInCriticPeriod = t - self.critic_clock

                # Update data buffers
                self.ubuffer = push_vec(self.ubuffer, self.uCurr)
                self.ybuffer = push_vec(self.ybuffer, y)

                if timeInCriticPeriod >= self.critic_period:
                    self.critic_clock = t

                    Wq, Wv = self._critic(self.Wq_prev, self.Wq_init, self.Wv_prev, self.Wv_init, self.ubuffer[-self.Ncritic:,:], self.ybuffer[-self.Ncritic:,:])
                    self.Wq_prev = Wq
                    self.Wv_prev = Wv

                else:
                    Wq = self.Wq_prev
                    Wv = self.Wv_prev

                if self.is_prob_noise and self.is_estimate_model:
                    u = self.prob_noise_pow * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and self.is_estimate_model:
                    u = self._actor(y,  Wq,[])
                else:
                    u = self._actor(y, Wq, Wv)

            self.uCurr = u

            return u

        else:
            return self.uCurr


class ctrl_nominal_3wrobot:
    """
    This is a class of nominal controllers for 3-wheel robots used for benchmarking of other controllers.

    The controller is sampled.

    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot
    ctrl_gain : : number
        Controller gain
    t0 : : number
        Initial value of the controller's internal clock
    sampling_time : : number
        Controller's sampling time (in seconds)

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self, m, I, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.uCurr = np.zeros(2)

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrl_clock = t0
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

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3

        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3

        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3

        return nablaF

    def _kappa(self, xNI, theta):
        """
        Stabilizing controller for NI-part

        """
        kappa_val = np.zeros(2)

        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )

        return kappa_val

    def _Fc(self, xNI, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde

        z = eta - self._kappa(xNI, theta)

        return F + 1/2 * np.dot(z, z)

    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        theta_val = minimize(lambda theta: self._Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x

        return theta_val

    def _Cart2NH(self, coords_Cart):
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

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )

        eta[0] = omega
        eta[1] = ( yc * np.cos(alpha) - xc * np.sin(alpha) ) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI):
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

        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
        uCart[1] = self.I * uNI[0]

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

        if time_in_sample >= self.sampling_time: # New sample
            self.ctrl_clock = t
            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH( y )
            theta_star = self._minimizer_theta(xNI, eta)
            kappa_val = self._kappa(xNI, theta_star)
            z = eta - kappa_val
            uNI = - self.ctrl_gain * z
            u = self._NH2ctrl_Cart(xNI, eta, uNI)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])

            self.uCurr = u

            return u

        else:
            return self.uCurr


class ctrl_nominal_kinematic_3wrobot:


    def __init__(self, ctrl_bnds, pose_goal, t0, sampling_time):
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        self.pose_goal = pose_goal

        self.uCurr = np.zeros(2)

    def compute_action(self, t, dstate):

        time_in_sample = t - self.ctrl_clock
        if time_in_sample >= self.sampling_time: # New sample
            self.ctrl_clock = t

            # This controller needs full-state measurement
            x = dstate[0]
            y = dstate[1]
            theta = dstate[2]
            # errors
            e_x = self.pose_goal[0] - x
            e_y = self.pose_goal[1] - y
            error_angle = self.pose_goal[1] - theta

            # distance to goal
            dist_to_goal = math.sqrt(e_x**2 + e_y**2)
            if theta > math.pi:
                error_angle = -np.arctan2(e_y, e_x) + theta - 2 * math.pi
            else:
                error_angle = -np.arctan2(e_y, e_x) + theta
            errors = [e_x, e_y, error_angle]
            u_v = 0.6 * dist_to_goal * np.cos(error_angle)
            u_w = - 0.4 * error_angle

            u = np.array([u_v, u_w])

            # theta_star = self._minimizer_theta(xNI, eta)
            # kappa_val = self._kappa(xNI, theta_star)
            # z = eta - kappa_val
            # uNI = - self.ctrl_gain * z
            # u = self._NH2ctrl_Cart(xNI, eta, uNI)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])

            self.uCurr = u

            return u

        else:
            return self.uCurr
