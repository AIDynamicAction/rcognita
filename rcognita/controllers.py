#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from .utilities import rc
from . import models
import numpy as np

import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from casadi import nlpsol

from casadi import Function
from optimizers import GradientOptimizer

# For debugging purposes
from tabulate import tabulate


class OptimalController(ABC):
    def __init__(
        self,
        action_init=[],
        t0=0,
        sampling_time=0.1,
        pred_step_size=0.1,
        state_dyn=[],
        sys_out=[],
        prob_noise_pow=1,
        is_est_model=0,
        model_est_stage=1,
        model_est_period=0.1,
        buffer_size=20,
        model_order=3,
        model_est_checks=0,
        critic_period=0.1,
        actor=[],
        critic=[],
        observation_target=[],
    ):

        self.actor = actor

        self.dim_input = self.actor.dim_input
        self.dim_output = self.actor.dim_output

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        # Controller: common
        self.pred_step_size = pred_step_size

        if isinstance(self.actor.control_bounds, (list, np.ndarray)):
            self.action_min = self.actor.control_bounds[0][: self.dim_input]

        else:
            self.action_min = self.actor.control_bounds.lb[: self.dim_input]

        if len(action_init) == 0:
            self.action_prev = self.action_min / 10
            self.action_sqn_init = rc.rep_mat(
                self.action_min / 10, 1, self.actor.Nactor
            )
        else:
            self.action_prev = action_init
            self.action_sqn_init = rc.rep_mat(action_init, 1, self.actor.Nactor)

        self.action_buffer = rc.zeros([buffer_size, self.dim_input])
        self.observation_buffer = rc.zeros([buffer_size, self.dim_output])

        # Exogeneous model's things
        self.state_dyn = state_dyn
        self.sys_out = sys_out
        # Model estimator's things
        self.is_est_model = is_est_model
        self.est_clock = t0
        self.is_prob_noise = 1
        self.prob_noise_pow = prob_noise_pow
        self.model_est_stage = model_est_stage
        self.model_est_period = model_est_period
        self.buffer_size = buffer_size
        self.model_order = model_order
        self.model_est_checks = model_est_checks

        # RL elements
        self.critic_clock = t0
        self.critic_period = critic_period
        self.critic = critic
        self.observation_target = observation_target

        self.accum_obj_val = 0

        self.control_mode = self.actor.control_mode

    def estimate_model(self, observation, t):
        if self.is_est_model or self.mode in ["RQL", "SQL"]:
            self.estimator.estimate_model(observation, t)

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_prev = self.action_min / 10

    def compute_action_sampled(self, t, observation, constraints=()):

        time_in_sample = t - self.ctrl_clock
        timeInCriticPeriod = t - self.critic_clock
        is_critic_update = timeInCriticPeriod >= self.critic_period

        if time_in_sample >= self.sampling_time:  # New sample
            if self.is_est_model:
                self.estimate_model(observation, t)
            # Update controller's internal clock
            self.ctrl_clock = t
            # DEBUG ==============================
            # print(self.ctrl_clock)
            # /DEBUG =============================
            action = self.compute_action(
                t, observation, is_critic_update=is_critic_update
            )

            return action

        else:
            return self.actor.action_prev

    @abstractmethod
    def compute_action(self):
        pass


class RLController(OptimalController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_action(
        self, t, observation, is_critic_update=False,
    ):
        # Critic

        # Update data buffers
        self.critic.update_buffers(observation, self.actor.action_prev)

        if is_critic_update:
            # Update critic's internal clock
            self.critic_clock = t
            self.critic.update(t=t)

        self.actor.update(observation)
        action = self.actor.get_action()

        return action


class CtrlNominal3WRobot:
    """
    This is a class of nominal controllers for 3-wheel robots used for benchmarking of other controllers.
    
    The controller is sampled.
    
    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_]).
  
    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot.
    ctrl_gain : : number
        Controller gain.       
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).       
    
    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594
        
    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887
       
    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)
    
    """

    def __init__(
        self, m, I, ctrl_gain=10, control_bounds=[], t0=0, sampling_time=0.1,
    ):

        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.control_bounds = control_bounds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.action_prev = rc.zeros(2)

    def reset(self, t0):

        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_prev = rc.zeros(2)

    def _zeta(self, xNI, theta):

        """
        Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators).

        """

        #                                 3
        #                             |x |
        #         4     4             | 3|
        # L(x) = x  +  x  +  ----------------------------------=   min F(x)
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

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde ** 3
        )

        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde ** 3
        )

        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * rc.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde ** 3
        )

        return nablaF

    def _kappa(self, xNI, theta):

        """
        Stabilizing controller for NI-part.

        """
        kappa_val = rc.zeros(2)

        G = rc.zeros([3, 2])
        G[:, 0] = [1, 0, xNI[1]]
        G[:, 1] = [0, 1, -xNI[0]]

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = -rc.abs(rc.dot(zeta_val, G[:, 0])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rc.abs(rc.dot(zeta_val, G[:, 1])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _Fc(self, xNI, eta, theta):

        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation.

        """

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + rc.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * rc.dot(z, z)

    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {"maxiter": 50, "disp": False}

        theta_val = minimize(
            lambda theta: self._Fc(xNI, eta, theta),
            thetaInit,
            method="trust-constr",
            tol=1e-6,
            bounds=bnds,
            options=options,
        ).x

        return theta_val

    def _Cart2NH(self, coords_Cart):

        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        xNI = rc.zeros(3)
        eta = rc.zeros(2)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = alpha
        xNI[1] = xc * rc.cos(alpha) + yc * rc.sin(alpha)
        xNI[2] = -2 * (yc * rc.cos(alpha) - xc * rc.sin(alpha)) - alpha * (
            xc * rc.cos(alpha) + yc * rc.sin(alpha)
        )

        eta[0] = omega
        eta[1] = (yc * rc.cos(alpha) - xc * rc.sin(alpha)) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI):

        """
        Get control for Cartesian NI from NH coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
        

        """

        uCart = rc.zeros(2)

        uCart[0] = self.m * (
            uNI[1]
            + xNI[1] * eta[0] ** 2
            + 1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2])
        )
        uCart[1] = self.I * uNI[0]

        return uCart

    def compute_action_sampled(self, t, observation):
        """
        See algorithm description in [[1]_], [[2]_].
        
        **This algorithm needs full-state measurement of the robot**.
        
        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887
           
        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)
        
        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            # Update internal clock
            self.ctrl_clock = t

            # This controller needs full-state measurement
            action = self.compute_action(observation)

            if self.control_bounds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.control_bounds[k, 0], self.control_bounds[k, 1]
                    )

            self.action_prev = action

            # DEBUG ===================================================================
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
            # print(R+table+Bl)
            # /DEBUG ===================================================================

            return action

        else:
            return self.action_prev

    def compute_action(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobot.compute_action`, but without invoking the internal clock.

        """

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = -self.ctrl_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)

        self.action_prev = action

        return action

    def compute_LF(self, observation):

        xNI, eta = self._Cart2NH(observation)
        theta_star = self._minimizer_theta(xNI, eta)

        return self._Fc(xNI, eta, theta_star)


class CtrlNominal3WRobotNI:
    """
    Nominal parking controller for NI using disassembled subgradients.
    
    """

    def __init__(self, ctrl_gain=10, control_bounds=[], t0=0, sampling_time=0.1):

        self.ctrl_gain = ctrl_gain
        self.control_bounds = control_bounds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.action_prev = rc.zeros(2)

    def reset(self, t0):

        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_prev = rc.zeros(2)

    def _zeta(self, xNI):

        """
        Analytic disassembled subgradient, without finding minimizer theta.

        """

        #                                 3
        #                             |x |
        #         4     4             | 3|
        # L(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta
        #                     /     / 2   2 \             \ 2
        #                     | sqrt| x + x   | + sqrt|x | |
        #                     \     \ 1   2 /        | 3| /
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                                3
        #                                            |x |
        #                4     4                     | 3|
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #                1     2
        #                           /                                      \ 2
        #                           | x cos theta + x sin theta + sqrt|x | |
        #                           \ 1             2                | 3|  /
        #                              \_______________  ______________/
        #                                              \/
        #                                             sigma~

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(abs(xNI[2]))

        nablaL = rc.zeros(3)

        nablaL[0] = (
            4 * xNI[0] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[0]
        )
        nablaL[1] = (
            4 * xNI[1] ** 3
            + rc.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[1]
        )
        nablaL[2] = 3 * rc.abs(xNI[2]) ** 2 * rc.sign(xNI[2]) + rc.abs(
            xNI[2]
        ) ** 3 / sigma ** 3 * 1 / np.sqrt(rc.abs(xNI[2])) * rc.sign(xNI[2])

        theta = 0

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        nablaF = rc.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.cos(theta) / sigma_tilde ** 3
        )
        nablaF[1] = (
            4 * xNI[1] ** 3 - 2 * rc.abs(xNI[2]) ** 3 * rc.sin(theta) / sigma_tilde ** 3
        )
        nablaF[2] = (
            (
                3 * xNI[0] * rc.cos(theta)
                + 3 * xNI[1] * rc.sin(theta)
                + 2 * np.sqrt(rc.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * rc.sign(xNI[2])
            / sigma_tilde ** 3
        )

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI):

        """
        Stabilizing controller for NI-part.

        """
        kappa_val = rc.zeros(2)

        G = rc.zeros([3, 2])
        G[:, 0] = rc.array([1, 0, xNI[1]], prototype=G)
        G[:, 1] = rc.array([0, 1, -xNI[0]], prototype=G)

        zeta_val = self._zeta(xNI)

        kappa_val[0] = -rc.abs(np.dot(zeta_val, G[:, 0])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -rc.abs(np.dot(zeta_val, G[:, 1])) ** (1 / 3) * rc.sign(
            rc.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _F(self, xNI, eta, theta):

        """
        Marginal function for NI.

        """

        sigma_tilde = (
            xNI[0] * rc.cos(theta) + xNI[1] * rc.sin(theta) + np.sqrt(rc.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _Cart2NH(self, coords_Cart):

        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.

        """

        xNI = rc.zeros(3)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]

        xNI[0] = alpha
        xNI[1] = xc * rc.cos(alpha) + yc * rc.sin(alpha)
        xNI[2] = -2 * (yc * rc.cos(alpha) - xc * rc.sin(alpha)) - alpha * (
            xc * rc.cos(alpha) + yc * rc.sin(alpha)
        )

        return xNI

    def _NH2ctrl_Cart(self, xNI, uNI):

        """
        Get control for Cartesian NI from NH coordinates.       

        """

        uCart = rc.zeros(2)

        uCart[0] = uNI[1] + 1 / 2 * uNI[0] * (xNI[2] + xNI[0] * xNI[1])
        uCart[1] = uNI[0]

        return uCart

    def compute_action_sampled(self, t, observation):
        """
        Compute sampled action.
        
        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            # Update internal clock
            self.ctrl_clock = t

            action = self.compute_action(observation)

            if self.control_bounds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.control_bounds[k, 0], self.control_bounds[k, 1]
                    )

            self.action_prev = action

            # DEBUG ===================================================================
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']
            # dataRow = [self.compute_LF(observation)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
            # print(R+table+Bl)
            # /DEBUG ===================================================================

            return action

        else:
            return self.action_prev

    def compute_action(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobotNI.compute_action`, but without invoking the internal clock.

        """

        xNI = self._Cart2NH(observation)
        kappa_val = self._kappa(xNI)
        uNI = self.ctrl_gain * kappa_val
        action = self._NH2ctrl_Cart(xNI, uNI)

        self.action_prev = action

        return action

    def compute_LF(self, observation):

        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(rc.abs(xNI[2]))

        return xNI[0] ** 4 + xNI[1] ** 4 + rc.abs(xNI[2]) ** 3 / sigma ** 2
