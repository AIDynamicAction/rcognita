#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from .utilities import dss_sim
from .utilities import rep_mat
from .utilities import uptria2vec
from .utilities import push_vec
from . import models
import numpy as np
from .npcasadi_api import SymbolicHandler
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from numpy.linalg import lstsq
from numpy import reshape
from functools import partial
from abc import ABC, abstractmethod
from casadi import nlpsol
import warnings
from casadi import Function

# For debugging purposes
from tabulate import tabulate

try:
    import sippy
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting sippy failed. You may still use rcognita, but"
        + " without model identification capability. \nRead on how"
        + " to install sippy at https://github.com/AIDynamicAction/rcognita\n",
        UserWarning,
        __file__,
        33,
    )


class OptimalController(ABC):
    def __init__(
        self,
        action_init=[],
        t0=0,
        sampling_time=0.1,
        pred_step_size=0.1,
        state_dyn=[],
        sys_out=[],
        state_sys=[],
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
        stage_obj_pars=[],
        observation_target=[],
    ):

        self.actor = actor
        is_symbolic = self.actor.actor_optimizer.is_symbolic

        npcsd = SymbolicHandler(is_symbolic=is_symbolic)
        # npcsd = SymbolicHandler(is_symbolic=self.critic_optimizer.is_symbolic)

        self.dim_input = self.actor.dim_input
        self.dim_output = self.actor.dim_output

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        # Controller: common
        self.pred_step_size = pred_step_size

        if isinstance(self.actor.actor_optimizer.bounds, list):
            self.action_min = np.array(
                self.actor.actor_optimizer.bounds[0][: self.dim_input]
            )
        else:
            self.action_min = np.array(
                self.actor.actor_optimizer.bounds.lb[: self.dim_input]
            )

        if len(action_init) == 0:
            self.action_prev = self.action_min / 10
            self.action_sqn_init = npcsd.rep_mat(
                self.action_min / 10, 1, self.actor.Nactor
            )
        else:
            self.action_prev = action_init
            self.action_sqn_init = npcsd.rep_mat(action_init, 1, self.actor.Nactor)

        self.action_buffer = npcsd.zeros([buffer_size, self.dim_input])
        self.observation_buffer = npcsd.zeros([buffer_size, self.dim_output])

        # Exogeneous model's things
        self.state_dyn = state_dyn
        self.sys_out = sys_out
        self.state_sys = state_sys
        self.state_sys_prev = state_sys
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

        A = npcsd.zeros([self.model_order, self.model_order])
        B = npcsd.zeros([self.model_order, self.dim_input])
        C = npcsd.zeros([self.dim_output, self.model_order])
        D = npcsd.zeros([self.dim_output, self.dim_input])
        x0est = npcsd.zeros(self.model_order)

        self.my_model = models.ModelSS(A, B, C, D, x0est)

        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)

        # RL elements
        self.critic_clock = t0
        self.critic_period = critic_period
        self.critic = critic
        self.stage_obj_pars = stage_obj_pars
        self.stage_obj_model = models.ModelQuadForm(
            **{f"R{i+1}": x for (i, x) in enumerate(self.stage_obj_pars)}
        )
        self.observation_target = observation_target

        self.accum_obj_val = 0

        self.ctrl_mode = self.actor.ctrl_mode

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_prev = self.action_min / 10

    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys_prev = self.state_sys
        self.state_sys = state

    def stage_obj(self, observation, action, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = npcsd.concatenate([observation.T, action.T])
        else:
            chi = npcsd.concatenate([(observation - self.observation_target), action])

        stage_obj = self.stage_obj_model(chi, chi, is_symbolic=is_symbolic)

        return stage_obj

    def upd_accum_obj(self, observation, action, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """

        self.accum_obj_val += (
            self.stage_obj(observation.T, action.T, is_symbolic) * self.sampling_time
        )

    def _estimate_model(self, t, observation):
        """
        Estimate model parameters by accumulating data buffers ``action_buffer`` and ``observation_buffer``.
        
        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            # Update buffers when using RL or requiring estimated model
            if self.is_est_model or self.mode in ["RQL", "SQL"]:
                time_in_est_period = t - self.est_clock

                # Estimate model if required
                if (time_in_est_period >= self.model_est_period) and self.is_est_model:
                    # Update model estimator's internal clock
                    self.est_clock = t

                    try:
                        # Using Github:CPCLAB-UNIPI/SIPPY
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(
                            self.observation_buffer,
                            self.action_buffer,
                            id_method="N4SID",
                            SS_fixed_order=self.model_order,
                            SS_D_required=False,
                            SS_A_stability=False,
                            # SS_f=int(self.buffer_size/12),
                            # SS_p=int(self.buffer_size/10),
                            SS_PK_B_reval=False,
                            tsample=self.sampling_time,
                        )

                        self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)

                        # ToDo: train an NN via Torch
                        # NN_wgts = NN_train(...)

                    except:
                        print("Model estimation problem")
                        self.my_model.upd_pars(
                            np.zeros([self.model_order, self.model_order]),
                            np.zeros([self.model_order, self.dim_input]),
                            np.zeros([self.dim_output, self.model_order]),
                            np.zeros([self.dim_output, self.dim_input]),
                        )

                    # Model checks
                    if self.model_est_checks > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        tot_abs_err_curr = 1e8
                        for k in range(self.model_est_checks):
                            A, B, C, D = (
                                self.model_stack[k].A,
                                self.model_stack[k].B,
                                self.model_stack[k].C,
                                self.model_stack[k].D,
                            )
                            x0est, _, _, _ = np.linalg.lstsq(C, observation)
                            Yest, _ = dss_sim(
                                A, B, C, D, self.action_buffer, x0est, observation
                            )
                            mean_err = np.mean(Yest - self.observation_buffer, axis=0)

                            # DEBUG ===================================================================
                            # ================================Interm output of model prediction quality
                            # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                            # dataRow = []
                            # for k in range(dim_output):
                            #     dataRow.append( mean_err[k] )
                            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                            # print( table )
                            # /DEBUG ===================================================================

                            tot_abs_err = np.sum(np.abs(mean_err))
                            if tot_abs_err <= tot_abs_err_curr:
                                tot_abs_err_curr = tot_abs_err
                                self.my_model.upd_pars(
                                    SSest.A, SSest.B, SSest.C, SSest.D
                                )

                        # DEBUG ===================================================================
                        # ==========================================Print quality of the best model
                        # R  = '\033[31m'
                        # Bl  = '\033[30m'
                        # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, observation)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.action_buffer, x0est, observation)
                        # mean_err = np.mean(Yest - ctrlStat.observation_buffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                        # dataRow = []
                        # for k in range(dim_output):
                        #     dataRow.append( mean_err[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                        # print(R+table+Bl)
                        # /DEBUG ===================================================================

            # Update initial state estimate
            x0est, _, _, _ = np.linalg.lstsq(self.my_model.C, observation)
            self.my_model.updateIC(x0est)

            if t >= self.model_est_stage:
                # Drop probing noise
                self.is_prob_noise = 0

    def compute_action_sampled(self, t, observation, constraints=(), is_symbolic=False):

        time_in_sample = t - self.ctrl_clock
        timeInCriticPeriod = t - self.critic_clock
        is_critic_update = timeInCriticPeriod >= self.critic_period

        if time_in_sample >= self.sampling_time:  # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            # DEBUG ==============================
            print(self.ctrl_clock)
            # /DEBUG =============================
            action = self.compute_action(
                t,
                observation,
                is_critic_update=is_critic_update,
                is_symbolic=is_symbolic,
            )

            self.action_prev = action

            return action

        else:
            return self.action_prev

    @abstractmethod
    def compute_action(self):
        pass


class CtrlRLStab(OptimalController):
    def __init__(
        self,
        safe_ctrl=[],
        safe_decay_rate=[],
        eps_action=1e-3,
        eps_weights=1e-3,
        eps_critic=1e-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Stabilizing constraint stuff
        self.safe_ctrl = safe_ctrl  # Safe controller (agent)
        self.safe_decay_rate = safe_decay_rate
        self.eps_action = eps_action
        self.eps_weights = eps_weights
        self.eps_critic = eps_critic
        self.critic_values = []
        self.g_emergency_critic_deriv = []
        self.g_emerency_critic_diff_weights = []

    def update_critic(self, t=None):
        npcsd = SymbolicHandler(is_symbolic=True)
        g = lambda x: npcsd.norm_1(x - self.critic.weights_prev) - self.eps_weights
        self.critic.weights = self.critic._critic_optimizer(
            constraints=g, is_symbolic=True, t=t
        )

    def update_actor(self, observation, t=None):
        npcsd = SymbolicHandler(is_symbolic=True)
        g = lambda x: npcsd.norm_1(x - self.action_prev) - self.eps_action
        action = self.actor._actor_optimizer(
            observation, constraints=g, is_symbolic=True, t=t
        )
        self.actor.action_prev = action
        return action

    def compute_action(self, t, observation, is_critic_update=False, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic=True)

        observation_prev = self.critic.observation_buffer[-1, :]

        self.critic.action_buffer = push_vec(
            self.critic.action_buffer, self.action_prev, is_symbolic=is_symbolic
        )

        self.critic.observation_buffer = npcsd.push_vec(
            npcsd.array(self.critic.observation_buffer), npcsd.array(observation)
        )
        critic_diff_factual = self.critic(
            self.critic.weights, observation, is_symbolic=is_symbolic
        ) - self.critic(self.critic.weights, observation_prev, is_symbolic=is_symbolic)

        # DEBUG ===================================================================
        self.critic_values.append(
            [
                SymbolicHandler(is_symbolic).array(
                    self.critic(
                        self.critic.weights, observation, is_symbolic=is_symbolic
                    )
                ),
                npcsd.array(critic_diff_factual),
                t,
            ]
        )
        # /DEBUG ===================================================================

        if critic_diff_factual <= self.safe_decay_rate * self.sampling_time:
            action = self.update_actor(observation, t=t)
            # self.critic.weights = (
            #     self.critic.weights + self.eps_weights
            # )
            self.update_critic(t=t)
        else:
            action = self.safe_ctrl.compute_action(observation)
            self.critic.weights = self.emergency_search(
                action, observation, is_symbolic=is_symbolic, t=t
            )

        self.critic.weights_prev = self.critic.weights

        return action

    def emergency_search(self, action, observation, is_symbolic=False, t=None):
        print("Emergency started!!!!!!!!!!!!")
        npcsd = SymbolicHandler(is_symbolic)

        options = {
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-4,
        }

        state_dyn_vector = self.state_dyn([], observation, action)

        critic_gradient, weights_symbolic = self.critic.grad_observation(
            observation, is_symbolic
        )
        critic_deriv = npcsd.dot(critic_gradient, state_dyn_vector)
        critic_curr = self.critic(
            weights_symbolic, npcsd.array(observation), is_symbolic=True
        )
        critic_prev = self.critic(
            self.critic.weights, observation, is_symbolic=is_symbolic
        )
        critic_diff_weights = critic_curr - critic_prev

        nlp = {}
        nlp["x"] = weights_symbolic
        nlp["f"] = critic_deriv
        nlp["g"] = npcsd.concatenate([critic_deriv, critic_diff_weights])
        solver = nlpsol("solver", "ipopt", nlp, options)
        result = solver(
            x0=npcsd.array(self.critic.weights),
            ubx=self.critic.Wmax,
            lbx=self.critic.Wmin,
            ubg=npcsd.array([-2 * self.safe_decay_rate, self.eps_critic]),
        )["x"]

        ##### DEBUG
        g1 = Function("g1", [weights_symbolic], [critic_diff_weights])
        g2 = Function(
            "g2", [weights_symbolic], [critic_deriv + 2 * self.safe_decay_rate]
        )
        critic_deriv = Function("f", [weights_symbolic], [critic_deriv])

        self.g_emergency_critic_deriv.append([g2(result), t])
        self.g_emerency_critic_diff_weights.append([g1(result), t])

        # row_header = ["Dg1", "Dg2", "critic_deriv"]
        # row_data = [g1(result), g2(result), critic_deriv(result)]
        # row_format = ("8.3f", "8.3f", "8.3f")
        # table = tabulate(
        #     [row_header, row_data],
        #     floatfmt=row_format,
        #     headers="firstrow",
        #     tablefmt="grid",
        # )

        # print(table)
        ##### DEBUG

        return result


class CtrlOptPred(OptimalController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_action(
        self, t, observation, is_critic_update=False, is_symbolic=False,
    ):
        npcsd = SymbolicHandler(is_symbolic)

        if self.ctrl_mode == "MPC":
            # Apply control when model estimation phase is over
            if self.is_prob_noise and self.is_est_model:
                return self.prob_noise_pow * (rand(self.dim_input) - 0.5)

            elif not self.is_prob_noise and self.is_est_model:
                action = self.actor._actor_optimizer(
                    observation, is_symbolic=is_symbolic
                )

            elif self.ctrl_mode == "MPC":
                action = self.actor._actor_optimizer(
                    observation, is_symbolic=is_symbolic
                )

        elif self.ctrl_mode in ["RQL", "SQL"]:
            # Critic

            # Update data buffers
            self.critic.action_buffer = push_vec(
                self.critic.action_buffer, self.action_prev, is_symbolic=is_symbolic
            )

            self.critic.observation_buffer = npcsd.push_vec(
                npcsd.array(self.critic.observation_buffer), npcsd.array(observation)
            )

            if is_critic_update:
                # Update critic's internal clock
                self.critic_clock = t

                self.critic.weights = self.critic._critic_optimizer(
                    is_symbolic=is_symbolic
                )
                self.critic.weights_prev = self.critic.weights

                # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                # self.weights_init = self.weights_prev

            else:
                self.critic.weights = self.critic.weights_prev

            # Actor. Apply control when model estimation phase is over
            if self.is_prob_noise and self.is_est_model:
                action = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
            elif not self.is_prob_noise and self.is_est_model:
                action = self.actor._actor_optimizer(observation)

            elif self.ctrl_mode in ["RQL", "SQL"]:
                action = self.actor._actor_optimizer(
                    observation, is_symbolic=is_symbolic,
                )

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
        self,
        m,
        I,
        ctrl_gain=10,
        ctrl_bnds=[],
        t0=0,
        sampling_time=0.1,
        is_symbolic=False,
    ):
        npcsd = SymbolicHandler(is_symbolic)

        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.action_prev = npcsd.zeros(2)

    def reset(self, t0, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_prev = npcsd.zeros(2)

    def _zeta(self, xNI, theta, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
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
            xNI[0] * npcsd.cos(theta)
            + xNI[1] * npcsd.sin(theta)
            + np.sqrt(npcsd.abs(xNI[2]))
        )

        nablaF = npcsd.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3
            - 2 * npcsd.abs(xNI[2]) ** 3 * npcsd.cos(theta) / sigma_tilde ** 3
        )

        nablaF[1] = (
            4 * xNI[1] ** 3
            - 2 * npcsd.abs(xNI[2]) ** 3 * npcsd.sin(theta) / sigma_tilde ** 3
        )

        nablaF[2] = (
            (
                3 * xNI[0] * npcsd.cos(theta)
                + 3 * xNI[1] * npcsd.sin(theta)
                + 2 * np.sqrt(npcsd.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * npcsd.sign(xNI[2])
            / sigma_tilde ** 3
        )

        return nablaF

    def _kappa(self, xNI, theta, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Stabilizing controller for NI-part.

        """
        kappa_val = npcsd.zeros(2)

        G = npcsd.zeros([3, 2])
        G[:, 0] = npcsd.array([1, 0, xNI[1]])
        G[:, 1] = npcsd.array([0, 1, -xNI[0]])

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = -npcsd.abs(np.dot(zeta_val, G[:, 0])) ** (1 / 3) * npcsd.sign(
            np.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -npcsd.abs(np.dot(zeta_val, G[:, 1])) ** (1 / 3) * npcsd.sign(
            np.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _Fc(self, xNI, eta, theta, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation.

        """

        sigma_tilde = (
            xNI[0] * npcsd.cos(theta)
            + xNI[1] * npcsd.sin(theta)
            + np.sqrt(npcsd.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + npcsd.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

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

    def _Cart2NH(self, coords_Cart, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        xNI = npcsd.zeros(3)
        eta = npcsd.zeros(2)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = alpha
        xNI[1] = xc * npcsd.cos(alpha) + yc * npcsd.sin(alpha)
        xNI[2] = -2 * (yc * npcsd.cos(alpha) - xc * npcsd.sin(alpha)) - alpha * (
            xc * npcsd.cos(alpha) + yc * npcsd.sin(alpha)
        )

        eta[0] = omega
        eta[1] = (yc * npcsd.cos(alpha) - xc * npcsd.sin(alpha)) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Get control for Cartesian NI from NH coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
        

        """

        uCart = npcsd.zeros(2)

        uCart[0] = self.m * (
            uNI[1]
            + xNI[1] * eta[0] ** 2
            + 1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2])
        )
        uCart[1] = self.I * uNI[0]

        return uCart

    def compute_action_sampled(self, t, observation, is_symbolic=False):
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

            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1]
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

    def compute_action(self, observation, is_symbolic=False):
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

    def __init__(
        self, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1, is_symbolic=False
    ):
        npcsd = SymbolicHandler(is_symbolic)
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.action_prev = npcsd.zeros(2)

    def reset(self, t0, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_prev = npcsd.zeros(2)

    def _zeta(self, xNI, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
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

        nablaL = npcsd.zeros(3)

        nablaL[0] = (
            4 * xNI[0] ** 3
            + npcsd.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[0]
        )
        nablaL[1] = (
            4 * xNI[1] ** 3
            + npcsd.abs(xNI[2]) ** 3
            / sigma ** 3
            * 1
            / np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) ** 3
            * 2
            * xNI[1]
        )
        nablaL[2] = 3 * npcsd.abs(xNI[2]) ** 2 * npcsd.sign(xNI[2]) + npcsd.abs(
            xNI[2]
        ) ** 3 / sigma ** 3 * 1 / np.sqrt(npcsd.abs(xNI[2])) * npcsd.sign(xNI[2])

        theta = 0

        sigma_tilde = (
            xNI[0] * npcsd.cos(theta)
            + xNI[1] * npcsd.sin(theta)
            + np.sqrt(npcsd.abs(xNI[2]))
        )

        nablaF = npcsd.zeros(3)

        nablaF[0] = (
            4 * xNI[0] ** 3
            - 2 * npcsd.abs(xNI[2]) ** 3 * npcsd.cos(theta) / sigma_tilde ** 3
        )
        nablaF[1] = (
            4 * xNI[1] ** 3
            - 2 * npcsd.abs(xNI[2]) ** 3 * npcsd.sin(theta) / sigma_tilde ** 3
        )
        nablaF[2] = (
            (
                3 * xNI[0] * npcsd.cos(theta)
                + 3 * xNI[1] * npcsd.sin(theta)
                + 2 * np.sqrt(npcsd.abs(xNI[2]))
            )
            * xNI[2] ** 2
            * npcsd.sign(xNI[2])
            / sigma_tilde ** 3
        )

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Stabilizing controller for NI-part.

        """
        kappa_val = npcsd.zeros(2)

        G = npcsd.zeros([3, 2])
        G[:, 0] = npcsd.rc_array([1, 0, xNI[1]])
        G[:, 1] = npcsd.rc_array([0, 1, -xNI[0]])

        zeta_val = self._zeta(xNI)

        kappa_val[0] = -npcsd.abs(np.dot(zeta_val, G[:, 0])) ** (1 / 3) * npcsd.sign(
            np.dot(zeta_val, G[:, 0])
        )
        kappa_val[1] = -npcsd.abs(np.dot(zeta_val, G[:, 1])) ** (1 / 3) * npcsd.sign(
            np.dot(zeta_val, G[:, 1])
        )

        return kappa_val

    def _F(self, xNI, eta, theta, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Marginal function for NI.

        """

        sigma_tilde = (
            xNI[0] * npcsd.cos(theta)
            + xNI[1] * npcsd.sin(theta)
            + np.sqrt(npcsd.abs(xNI[2]))
        )

        F = xNI[0] ** 4 + xNI[1] ** 4 + npcsd.abs(xNI[2]) ** 3 / sigma_tilde ** 2

        z = eta - self._kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _Cart2NH(self, coords_Cart, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.

        """

        xNI = npcsd.zeros(3)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]

        xNI[0] = alpha
        xNI[1] = xc * npcsd.cos(alpha) + yc * npcsd.sin(alpha)
        xNI[2] = -2 * (yc * npcsd.cos(alpha) - xc * npcsd.sin(alpha)) - alpha * (
            xc * npcsd.cos(alpha) + yc * npcsd.sin(alpha)
        )

        return xNI

    def _NH2ctrl_Cart(self, xNI, uNI, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Get control for Cartesian NI from NH coordinates.       

        """

        uCart = npcsd.zeros(2)

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

            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(
                        action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1]
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

    def compute_LF(self, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)

        xNI = self._Cart2NH(observation)

        sigma = np.sqrt(xNI[0] ** 2 + xNI[1] ** 2) + np.sqrt(npcsd.abs(xNI[2]))

        return xNI[0] ** 4 + xNI[1] ** 4 + npcsd.abs(xNI[2]) ** 3 / sigma ** 2
