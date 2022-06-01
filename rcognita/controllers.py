#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains controllers (agents).

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""
import rospy
from .utilities import dss_sim
from .utilities import rep_mat
from .utilities import uptria2vec
from .utilities import push_vec
from . import models
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from numpy.linalg import lstsq
from numpy import reshape
import warnings
from functools import partial
from shapely.geometry import Point

# For debugging purposes
from tabulate import tabulate
import time

try:
    import sippy
except ModuleNotFoundError:
    warnings.warn_explicit('\nImporting sippy failed. You may still use rcognita, but' +
                  ' without model identification capability. \nRead on how' +
                  ' to install sippy at https://github.com/AIDynamicAction/rcognita\n', 
                  UserWarning, __file__, 33)

def ctrl_selector(t, observation, action_manual, ctrl_nominal, ctrl_benchmarking, mode, constraints=[], line_constraints=[], circ_constraints=[]):
    """
    Main interface for various controllers.

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method.

    Returns
    -------
    action : : array of shape ``[dim_input, ]``.
        Control action.

    """
    
    if mode=='manual': 
        action = action_manual
    elif mode=='nominal': 
        action = ctrl_nominal.compute_action(t, observation)
    else: # Controller for benchmakring
        action = ctrl_benchmarking.compute_action(t, observation, constraints, line_constraints, circ_constraints)
        
    return action

class CtrlRLStab:
    """
    Class of reinforcement learning agents with stabilizing constraints.
    
    Sampling here is similar to the predictive controller agent ``CtrlOptPred``
    
    Needs a nominal controller object ``safe_ctrl`` with a respective Lyapunov function.
    
    Actor
    -----
    
    ``w_actor`` : weights.

    Feature structure is defined via a string flag ``actor_struct``. Read more on features in class description of ``controllers.CtrlOptPred``.
    
    Critic
    -----
    
    ``w_critic`` : weights.
    
    Feature structure is defined via a string flag ``critic_struct``. Read more on features in class description of ``controllers.CtrlOptPred``.
    
    Attributes
    ----------
    mode : : string
        Controller mode. Currently available only JACS, joint actor-critic (stabilizing).   
    
    Read more
    ---------

    Osinenko, P., Beckenbach, L., Göhrt, T., & Streif, S. (2020). A reinforcement learning method with closed-loop stability guarantee. IFAC-PapersOnLine  
    
    """

    def __init__(self,
                 dim_input,
                 dim_output,
                 mode = 'JACS',
                 ctrl_bnds = [],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 is_est_model=0,
                 model_est_stage=1,
                 model_est_period=0.1,
                 buffer_size=20,
                 model_order=3,
                 model_est_checks=0,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 actor_struct='quad-nomix',
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[],   
                 safe_ctrl=[],
                 safe_decay_rate=[]):
        
        """
        Parameter specification largely resembles that of ``CtrlOptPred`` class.
        
        Parameters
        ----------
        dim_input, dim_output : : integer
            Dimension of input and output which should comply with the system-to-be-controlled.  
    
        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default).
        action_init : : array of shape ``[dim_input, ]``   
            Initial action to initialize optimizers.         
        t0 : : number
            Initial value of the controller's internal clock.
        sampling_time : : number
            Controller's sampling time (in seconds).
        sys_rhs, sys_out : : functions        
            Functions that represent the right-hand side, resp., the output of the exogenously passed model.
            The latter could be, for instance, the true model of the system.
            In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
            Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
        prob_noise_pow : : number
            Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
        is_est_model : : number
            Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
        model_est_stage : : number
            Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
        model_est_period : : number
            Time between model estimate updates (in seconds).
        buffer_size : : natural number
            Size of the buffer to store data.
        model_order : : natural number
            Order of the state-space estimation model
            
            .. math::
                \\begin{array}{ll}
        			\\hat x^+ & = A \\hat x + B action, \\newline
        			observation^+  & = C \\hat x + D action.
                \\end{array}             
            
            See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
            When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
            neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``. 
        model_est_checks : : natural number
            Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
            May improve the prediction quality somewhat.
        gamma : : number in (0, 1]
            Discounting factor.
            Characterizes fading of stage objectives along horizon.
        Ncritic : : natural number
            Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
            optimal infinite-horizon objective (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
        critic_period : : number
            The same meaning as ``model_est_period`.` 
        critic_struct, actor_struct : : string
            Choice of the structure of the critic's and actor's features.
            
            Currently available:
                
            .. list-table:: Feature structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quad-lin'
                 - Quadratic-linear
               * - 'quadratic'
                 - Quadratic
               * - 'quad-nomix'
                 - Quadratic, no mixed terms
           
            *Add your specification into the table when customizing the actor and critic*. 
        stage_obj_struct : : string
            Choice of the stage objective structure.
            
            Currently available:
               
            .. list-table:: Running objective structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quadratic'
                 - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
               * - 'biquadratic'
                 - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
                   should be ``[R1, R2]``
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        
        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)            
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys
        
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
        
        self.my_model = models.ModelSS(A, B, C, D, x0est)
        
        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)        
        
        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.actor_struct = actor_struct
        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( (  self.dim_output  + 1 ) *  self.dim_output / 2 + self.dim_output )
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( self.dim_output + 1 ) * self.dim_output / 2 ).astype(int)
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)
 
        self.w_critic_prev = self.Wmin
        self.w_critic_init = np.ones(self.dim_critic)
        
        self.lmbd_prev = 0
        self.lmbd_init = 0
        
        self.lmbd_min = 0
        self.lmbd_max = 1
           
        if self.actor_struct == 'quad-lin':
            self.dim_actor_per_input = int( ( self.dim_output  + 1 ) *  self.dim_output / 2 + self.dim_output ) 
        elif self.actor_struct == 'quadratic':
            self.dim_actor_per_input = int( ( self.dim_output + 1 ) * self.dim_output / 2 )
        elif self.actor_struct == 'quad-nomix':
            self.dim_actor_per_input = self.dim_output
          
        self.dim_actor = self.dim_actor_per_input * self.dim_input   
            
        self.Hmin = -0.5e1*np.ones(self.dim_actor) 
        self.Hmax = 0.5e1*np.ones(self.dim_actor)   
 
        # Stabilizing constraint stuff
        self.safe_ctrl = safe_ctrl          # Safe controller (agent)
        self.safe_decay_rate = safe_decay_rate

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        stage_obj = 0

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return stage_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time
    
    def _actor(self, observation, w_actor):
        """
        Actor: a routine that models the policy.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.actor_struct == 'quad-lin':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ), observation ])
        elif self.actor_struct == 'quadratic':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ) ])   
        elif self.actor_struct == 'quad-nomix':
            regressor_actor = observation * observation        

        return reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ regressor_actor

    def _critic(self, observation, w_critic, lmbd):
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        The parameter ``lmbd`` is needed here specifically for joint actor-critic (stabilizing) a.k.a. JACS.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = observation
        else:
            chi = observation - self.observation_target
        
        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi

        return lmbd * w_critic @ regressor_critic + ( 1 - lmbd ) * self.safe_ctrl.compute_LF( observation )

    def _w_actor_from_action(self, action, observation):
        """
        Compute actor weights from a given action.
        
        The current implementation is for linearly parametrized models so far.

        """
        
        if self.actor_struct == 'quad-lin':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ), observation ])
        elif self.actor_struct == 'quadratic':
            regressor_actor = np.concatenate([ uptria2vec( np.outer(observation, observation) ) ])   
        elif self.actor_struct == 'quad-nomix':
            regressor_actor = observation * observation          
        
        return reshape(lstsq( np.array( [ regressor_actor ] ), np.array( [ action ] ) )[0].T, self.dim_actor ) 

    def _actor_critic_cost(self, w_all):
        """
        Cost (loss) of joint actor-critic (stabilizing) a.k.a. JACS
       
        """        
        
        observation_sqn = self.observation_buffer[-self.Ncritic:,:]
        
        w_critic = w_all[:self.dim_critic]
        # lmbd = w_all[self.dim_critic+1]
        w_actor = w_all[-self.dim_actor:]         
        
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            observation_prev = observation_sqn[k-1, :]
            observation_next = observation_sqn[k, :]
            
            critic_prev = self._critic(observation_prev, w_critic, 1)
            critic_next = self._critic(observation_next, self.w_critic_prev, 1)
            
            action = self._actor(observation_prev, w_actor)
            
            # Temporal difference
            e = critic_prev - self.gamma * critic_next - self.stage_obj(observation_prev, action)
            
            Jc += 1/2 * e**2
        
        return Jc

    def _actor_critic_optimizer(self, observation):
        """
        This method is effectively a wrapper for an optimizer that minimizes :func:`~controllers.CtrlRLStab._actor_critic_cost`.
        It implements the stabilizing constraints.
        
        The variable ``w_all`` here is a stack of actor, critic and auxiliary critic weights.
        
        Important remark: although the current implementation concentrates on a joint coss (loss) of actor-critic (see :func:`~controllers.CtrlRLStab._actor_critic_cost`),
        nothing stops us from doing a usual split actor-critic training.
        The key point of CtrlRLStab agent is its stabilizing constraints that can actually be invoked as a filter (a safety checker), that replaces the action and 
        critic parameters for safe ones if any of the stabilizing constraints are violated.

        """  

        def constr_stab_par_decay(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_curr = self._critic(observation, self.w_critic_prev, self.lmbd_prev)   
            critic_new = self._critic(observation, w_critic, lmbd)
            
            return critic_new - critic_curr
            
        def constr_stab_LF_bound(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:] 
                        
            action = self._actor(observation, w_actor)
            
            observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)  # Euler scheme
            
            critic_next = self._critic(observation_next, w_critic, lmbd) 
            
            return self.safe_ctrl.compute_LF(observation_next) - critic_next        
        
        def constr_stab_decay(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:]   
            
            action = self._actor(observation, w_actor)
            
            observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)  # Euler scheme
            
            critic_new = self._critic(observation, w_critic, lmbd)   
            critic_next = self._critic(observation_next, w_critic, lmbd)   
            
            return critic_next - critic_new + self.safe_decay_rate

        def constr_stab_positive(w_all, observation):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_new = self._critic(observation, w_critic, lmbd)
            
            return - critic_new

        # Constraint violation tolerance
        eps1 = 1e-3
        eps2 = 1e-3
        eps3 = 1e-3
        eps4 = 1e-3
        
        # my_constraints = (
        #     NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, observation ), -np.inf, eps1, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, observation ), -np.inf, eps2, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, observation ), -np.inf, eps3, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, observation ), -np.inf, eps4, keep_feasible=True)
        #     )
        
        my_constraints = (
            NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, observation ), -np.inf, eps1),
            NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, observation ), -np.inf, eps2),
            NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, observation ), -np.inf, eps3),
            NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, observation ), -np.inf, eps4)
            )        

        # Optimization methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        opt_method = 'SLSQP'
        if opt_method == 'trust-constr':
            opt_options = {'maxiter': 10, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            opt_options = {'maxiter': 10, 'maxfev': 10, 'disp': False, 'adaptive': True, 'xatol': 1e-4, 'fatol': 1e-4} # 'disp': True, 'verbose': 2} 
        
        # Bounds are not practically necessary for stabilizing joint actor-critic to function
        # bnds = sp.optimize.Bounds(np.hstack([self.Wmin, self.lmbd_min, self.Hmin]), 
        #                           np.hstack([self.Wmax, self.lmbd_max, self.Hmax]), 
        #                           keep_feasible=True)
        
        # ToDo: make a better routine to determine initial actor weights for the given action
        self.w_actor_init = self._w_actor_from_action( self.safe_ctrl.compute_action_vanila( observation ), observation )
        
        # DEBUG ===================================================================
        # ================================Constraint debugger

        # w_all = np.concatenate([self.w_critic_init, np.array([self.lmbd_init]), self.w_actor_init])
        
        # w_critic = w_all[:self.dim_critic]
        # lmbd = w_all[self.dim_critic]
        # w_actor = w_all[-self.dim_actor:] 
                    
        # action = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._regressor_actor( observation )
        
        # constr_stab_par_decay(w_all, observation)
        # constr_stab_LF_bound(w_all, observation)
        # constr_stab_decay(w_all, observation)
        # constr_stab_positive(w_all, observation)

        # /DEBUG ===================================================================         
        
        # Notice `bounds=bnds` is removed from arguments of minimize.
        # It is because bounds are not practically necessary for stabilizing joint actor-critic to function
        # w_all = minimize(self._actor_critic_cost,
        #                     np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.w_actor_init]),
        #                     method=opt_method, tol=1e-4, constraints=my_constraints, options=opt_options).x
        
        w_all = minimize(self._actor_critic_cost,
                            np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.w_actor_init]),
                            method=opt_method,
                            tol=1e-4,
                            options=opt_options).x        
        
        w_critic = w_all[:self.dim_critic]
        lmbd = w_all[self.dim_critic]
        w_actor = w_all[-self.dim_actor:]       
        
        action = self._actor(observation, w_actor)
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        # dataRow = [constr_stab_par_decay(w_all, observation), constr_stab_LF_bound(w_all, observation), constr_stab_decay(w_all, observation), constr_stab_positive(w_all, observation)]
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ===================================================================        
        
        # Safety checker!
        if constr_stab_par_decay(w_all, observation) >= eps1 or \
            constr_stab_LF_bound(w_all, observation) >= eps2 or \
            constr_stab_decay(w_all, observation) >= eps3 or \
            constr_stab_positive(w_all, observation) >= eps4 :
                
            w_critic = self.w_critic_init
            lmbd = self.lmbd_init

            action = self.safe_ctrl.compute_action_vanila( observation )
            
            w_actor = self._w_actor_from_action( action, observation )
       
        # DEBUG ===================================================================   
        # ================================Put safe controller through        
        # w_critic = self.w_critic_init
        # lmbd = self.lmbd_init
        # action = self.safe_ctrl.compute_action_vanila(observation)        
        # /DEBUG ===================================================================         
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        # dataRow = [constr_stab_par_decay(w_all, observation), constr_stab_LF_bound(w_all, observation), constr_stab_decay(w_all, observation), constr_stab_positive(w_all, observation)]
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ===================================================================  
        
        # STUB ===================================================================   
        # ================================Optimization of one stage_obj + LF_next
        # def J_tmp(action, observation):
        #     observation_next = observation + self.pred_step_size * self.sys_rhs([], observation, action)
        #     return self.safe_ctrl.compute_LF(observation_next) + self.stage_obj(observation_next, action) 
        #     # return self.safe_ctrl.compute_LF(observation_next)
        
        # action = minimize(lambda action: J_tmp(action, observation),
        #               np.zeros(2),
        #               method=opt_method, tol=1e-6, options=opt_options).x        
        
        # /STUB ===================================================================
        
        return w_critic, lmbd, action
        
    def compute_action(self, t, observation):

        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            # Update data buffers
            self.action_buffer = push_vec(self.action_buffer, self.action_curr)
            self.observation_buffer = push_vec(self.observation_buffer, observation)          
            
            w_critic, lmbd, action = self._actor_critic_optimizer(observation)
            
            self.w_critic_prev = w_critic            
            self.lmbd_prev = lmbd

            for k in range(2):
                action[k] = np.clip(action[k], self.action_min[k], self.action_max[k]) 

            self.action_curr = action

            return action
        
        else:
            return self.action_curr        

class CtrlOptPred:
    """
    Class of predictive optimal controllers, primarily model-predictive control and predictive reinforcement learning, that optimize a finite-horizon cost.
    
    Currently, the actor model is trivial: an action is generated directly without additional policy parameters.
        
    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled.
    mode : : string
        Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):
          
        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1
    
           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
             - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
           * - 'SQL' - RL/ADP via stacked Q-learning
             - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\hat \\gamma^{k-1} Q^{\\theta}(y_{N_a}, u_{N_a})`               
        
        Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
        
        *Add your specification into the table when customizing the agent*.    

    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default).
    action_init : : array of shape ``[dim_input, ]``   
        Initial action to initialize optimizers.          
    t0 : : number
        Initial value of the controller's internal clock.
    sampling_time : : number
        Controller's sampling time (in seconds).
    Nactor : : natural number
        Size of prediction horizon :math:`N_a`. 
    pred_step_size : : number
        Prediction step size in :math:`J_a` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon.
    sys_rhs, sys_out : : functions        
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
    is_est_model : : number
        Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
    model_est_stage : : number
        Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
    model_est_period : : number
        Time between model estimate updates (in seconds).
    buffer_size : : natural number
        Size of the buffer to store data.
    model_order : : natural number
        Order of the state-space estimation model
        
        .. math::
            \\begin{array}{ll}
    			\\hat x^+ & = A \\hat x + B action, \\newline
    			observation^+  & = C \\hat x + D action.
            \\end{array}             
        
        See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
        When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons``. 
    model_est_checks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
        May improve the prediction quality somewhat.
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of stage objectives along horizon.
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
    critic_period : : number
        The same meaning as ``model_est_period``. 
    critic_struct : : natural number
        Choice of the structure of the critic's features.
        
        Currently available:
            
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector
       
        *Add your specification into the table when customizing the critic*. 
    stage_obj_struct : : string
        Choice of the stage objective structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
               should be ``[R1, R2]``   
        
        *Pass correct stage objective parameters in* ``stage_obj_pars`` *(as a list)*
        
        *When customizing the stage objective, add your specification into the table above*
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """    
         
    def __init__(self,
                 dim_input,
                 dim_output,
                 mode='MPC',
                 ctrl_bnds=[],
                 action_init = [],
                 t0=0,
                 sampling_time=0.1,
                 Nactor=1,
                 pred_step_size=0.1,
                 sys_rhs=[],
                 sys_out=[],
                 state_sys=[],
                 prob_noise_pow = 1,
                 is_est_model=0,
                 model_est_stage=1,
                 model_est_period=0.1,
                 buffer_size=20,
                 model_order=3,
                 model_est_checks=0,
                 gamma=1,
                 Ncritic=4,
                 critic_period=0.1,
                 critic_struct='quad-nomix',
                 stage_obj_struct='quadratic',
                 stage_obj_pars=[],
                 observation_target=[]):
        """
        Parameters
        ----------
        dim_input, dim_output : : integer
            Dimension of input and output which should comply with the system-to-be-controlled.
        mode : : string
            Controller mode. Currently available (:math:`\\rho` is the stage objective, :math:`\\gamma` is the discounting factor):
              
            .. list-table:: Controller modes
               :widths: 75 25
               :header-rows: 1
        
               * - Mode
                 - Cost function
               * - 'MPC' - Model-predictive control (MPC)
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right)= \\sum_{k=1}^{N_a} \\gamma^{k-1} \\rho(y_k, u_k)`
               * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`\\rho`
                 - :math:`J_a \\left( y_1, \\{action\}_{1}^{N_a}\\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\rho(y_k, u_k) + \\hat Q^{\\theta}(y_{N_a}, u_{N_a})` 
               * - 'SQL' - RL/ADP via stacked Q-learning
                 - :math:`J_a \\left( y_1, \\{action\\}_1^{N_a} \\right) = \\sum_{k=1}^{N_a-1} \\gamma^{k-1} \\hat Q^{\\theta}(y_{N_a}, u_{N_a})`               
            
            Here, :math:`\\theta` are the critic parameters (neural network weights, say) and :math:`y_1` is the current observation.
            
            *Add your specification into the table when customizing the agent* .   
    
        ctrl_bnds : : array of shape ``[dim_input, 2]``
            Box control constraints.
            First element in each row is the lower bound, the second - the upper bound.
            If empty, control is unconstrained (default).
        action_init : : array of shape ``[dim_input, ]``   
            Initial action to initialize optimizers.              
        t0 : : number
            Initial value of the controller's internal clock
        sampling_time : : number
            Controller's sampling time (in seconds)
        Nactor : : natural number
            Size of prediction horizon :math:`N_a` 
        pred_step_size : : number
            Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``sampling_time``. Commonly, equals it, but here left adjustable for
            convenience. Larger prediction step size leads to longer factual horizon.
        sys_rhs, sys_out : : functions        
            Functions that represent the right-hand side, resp., the output of the exogenously passed model.
            The latter could be, for instance, the true model of the system.
            In turn, ``state_sys`` represents the (true) current state of the system and should be updated accordingly.
            Parameters ``sys_rhs, sys_out, state_sys`` are used in those controller modes which rely on them.
        prob_noise_pow : : number
            Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control.   
        is_est_model : : number
            Flag whether to estimate a system model. See :func:`~controllers.CtrlOptPred._estimate_model`. 
        model_est_stage : : number
            Initial time segment to fill the estimator's buffer before applying optimal control (in seconds).      
        model_est_period : : number
            Time between model estimate updates (in seconds).
        buffer_size : : natural number
            Size of the buffer to store data.
        model_order : : natural number
            Order of the state-space estimation model
            
            .. math::
                \\begin{array}{ll}
        			\\hat x^+ & = A \\hat x + B action, \\newline
        			observation^+  & = C \\hat x + D action.
                \\end{array}             
            
            See :func:`~controllers.CtrlOptPred._estimate_model`. This is just a particular model estimator.
            When customizing, :func:`~controllers.CtrlOptPred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
            neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons`` 
        model_est_checks : : natural number
            Estimated model parameters can be stored in stacks and the best among the ``model_est_checks`` last ones is picked.
            May improve the prediction quality somewhat.
        gamma : : number in (0, 1]
            Discounting factor.
            Characterizes fading of stage objectives along horizon.
        Ncritic : : natural number
            Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
            optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer.
        critic_period : : number
            The same meaning as ``model_est_period``. 
        critic_struct : : natural number
            Choice of the structure of the critic's features.
            
            Currently available:
                
            .. list-table:: Critic feature structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quad-lin'
                 - Quadratic-linear
               * - 'quadratic'
                 - Quadratic
               * - 'quad-nomix'
                 - Quadratic, no mixed terms
               * - 'quad-mix'
                 - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
                   where :math:`w` is the critic's weights
           
            *Add your specification into the table when customizing the critic*.
        stage_obj_struct : : string
            Choice of the stage objective structure.
            
            Currently available:
               
            .. list-table:: Running objective structures
               :widths: 10 90
               :header-rows: 1
        
               * - Mode
                 - Structure
               * - 'quadratic'
                 - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars`` should be ``[R1]``
               * - 'biquadratic'
                 - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [observation, action]`, ``stage_obj_pars``
                   should be ``[R1, R2]``
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.mode = mode

        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        # Controller: common
        self.Nactor = Nactor 
        self.pred_step_size = pred_step_size
        
        self.action_min = np.array( ctrl_bnds[:,0] )
        self.action_max = np.array( ctrl_bnds[:,1] )
        self.action_sqn_min = rep_mat(self.action_min, 1, Nactor)
        self.action_sqn_max = rep_mat(self.action_max, 1, Nactor) 
        
        if len(action_init) == 0:
            self.action_curr = self.action_min/10
            self.action_sqn_init = rep_mat( self.action_min/10 , 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = rep_mat( action_init , 1, self.Nactor)
        
        self.action_buffer = np.zeros( [buffer_size, dim_input] )
        self.observation_buffer = np.zeros( [buffer_size, dim_output] )        
        
        # Exogeneous model's things
        self.sys_rhs = sys_rhs
        self.sys_out = sys_out
        self.state_sys = state_sys
        
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
        
        A = np.zeros( [self.model_order, self.model_order] )
        B = np.zeros( [self.model_order, self.dim_input] )
        C = np.zeros( [self.dim_output, self.model_order] )
        D = np.zeros( [self.dim_output, self.dim_input] )
        x0est = np.zeros( self.model_order )
        
        self.my_model = models.ModelSS(A, B, C, D, x0est)
        
        self.model_stack = []
        for k in range(self.model_est_checks):
            self.model_stack.append(self.my_model)        
        
        # RL elements
        self.critic_clock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.buffer_size-1]) # Clip critic buffer size
        self.critic_period = critic_period
        self.critic_struct = critic_struct
        self.stage_obj_struct = stage_obj_struct
        self.stage_obj_pars = stage_obj_pars
        self.observation_target = observation_target
        
        self.accum_obj_val = 0

        if self.critic_struct == 'quad-lin':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 + (self.dim_output + self.dim_input) ) 
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic) 
        elif self.critic_struct == 'quadratic':
            self.dim_critic = int( ( ( self.dim_output + self.dim_input ) + 1 ) * ( self.dim_output + self.dim_input )/2 )
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-nomix':
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)    
        elif self.critic_struct == 'quad-mix':
            self.dim_critic = int( self.dim_output + self.dim_output * self.dim_input + self.dim_input )
            self.Wmin = -1e3*np.ones(self.dim_critic) 
            self.Wmax = 1e3*np.ones(self.dim_critic)
            
        self.w_critic_prev = np.zeros(self.dim_critic)  
        self.w_critic_init = self.w_critic_prev
        
        # self.big_number = 1e4
        self.counter = 0
        self.total_time = 0
        self.max_time = 0
        self.start_iter_time = rospy.get_time()

    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained.
        
        """
        self.ctrl_clock = t0
        self.action_curr = self.action_min/10
    
    def receive_sys_state(self, state):
        """
        Fetch exogenous model state. Used in some controller modes. See class documentation.

        """
        self.state_sys = state
    
    def stage_obj(self, observation, action):
        """
        Stage (equivalently, instantaneous or running) objective. Depending on the context, it is also called utility, reward, running cost etc.
        
        See class documentation.
        """
        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        stage_obj = 0
        #print('ACTION:', action)

        if self.stage_obj_struct == 'quadratic':
            R1 = self.stage_obj_pars[0]
            #print('SHAPES:', chi.shape, R1.shape)
            stage_obj = chi @ R1 @ chi
        elif self.stage_obj_struct == 'biquadratic':
            R1 = self.stage_obj_pars[0]
            R2 = self.stage_obj_pars[1]
            stage_obj = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return stage_obj
        
    def upd_accum_obj(self, observation, action):
        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead).
        
        """
        # self.iter_time = time.time()
        # iter_time = self.iter_time - self.start_iter_time
        # self.accum_obj_val += self.stage_obj(observation, action) * iter_time
        # self.start_iter_time = time.time()
        self.accum_obj_val += self.stage_obj(observation, action)*self.sampling_time
    
    def _estimate_model(self, t, observation):
        """
        Estimate model parameters by accumulating data buffers ``action_buffer`` and ``observation_buffer``.
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update buffers when using RL or requiring estimated model
            if self.is_est_model or self.mode in ['RQL', 'SQL']:
                time_in_est_period = t - self.est_clock
                
                # Estimate model if required
                if (time_in_est_period >= self.model_est_period) and self.is_est_model:
                    # Update model estimator's internal clock
                    self.est_clock = t
                    
                    try:
                        # Using Github:CPCLAB-UNIPI/SIPPY 
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(self.observation_buffer, self.action_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.buffer_size/12),
                                                            # SS_p=int(self.buffer_size/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sampling_time)
                        
                        self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # ToDo: train an NN via Torch
                        # NN_wgts = NN_train(...)
                        
                    except:
                        print('Model estimation problem')
                        self.my_model.upd_pars(np.zeros( [self.model_order, self.model_order] ),
                                                np.zeros( [self.model_order, self.dim_input] ),
                                                np.zeros( [self.dim_output, self.model_order] ),
                                                np.zeros( [self.dim_output, self.dim_input] ) )
                    
                    # Model checks
                    if self.model_est_checks > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        tot_abs_err_curr = 1e8
                        for k in range(self.model_est_checks):
                            A, B, C, D = self.model_stack[k].A, self.model_stack[k].B, self.model_stack[k].C, self.model_stack[k].D
                            x0est,_,_,_ = np.linalg.lstsq(C, observation)
                            Yest,_ = dss_sim(A, B, C, D, self.action_buffer, x0est, observation)
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
                            
                            tot_abs_err = np.sum( np.abs( mean_err ) )
                            if tot_abs_err <= tot_abs_err_curr:
                                tot_abs_err_curr = tot_abs_err
                                self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
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
            x0est,_,_,_ = np.linalg.lstsq(self.my_model.C, observation)
            self.my_model.updateIC(x0est)
     
            if t >= self.model_est_stage:
                    # Drop probing noise
                    self.is_prob_noise = 0 

    def _critic(self, observation, action, w_critic):
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            chi = np.concatenate([observation, action])
        else:
            chi = np.concatenate([observation - self.observation_target, action])
        
        if self.critic_struct == 'quad-lin':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            regressor_critic = np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            regressor_critic = chi * chi
        elif self.critic_struct == 'quad-mix':
            regressor_critic = np.concatenate([ observation**2, np.kron(observation, action), action**2 ]) 

        return w_critic @ regressor_critic
    
    def _critic_cost(self, w_critic):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0
        
        for k in range(self.buffer_size-1, self.buffer_size - self.Ncritic, -1):
            observation_prev = self.observation_buffer[k-1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k-1, :]
            action_next = self.action_buffer[k, :]
            # Temporal difference
            
            critic_prev = self._critic(observation_prev, action_prev, w_critic)
            critic_next = self._critic(observation_next, action_next, self.w_critic_prev)      
            st_obj = self.stage_obj(observation_prev, action_prev)      

            e = critic_prev - self.gamma * critic_next - st_obj
            Jc += 1/2 * e**2
        return Jc
        
        
    def _critic_optimizer(self):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._critic_cost`.

        """        
        
        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
        
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)
    
        w_critic = minimize(self._critic_cost, self.w_critic_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x
        print('w_critic after optimization', w_critic)
        # DEBUG ===================================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( w_critic )
        # /DEBUG ==================================================================
        
        return w_critic
    
    def _actor_cost(self, action_sqn, observation):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """
        
        my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        #print(my_action_sqn)
        observation_sqn = np.zeros([self.Nactor, self.dim_output])
        
        # System output prediction
        if not self.is_est_model:    # Via exogenously passed model
            observation_sqn[0, :] = observation
            state = self.state_sys
            for k in range(1, self.Nactor):
                # state = get_next_state(state, my_action_sqn[k-1, :], delta)         TODO
                state = state + self.pred_step_size * self.sys_rhs([], state, my_action_sqn[k-1, :])  # Euler scheme
                
                observation_sqn[k, :] = self.sys_out(state)

        elif self.is_est_model:    # Via estimated model
            my_action_sqn_upsampled = my_action_sqn.repeat(int(self.pred_step_size/self.sampling_time), axis=0)
            observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
            observation_sqn = observation_sqn_upsampled[::int(self.pred_step_size/self.sampling_time)]
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
        elif self.mode=='RQL':     # RL: Q-learning with Ncritic-1 roll-outs of stage objectives
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.stage_obj(observation_sqn[k, :], my_action_sqn[k, :])
             J += self._critic(observation_sqn[-1, :], my_action_sqn[-1, :], self.w_critic)
        elif self.mode=='SQL':     # RL: stacked Q-learning
             for k in range(self.Nactor): 
                Q = self._critic(observation_sqn[k, :], my_action_sqn[k, :], self.w_critic)
                
                # With state constraints via indicator function
                # Q = w_critic @ self._regressor_critic( observation_sqn[k, :], my_action_sqn[k, :] ) + state_constraint_indicator(observation_sqn[k, 0])
                
                # DEBUG ===================================================================
                # =========================================================================
                # R  = '\033[31m'
                # Bl  = '\033[30m'
                # if state_constraint_indicator(observation_sqn[k, 0]) > 1:
                #     print(R+str(state_constraint_indicator(observation_sqn[k, 0]))+Bl)
                # /DEBUG ==================================================================                 
                
                J += Q 

        return J
    
    def _actor_optimizer(self, observation, constraints=[], line_constraints=None, circ_constraints=None):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._actor_cost`.
        See class documentation.
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~controllers.CtrlOptPred._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm.

        """

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(action_sqn, idx):
            
        #     my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])
            
        #     observation_sqn = np.zeros([idx, self.dim_output])    
            
        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         observation_sqn[0, :] = observation
        #         state = self.state_sys
        #         Y[0, :] = observation
        #         x = self.x_s
        #         for k in range(1, idx):
        #             # state = get_next_state(state, my_action_sqn[k-1, :], delta)
        #             state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #             observation_sqn[k, :] = self.sys_out(state)            
            
        #     return observation_sqn[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda action_sqn: state_constraint(action_sqn, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powel

        # def constraint(action_sqn, y, constr):
        #     if constraints is None:
        #         return None
        #     res = y
        #     res_constr = []
        #     my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        #     for i in range(0, self.Nactor, 1):
        #         res = res + self.pred_step_size * self.sys_rhs([], res, my_action_sqn[i, :]) #предсказание следующих шагов

        #         cons = []
        #         # for constr in constraints:
        #         #     #cons.append(-constr(res))
        #         #     cons.append(constr.contains(Point(res)))
        #         f1 = constr.contains(Point(res)) #np.sum(cons)
        #         if f1 > 0:
        #             res_constr.append(-1)
        #         else:
        #             res_constr.append(1)
        #     return res_constr

        # def constraint(action_sqn, y, x1, y1, x2, y2):
        #     res = y
        #     #print('current state:', res)
        #     res_constr = []
        #     my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        #     for i in range(1, self.Nactor, 1):
        #         res = res + self.pred_step_size * self.sys_rhs([], res, my_action_sqn[i-1, :]) #предсказание следующих шагов
        #         xk = res[0] - (x1+x2)/2
        #         yk = res[1] - (y1+y2)/2
        #         f1 = (2*xk/abs(x2-x1))**64 + (2*yk/abs(y2-y1))**64 - 1
        #         res_constr.append(f1)
        #     # if np.sum(np.array(res_constr) < 0) > 0:
        #     #     print('predicted entering the prohibited zone', np.array(res_constr))
        #     #     raise RuntimeError('predicted entering the prohibited zone')
        #     return res_constr

        # def constraint_circ(action_sqn, y, x1, y1, r):
        #     res = y
        #     #print('current state:', res)
        #     res_constr = []
        #     my_action_sqn = np.reshape(action_sqn, [self.Nactor, self.dim_input])
        #     for i in range(1, self.Nactor, 1):
        #         res = res + self.pred_step_size * self.sys_rhs([], res, my_action_sqn[i-1, :])  #предсказание следующих шагов
        #         xk = res[0] - x1
        #         yk = res[1] - y1
        #         f1 = (2*xk/r)**2 + (2*yk/r)**2 - 1
        #         res_constr.append(f1)
        #     return res_constr
        #print('INSIDE CONSTRS', constraints)

        # cons = []
        # f = 0
        # for constr in constraints:
        #         #cons.append(-constr(res))
        #     cons.append(constr.contains(Point(observation[:2])))
        #     f1 = np.sum(cons)
        #     if f1 > 0:
        #         f = -1
        #     else:
        #         f = 1
        # if f < 0:
        #     print('COLLISION IN CONTROLLERS!!!')
        
        actor_opt_method = 'SLSQP'
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {'maxiter': 70, 'maxfev': 1000, 'disp': False, 'xatol': 1e-4, 'fatol': 1e-4} # 'disp': True, 'verbose': 2} 
       
        isGlobOpt = 0
        
        my_action_sqn_init = np.reshape(self.action_sqn_init, [self.Nactor*self.dim_input,])
        
        bnds = sp.optimize.Bounds(self.action_sqn_min, self.action_sqn_max, keep_feasible=True)

        final_constraints = []


        def constrs(u, constraints, y):
            res = y
            cons = []
            for constr in constraints:
                cons.append(constr(res))
            f1 = np.max(cons)
            start_in_danger = f1 > 0.

            res_constr = []
            f1 = -1
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(u), len(y))
            # print('u = ', u)
            # print('y = ', y)
            my_action_sqn = np.reshape(u, [self.Nactor, self.dim_input])
            for i in range(1, self.Nactor, 1):
                if f1 > 0. and not start_in_danger:
                    res_constr.append(f1)
                    continue
                res = res + self.pred_step_size * self.sys_rhs([], res, my_action_sqn[i-1, :])
                cons = []
                for constr in constraints:
                    cons.append(constr(res))
                f1 = np.max(cons)
                res_constr.append(f1)
            #print('lens', len(my_action_sqn), len(res_constr))
            # print('res_constr = ', res_constr)
            # print('==================================')
            return res_constr


        if len(constraints) > 0:
            final_constraints.append(sp.optimize.NonlinearConstraint(partial(constrs, constraints=constraints, y=observation), -np.inf, 0))

        try:
            start = rospy.get_time()
            if isGlobOpt:
                minimizer_kwargs = {'method': actor_opt_method, 'bounds': bnds, 
                'constraints': final_constraints, 'tol': 1e-7, 'options': actor_opt_options}
                action_sqn = basinhopping(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                          my_action_sqn_init,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter = 10).x
            else:
                action_sqn = minimize(lambda action_sqn: self._actor_cost(action_sqn, observation),
                                      my_action_sqn_init,
                                      method=actor_opt_method,
                                      tol=1e-5,
                                      bounds=bnds,
                                      constraints=final_constraints,
                                      options=actor_opt_options).x   
            final_time = rospy.get_time() - start
            self.total_time += final_time
            self.counter += 1
            if final_time > self.max_time:
                self.max_time = final_time
            print('minimizer working time:', final_time, '||| avg time:', self.total_time / self.counter, '||| max time:', self.max_time)
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            action_sqn = my_action_sqn_init
        
        # DEBUG ===================================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # my_action_sqn = np.reshape(action_sqn, [N, self.dim_input])    
        # my_action_sqn_upsampled = my_action_sqn.repeat(int(delta/self.sampling_time), axis=0)
        # observation_sqn_upsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, my_action_sqn_upsampled, self.my_model.x0est, observation)
        # observation_sqn = observation_sqn_upsampled[::int(delta/self.sampling_time)]
        # Yt = np.zeros([N, self.dim_output])
        # Yt[0, :] = observation
        # state = self.state_sys
        # for k in range(1, Nactor):
        #     state = state + delta * self.sys_rhs([], state, my_action_sqn[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sys_out(state)           
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
        # dataRow = []
        # for k in range(dim_output):
        #     dataRow.append( np.mean(observation_sqn[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ==================================================================     
        
        return action_sqn[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, observation, constraints=None, line_constrs=None, circ_constrs=None):
        """
        Main method. See class documentation.
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~controllers.CtrlOptPred._actor_cost`, here.

        """       
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            if self.mode == 'MPC':  
                
                # Apply control when model estimation phase is over  
                if self.is_prob_noise and self.is_est_model:
                    return self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                
                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation, constraints, line_constrs, circ_constrs)

                elif self.mode=='MPC':
                    action = self._actor_optimizer(observation, constraints, line_constrs, circ_constrs)
                    
            elif self.mode in ['RQL', 'SQL']:
                # Critic
                timeInCriticPeriod = t - self.critic_clock
                
                # Update data buffers
                self.action_buffer = push_vec(self.action_buffer, self.action_curr)
                self.observation_buffer = push_vec(self.observation_buffer, observation)

                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t
                    
                    self.w_critic = self._critic_optimizer()
                    self.w_critic_prev = self.w_critic
                    
                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.w_critic_init = self.w_critic_prev
                    
                else:
                    self.w_critic = self.w_critic_prev
                    
                # Actor. Apply control when model estimation phase is over
                if self.is_prob_noise and self.is_est_model:
                    action = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and self.is_est_model:
                    action = self._actor_optimizer(observation)
                    
                elif self.mode in ['RQL', 'SQL']:
                    #print('WEIGHTS', self.w_critic)
                    action = self._actor_optimizer(observation, constraints, line_constrs, circ_constrs) 
            
            self.action_curr = action
            
            return action    
    
        else:
            return self.action_curr
        
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
    
    def __init__(self, m, I, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        self.action_curr = np.zeros(2)
   
    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_curr = np.zeros(2)   
    
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
    
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3
        
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3
        
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3  
    
        return nablaF
    
    def _kappa(self, xNI, theta): 
        """
        Stabilizing controller for NI-part.

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
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation.

        """
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        
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
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
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
        Get control for Cartesian NI from NH coordinates.
        See Section VIII.A in [[1]_].
        
        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`.
        
        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
        

        """

        uCart = np.zeros(2)
        
        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
        uCart[1] = self.I * uNI[0]
        
        return uCart

    def compute_action(self, t, observation):
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
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t
            
            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH( observation ) 
            theta_star = self._minimizer_theta(xNI, eta)
            kappa_val = self._kappa(xNI, theta_star)
            z = eta - kappa_val
            uNI = - self.ctrl_gain * z
            action = self._NH2ctrl_Cart(xNI, eta, uNI)
            
            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])           
            
            self.action_curr = action

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
            return self.action_curr

    def compute_action_vanila(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobot.compute_action`, but without invoking the internal clock.

        """
        
        xNI, eta = self._Cart2NH( observation ) 
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = - self.ctrl_gain * z
        action = self._NH2ctrl_Cart(xNI, eta, uNI)
        
        self.action_curr = action
        
        return action

    def compute_LF(self, observation):
        
        xNI, eta = self._Cart2NH( observation ) 
        theta_star = self._minimizer_theta(xNI, eta)
        
        return self._Fc(xNI, eta, theta_star)
    
class CtrlNominal3WRobotNI:
    """
    Nominal parking controller for NI using disassembled subgradients.
    
    """
    
    def __init__(self, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time
        
        self.action_curr = np.zeros(2)
   
    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation.
        
        """
        self.ctrl_clock = t0
        self.action_curr = np.zeros(2)   
    
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
    
    
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt(abs(xNI[2]));
        
        nablaL = np.zeros(3)
        
        nablaL[0] = 4*xNI[0]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[0];
        nablaL[1] = 4*xNI[1]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[1]; 
        nablaL[2] = 3 * np.abs(xNI[2])**2 * np.sign(xNI[2]) + np.abs(xNI[2])**3 / sigma**3 * 1/np.sqrt(np.abs(xNI[2])) * np.sign(xNI[2]);
    
        theta = 0
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3  
    
        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL
    
    def _kappa(self, xNI): 
        """
        Stabilizing controller for NI-part.

        """
        kappa_val = np.zeros(2)
        
        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])
                         
        zeta_val = self._zeta(xNI)
        
        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )
        
        return kappa_val
    
    def _F(self, xNI, eta, theta):
        """
        Marginal function for NI.

        """
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        
        z = eta - self._kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)
      
    def _Cart2NH(self, coords_Cart): 
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates.

        """
        
        xNI = np.zeros(3)
        
        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        
        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )
        
        return xNI
  
    def _NH2ctrl_Cart(self, xNI, uNI): 
        """
        Get control for Cartesian NI from NH coordinates.       

        """

        uCart = np.zeros(2)
        
        uCart[0] = uNI[1] + 1/2 * uNI[0] * ( xNI[2] + xNI[0] * xNI[1] )
        uCart[1] = uNI[0]
        
        return uCart

    def compute_action(self, t, observation):
        """
        Compute sampled action.
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t
            
            xNI = self._Cart2NH( observation ) 
            kappa_val = self._kappa(xNI)
            uNI = self.ctrl_gain * kappa_val
            action = self._NH2ctrl_Cart(xNI, uNI)
            
            if self.ctrl_bnds.any():
                for k in range(2):
                    action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])           
            
            self.action_curr = action
            
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
            return self.action_curr

    def compute_action_vanila(self, observation):
        """
        Same as :func:`~CtrlNominal3WRobotNI.compute_action`, but without invoking the internal clock.

        """
        
        xNI = self._Cart2NH( observation ) 
        kappa_val = self._kappa(xNI)
        uNI = self.ctrl_gain * kappa_val
        action = self._NH2ctrl_Cart(xNI, uNI)
        
        self.action_curr = action
        
        return action

    def compute_LF(self, observation):
        
        xNI = self._Cart2NH( observation ) 
        
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt( np.abs(xNI[2]) )
        
        return xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma**2
