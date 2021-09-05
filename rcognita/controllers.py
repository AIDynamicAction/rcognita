#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains controllers (agents)

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

from .utilities import dss_sim
from .utilities import rep_mat
from .utilities import uptria2vec
from .utilities import push_vec
import numpy as np
import scipy as sp
from numpy.random import rand
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint
from numpy.linalg import lstsq
from numpy import reshape

# For debugging purposes
from tabulate import tabulate

# System identification packages
# import ssid  # Github:OsinenkoP/pyN4SID, fork of Githug:AndyLamperski/pyN4SID, with some errors fixed
# import sippy  # Github:CPCLAB-UNIPI/SIPPY

# [EXPERIMENTAL] Use MATLAB's system identification toolbox instead of ssid and sippy
# Compatible MATLAB Runtime and system identification toolbox must be installed
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'~/MATLAB/RL/ENDICart',nargout=0)

def ctrl_selector(t, y, uMan, ctrl_nominal, ctrl_benchmarking, mode):
    """
    Main interface for various controllers

    Parameters
    ----------
    mode : : string
        Controller mode as acronym of the respective control method

    Returns
    -------
    u : : array of shape ``[dim_input, ]``
        Control action

    """
    
    if mode=='manual': 
        u = uMan
    elif mode=='nominal': 
        u = ctrl_nominal.compute_action(t, y)
    else: # Controller for benchmakring
        u = ctrl_benchmarking.compute_action(t, y)
        
    return u

class CtrlRLStab:
    """
    Class of reinforcement learning agents with stabilizing constraints.
    
    Sampling here is similar to the predictive controller agent ``ctrl_opt_pred``
    
    Needs a nominal controller object ``safe_ctrl`` with a respective Lyapunov function.
    
    Actor
    -----
    
    ``w_actor`` : weights
    
    ``_psi``: regressor
    
    ``_psi`` is a vector, not a matrix. So, if the environment is multi-input, the input is actually computed as
    
    ``u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( y )``
    
    where ``y`` is the output.
    
    Actor structure is defined via a string flag ``actor_struct``. Structures are analogous to the critic ones - read more in class description of ``controllers.ctrl_opt_pred``
    
    Critic
    -----
    
    ``w_critic`` : weights
    
    ``_phi``: regressor   
    
    Attributes
    ----------
    mode : : string
        Controller mode. Currently available only JACS, joint actor-critic (stabilizing)   
    
    Read more
    ---------

    Osinenko, P., Beckenbach, L., Göhrt, T., & Streif, S. (2020). A reinforcement learning method with closed-loop stability guarantee. IFAC-PapersOnLine  
    
    """
    def __init__(self, dim_input, dim_output, mode='JACS', ctrl_bnds=[], t0=0, sampling_time=0.1, Nactor=1, pred_step_size=0.1,
                 sys_rhs=[], sys_out=[], x_sys=[], prob_noise_pow = 1, is_est_model=0, model_est_stage=1, model_est_period=0.1, buffer_size=20, model_order=3, model_est_checks=0,
                 gamma=1, Ncritic=4, critic_period=0.1, critic_struct='quad-nomix', actor_struct='quad-nomix', rcost_struct='quadratic', rcost_pars=[], y_target=[],
                 safe_ctrl=[], safe_decay_rate=[]):
        
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
        
        self.my_model = self.model(A, B, C, D, x0est)
        
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
        self.rcost_struct = rcost_struct
        self.rcost_pars = rcost_pars
        self.y_target = y_target
        
        self.icost_val = 0

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

    class model:
        """
            Class of estimated models
            
            So far, uses just the state-space structure:
                
        .. math::
            \\begin{array}{ll}
    			\\hat x^+ & = A \\hat x + B u \\newline
    			y^+  & = C \\hat x + D u,
            \\end{array}                 
            
        Attributes
        ---------- 
        A, B, C, D : : arrays of proper shape
            State-space model parameters
        x0set : : array
            Initial state estimate
            
        **When introducing your custom model estimator, adjust this class**    
            
        """
        
        def __init__(self, A, B, C, D, x0est):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.x0est = x0est
            
        def upd_pars(self, Anew, Bnew, Cnew, Dnew):
            self.A = Anew
            self.B = Bnew
            self.C = Cnew
            self.D = Dnew
            
        def updateIC(self, x0setNew):
            self.x0set = x0setNew    

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

        if self.rcost_struct == 'quadratic':
            R1 = self.rcost_pars[0]
            r = chi @ R1 @ chi
        elif self.rcost_struct == 'biquadratic':
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

    def _phi(self, y):
        """
        Feature vector of the critic

        """
        if self.y_target == []:
            chi = y
        else:
            chi = y - self.y_target
        
        if self.critic_struct == 'quad-lin':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            return chi * chi
        
    def _psi(self, y):
        """
        Feature vector of the actor

        """

        chi = y

        if self.actor_struct == 'quad-lin':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.actor_struct == 'quadratic':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.actor_struct == 'quad-nomix':
            return chi * chi        

    def _actor_critic_cost(self, W_lmbd_u):
        """
        Joint actor-critic cost function
       
        """        
        
        Y = self.ybuffer[-self.Ncritic:,:]
        
        w_critic = W_lmbd_u[:self.dim_critic]
        # lmbd = W_lmbd_u[self.dim_critic+1]
        w_actor = W_lmbd_u[-self.dim_actor:]         
        
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            yPrev = Y[k-1, :]
            yNext = Y[k, :]
            
            critic_prev = w_critic @ self._phi( yPrev )
            critic_next = self.w_critic_prev @ self._phi( yNext )
            
            u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( yPrev )
            
            # Temporal difference
            e = critic_prev - self.gamma * critic_next - self.rcost(yPrev, u)
            
            Jc += 1/2 * e**2
        
        return Jc

    def _actor_critic(self, y):
        """
        This method is effectively a wrapper for an optimizer that minimizes :func:`~controllers.ctrl_RL_stab._actor_critic_cost`.
        It implements the stabilizing constraints
        
        The variable ``w_all`` here is a stack of actor, critic and auxiliary critic weights
        

        """  

        def constr_stab_par_decay(w_all, y):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_curr = self.lmbd_prev * self.w_critic_prev @ self._phi( y ) + ( 1 - self.lmbd_prev ) * self.safe_ctrl.compute_LF(y)
            critic_new = lmbd * w_critic @ self._phi( y ) + ( 1 - lmbd ) * self.safe_ctrl.compute_LF(y)
            
            return critic_new - critic_curr
            
        def constr_stab_LF_bound(w_all, y):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:] 
                        
            u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( y )
            
            y_next = y + self.pred_step_size * self.sys_rhs([], y, u)  # Euler scheme
            
            critic_next = lmbd * w_critic @ self._phi( y_next ) + ( 1 - lmbd ) * self.safe_ctrl.compute_LF( y_next )
            
            return self.safe_ctrl.compute_LF(y_next) - critic_next        
        
        def constr_stab_decay(w_all, y):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            w_actor = w_all[-self.dim_actor:]   
            
            u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( y )
            
            y_next = y + self.pred_step_size * self.sys_rhs([], y, u)  # Euler scheme
            
            critic_new = lmbd * w_critic @ self._phi( y ) + ( 1 - lmbd ) * self.safe_ctrl.compute_LF(y)
            critic_next = lmbd * w_critic @ self._phi( y_next ) + ( 1 - lmbd ) * self.safe_ctrl.compute_LF( y_next )
            
            return critic_next - critic_new + self.safe_decay_rate

        def constr_stab_positive(w_all, y):
            w_critic = w_all[:self.dim_critic]
            lmbd = w_all[self.dim_critic]
            
            critic_new = lmbd * w_critic @ self._phi( y ) + ( 1 - lmbd ) * self.safe_ctrl.compute_LF(y)
            
            return - critic_new

        # Constraint violation tolerance
        eps1 = 1e-3
        eps2 = 1e-3
        eps3 = 1e-3
        eps4 = 1e-3
        
        # my_constraints = (
        #     NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, y ), -np.inf, eps1, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, y ), -np.inf, eps2, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, y ), -np.inf, eps3, keep_feasible=True),
        #     NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, y ), -np.inf, eps4, keep_feasible=True)
        #     )
        
        my_constraints = (
            NonlinearConstraint(lambda w_all: constr_stab_par_decay( w_all, y ), -np.inf, eps1),
            NonlinearConstraint(lambda w_all: constr_stab_LF_bound( w_all, y ), -np.inf, eps2),
            NonlinearConstraint(lambda w_all: constr_stab_decay( w_all, y ), -np.inf, eps3),
            NonlinearConstraint(lambda w_all: constr_stab_positive( w_all, y ), -np.inf, eps4)
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
        
        self.Hinit = reshape( lstsq( np.array( [ self._psi( y ) ] ), np.array( [ self.safe_ctrl.compute_action_vanila(y) ] ) )[0].T, self.dim_actor )
       
        # DEBUG ===================================================================
        # ================================Constraint debugger

        # w_all = np.concatenate([self.w_critic_init, np.array([self.lmbd_init]), self.Hinit])
        
        # w_critic = w_all[:self.dim_critic]
        # lmbd = w_all[self.dim_critic]
        # w_actor = w_all[-self.dim_actor:] 
                    
        # u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( y )
        
        # constr_stab_par_decay(w_all, y)
        # constr_stab_LF_bound(w_all, y)
        # constr_stab_decay(w_all, y)
        # constr_stab_positive(w_all, y)

        # /DEBUG ===================================================================         
        
        # Notice `bounds=bnds` is removed from arguments of minimize.
        # It is because bounds are not practically necessary for stabilizing joint actor-critic to function
        # w_all = minimize(self._actor_critic_cost,
        #                     np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.Hinit]),
        #                     method=opt_method, tol=1e-4, constraints=my_constraints, options=opt_options).x
        
        w_all = minimize(self._actor_critic_cost,
                            np.hstack([self.w_critic_init,np.array([self.lmbd_init]),self.Hinit]),
                            method=opt_method, tol=1e-4, options=opt_options).x        
        
        w_critic = w_all[:self.dim_critic]
        lmbd = w_all[self.dim_critic]
        w_actor = w_all[-self.dim_actor:]       
        
        u = reshape(w_actor, (self.dim_input, self.dim_actor_per_input)) @ self._psi( y )       
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        # dataRow = [constr_stab_par_decay(w_all, y), constr_stab_LF_bound(w_all, y), constr_stab_decay(w_all, y), constr_stab_positive(w_all, y)]
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ===================================================================        
        
        # Safety checker!
        if constr_stab_par_decay(w_all, y) >= eps1 or \
            constr_stab_LF_bound(w_all, y) >= eps2 or \
            constr_stab_decay(w_all, y) >= eps3 or \
            constr_stab_positive(w_all, y) >= eps4 :
                
            w_critic = self.w_critic_init
            lmbd = self.lmbd_init
            u = self.safe_ctrl.compute_action_vanila(y)
            w_actor = reshape( lstsq( np.array( [ self._psi( y ) ] ), np.array( [ u ] ) )[0].T, self.dim_actor )
       
        # DEBUG ===================================================================   
        # ================================Put safe controller through        
        # w_critic = self.w_critic_init
        # lmbd = self.lmbd_init
        # u = self.safe_ctrl.compute_action_vanila(y)        
        # /DEBUG ===================================================================         
        
        # DEBUG ===================================================================   
        # ================================Constraint debugger
        R  = '\033[31m'
        Bl  = '\033[30m'
        headerRow = ['par_decay', 'LF_bound', 'decay', 'stab_positive']  
        dataRow = [constr_stab_par_decay(w_all, y), constr_stab_LF_bound(w_all, y), constr_stab_decay(w_all, y), constr_stab_positive(w_all, y)]
        rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        print(R+table+Bl)
        # /DEBUG ===================================================================  
        
        # STUB ===================================================================   
        # ================================Optimization of one rcost + LF_next
        def J_tmp(u, y):
            y_next = y + self.pred_step_size * self.sys_rhs([], y, u)
            return self.safe_ctrl.compute_LF(y_next) + self.rcost(y_next, u) 
            # return self.safe_ctrl.compute_LF(y_next)
        
        u = minimize(lambda u: J_tmp(u, y),
                      np.zeros(2),
                      method=opt_method, tol=1e-6, options=opt_options).x        
        
        # /STUB ===================================================================
        
        return w_critic, lmbd, u
        
    def compute_action(self, t, y):

        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update controller's internal clock
            self.ctrl_clock = t
            
            # Update data buffers
            self.ubuffer = push_vec(self.ubuffer, self.uCurr)
            self.ybuffer = push_vec(self.ybuffer, y)          
            
            w_critic, lmbd, u = self._actor_critic(y)
            
            self.w_critic_prev = w_critic            
            self.lmbd_prev = lmbd

            for k in range(2):
                u[k] = np.clip(u[k], self.uMin[k], self.uMax[k]) 

            self.uCurr = u

            return u
        
        else:
            return self.uCurr        

class CtrlOptPred:
    """
    Class of predictive optimal controllers, primarily MPC and predictive RL, that optimize a finite-horizon cost
        
    Attributes
    ----------
    dim_input, dim_output : : integer
        Dimension of input and output which should comply with the system-to-be-controlled
    mode : : string
        Controller mode. Currently available (:math:`r` is the running cost, :math:`\\gamma` is the discounting factor):
          
        .. list-table:: Controller modes
           :widths: 75 25
           :header-rows: 1
    
           * - Mode
             - Cost function
           * - 'MPC' - Model-predictive control (MPC)
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right)=\\sum_{k=1}^{N_a} \\gamma^{k-1} r(y_k, u_k)`
           * - 'RQL' - RL/ADP via :math:`N_a-1` roll-outs of :math:`r`
             - :math:`J \\left( y_1, \\{u\}_{1}^{N_a}\\right) =\\sum_{k=1}^{N_a-1} \\gamma^{k-1} r(y_k, u_k) + \\hat Q(y_{N_a}, u_{N_a})` 
           * - 'SQL' - RL/ADP via stacked Q-learning [[1]_]
             - :math:`J \\left( y_1, \\{u\\}_1^{N_a} \\right) =\\frac{1}{N_a} \\sum_{k=1}^{N_a-1} \\hat Q(y_{N_a}, u_{N_a})`               
        
        *Add your specification into the table when customizing the agent*    

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
        Functions that represent the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``x_sys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sys_rhs, sys_out, x_sys`` are used in those controller modes which rely on them
    prob_noise_pow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control   
    is_est_model : : number
        Flag whether to estimate a system model. See :func:`~controllers.ctrl_opt_pred._estimate_model` 
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
        
        See :func:`~controllers.ctrl_opt_pred._estimate_model`. This is just a particular model estimator.
        When customizing, :func:`~controllers.ctrl_opt_pred._estimate_model` may be changed and in turn the parameter ``model_order`` also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers of neurons, in which case ``model_order`` could be substituted for, say, ``Nlayers``, ``Nneurons`` 
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
           * - 'quad-lin'
             - Quadratic-linear
           * - 'quadratic'
             - Quadratic
           * - 'quad-nomix'
             - Quadratic, no mixed terms
           * - 'quad-mix'
             - Quadratic, no mixed terms in input and output, i.e., :math:`w_1 y_1^2 + \\dots w_p y_p^2 + w_{p+1} y_1 u_1 + \\dots w_{\\bullet} u_1^2 + \\dots`, 
               where :math:`w` is the critic's weight vector
       
        *Add your specification into the table when customizing the critic* 
    rcost_struct : : string
        Choice of the running cost structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 'quadratic'
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcost_pars`` should be ``[R1]``
           * - 'biquadratic'
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcost_pars``
               should be ``[R1, R2]``   
        
        *Pass correct running cost parameters in* ``rcost_pars`` *(as a list)*
        
        *When customizing the running cost, add your specification into the table above*
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """    
         
    def __init__(self, dim_input, dim_output, mode='MPC', ctrl_bnds=[], t0=0, sampling_time=0.1, Nactor=1, pred_step_size=0.1,
                 sys_rhs=[], sys_out=[], x_sys=[], prob_noise_pow = 1, is_est_model=0, model_est_stage=1, model_est_period=0.1, buffer_size=20, model_order=3, model_est_checks=0,
                 gamma=1, Ncritic=4, critic_period=0.1, critic_struct='quad-nomix', rcost_struct='quadratic', rcost_pars=[], y_target=[]):
        
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
        
        self.my_model = self.model(A, B, C, D, x0est)
        
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
        self.rcost_struct = rcost_struct
        self.rcost_pars = rcost_pars
        self.y_target = y_target
        
        self.icost_val = 0

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
            
        self.w_critic_prev = np.ones(self.dim_critic)  
        self.w_critic_init = self.w_critic_prev
        
        # self.big_number = 1e4

    class model:
        """
            Class of estimated models
            
            So far, uses just the state-space structure:
                
        .. math::
            \\begin{array}{ll}
    			\\hat x^+ & = A \\hat x + B u \\newline
    			y^+  & = C \\hat x + D u,
            \\end{array}                 
            
        Attributes
        ---------- 
        A, B, C, D : : arrays of proper shape
            State-space model parameters
        x0set : : array
            Initial state estimate
            
        **When introducing your custom model estimator, adjust this class**    
            
        """
        
        def __init__(self, A, B, C, D, x0est):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            self.x0est = x0est
            
        def upd_pars(self, Anew, Bnew, Cnew, Dnew):
            self.A = Anew
            self.B = Bnew
            self.C = Cnew
            self.D = Dnew
            
        def updateIC(self, x0setNew):
            self.x0set = x0setNew

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

        if self.rcost_struct == 'quadratic':
            R1 = self.rcost_pars[0]
            r = chi @ R1 @ chi
        elif self.rcost_struct == 'biquadratic':
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
            if self.is_est_model or self.mode in ['RQL', 'SQL']:
                time_in_est_period = t - self.est_clock
                
                # Estimate model if required
                if (time_in_est_period >= self.model_est_period) and self.is_est_model:
                    # Update model estimator's internal clock
                    self.est_clock = t
                    
                    try:
                        # Using ssid from Githug:AndyLamperski/pyN4SID
                        # Aid, Bid, Cid, Did, _ ,_ = ssid.N4SID(serf.ubuffer.T,  self.ybuffer.T, 
                        #                                       NumRows = self.dim_input + self.model_order,
                        #                                       NumCols = self.buffer_size - (self.dim_input + self.model_order)*2,
                        #                                       NSig = self.model_order,
                        #                                       require_stable=False) 
                        # self.my_model.upd_pars(Aid, Bid, Cid, Did)
                        
                        # Using Github:CPCLAB-UNIPI/SIPPY 
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(self.ybuffer, self.ubuffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.buffer_size/12),
                                                            # SS_p=int(self.buffer_size/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sampling_time)
                        
                        self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # NN_wgts = NN_train(...)
                        
                        # [EXPERIMENTAL] Using MATLAB's system identification toolbox
                        # us_ml = eng.transpose(matlab.double(self.ubuffer.tolist()))
                        # ys_ml = eng.transpose(matlab.double(self.ybuffer.tolist()))
                        
                        # Aml, Bml, Cml, Dml = eng.mySSest_simple(ys_ml, us_ml, dt, model_order, nargout=4)
                        
                        # self.my_model.upd_pars(np.asarray(Aml), np.asarray(Bml), np.asarray(Cml), np.asarray(Dml) )
                        
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
                            x0est,_,_,_ = np.linalg.lstsq(C, y)
                            Yest,_ = dss_sim(A, B, C, D, self.ubuffer, x0est, y)
                            mean_err = np.mean(Yest - self.ybuffer, axis=0)
                            
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
                        # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, y)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.ubuffer, x0est, y)
                        # mean_err = np.mean(Yest - ctrlStat.ybuffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                        # dataRow = []
                        # for k in range(dim_output):
                        #     dataRow.append( mean_err[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                        # print(R+table+Bl)
                        # /DEBUG ===================================================================                    
            
            # Update initial state estimate
            x0est,_,_,_ = np.linalg.lstsq(self.my_model.C, y)
            self.my_model.updateIC(x0est)
     
            if t >= self.model_est_stage:
                    # Drop probing noise
                    self.is_prob_noise = 0 

    def _phi(self, y, u):
        """
        Feature vector of the critic
        
        In Q-learning mode, it uses both ``y`` and ``u``. In value function approximation mode, it should use just ``y``
        
        Customization
        -------------
        
        Adjust this method if you still sitck with a linearly parametrized approximator for Q-function, value function etc.
        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like ``w_critic @ self._phi( y, u )`` 
        within :func:`~controllers.ctrl_opt_pred._critic_cost`
        
        """
        if self.y_target == []:
            chi = np.concatenate([y, u])
        else:
            chi = np.concatenate([y - self.y_target, u])
        
        if self.critic_struct == 'quad-lin':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ), chi ])
        elif self.critic_struct == 'quadratic':
            return np.concatenate([ uptria2vec( np.outer(chi, chi) ) ])   
        elif self.critic_struct == 'quad-nomix':
            return chi * chi    
        elif self.critic_struct == 'quad-mix':
            return np.concatenate([ y**2, np.kron(y, u), u**2 ]) 
    
    def _critic_cost(self, w_critic):
        """
        Cost function of the critic
        
        Currently uses value-iteration-like method  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation 
       
        """
        Jc = 0
        
        for k in range(self.Ncritic-1, 0, -1):
            yPrev = self.ybuffer[k-1, :]
            yNext = self.ybuffer[k, :]
            uPrev = self.ubuffer[k-1, :]
            uNext = self.ubuffer[k, :]
            
            # Temporal difference
            e = w_critic @ self._phi( yPrev, uPrev ) - self.gamma * self.w_critic_prev @ self._phi( yNext, uNext ) - self.rcost(yPrev, uPrev)
            
            Jc += 1/2 * e**2
            
        return Jc
        
        
    def _critic(self):
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.ctrl_opt_pred._critic_cost`

        """        
        
        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        critic_opt_method = 'SLSQP'
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
        
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)
    
        w_critic = minimize(lambda w_critic: self._critic_cost(w_critic), self.w_critic_init, method=critic_opt_method, tol=1e-7, bounds=bnds, options=critic_opt_options).x
        
        # DEBUG ===================================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( w_critic )
        # /DEBUG ==================================================================
        
        return w_critic
    
    def _actor_cost(self, U, y):
        """
        See class documentation
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation

        """
        
        myU = np.reshape(U, [self.Nactor, self.dim_input])
        
        Y = np.zeros([self.Nactor, self.dim_output])
        
        # System output prediction
        if not self.is_est_model:    # Via exogenously passed model
            Y[0, :] = y
            x = self.x_sys
            for k in range(1, self.Nactor):
                # x = get_next_state(x, myU[k-1, :], delta)         TODO
                x = x + self.pred_step_size * self.sys_rhs([], x, myU[k-1, :])  # Euler scheme
                
                Y[k, :] = self.sys_out(x)

        elif self.is_est_model:    # Via estimated model
            myU_upsampled = myU.repeat(int(self.pred_step_size/self.sampling_time), axis=0)
            Yupsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0est, y)
            Y = Yupsampled[::int(self.pred_step_size/self.sampling_time)]
        
        J = 0         
        if self.mode=='MPC':
            for k in range(self.Nactor):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
        elif self.mode=='RQL':     # RL: Q-learning with Ncritic-1 roll-outs of running cost
             for k in range(self.Nactor-1):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
             J += self.w_critic @ self._phi( Y[-1, :], myU[-1, :] )
        elif self.mode=='SQL':     # RL: stacked Q-learning
             for k in range(self.Nactor): 
                Q = self.w_critic @ self._phi( Y[k, :], myU[k, :] )
                
                # With state constraints via indicator function
                # Q = w_critic @ self._phi( Y[k, :], myU[k, :] ) + state_constraint_indicator(Y[k, 0])
                
                # DEBUG ===================================================================
                # =========================================================================
                # R  = '\033[31m'
                # Bl  = '\033[30m'
                # if state_constraint_indicator(Y[k, 0]) > 1:
                #     print(R+str(state_constraint_indicator(Y[k, 0]))+Bl)
                # /DEBUG ==================================================================                 
                
                J += Q 

        return J
    
    def _actor(self, y):
        """
        See class documentation
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~controllers.ctrl_opt_pred._actor_cost` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # For direct implementation of state constraints, this needs `partial` from `functools`
        # See [here](https://stackoverflow.com/questions/27659235/adding-multiple-constraints-to-scipy-minimize-autogenerate-constraint-dictionar)
        # def state_constraint(U, idx):
            
        #     myU = np.reshape(U, [N, self.dim_input])
            
        #     Y = np.zeros([idx, self.dim_output])    
            
        #     # System output prediction
        #     if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
        #         Y[0, :] = y
        #         x = self.x_sys
        #         for k in range(1, idx):
        #             # x = get_next_state(x, myU[k-1, :], delta)
        #             x = x + delta * self.sys_rhs([], x, myU[k-1, :], [])  # Euler scheme
        #             Y[k, :] = self.sys_out(x)            
            
        #     return Y[-1, 1] - 1

        # my_constraints=[]
        # for my_idx in range(1, self.Nactor+1):
        #     my_constraints.append({'type': 'eq', 'fun': lambda U: state_constraint(U, idx=my_idx)})

        # my_constraints = {'type': 'ineq', 'fun': state_constraint}

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        # actor_opt_method = 'SLSQP' # Standard
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
                U = basinhopping(lambda U: self._actor_cost(U, y), myUinit, minimizer_kwargs=minimizer_kwargs, niter = 10).x
            else:
                U = minimize(lambda U: self._actor_cost(U, y), myUinit, method=actor_opt_method, tol=1e-7, bounds=bnds, options=actor_opt_options).x        

        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myUinit
        
        # DEBUG ===================================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # myU = np.reshape(U, [N, self.dim_input])    
        # myU_upsampled = myU.repeat(int(delta/self.sampling_time), axis=0)
        # Yupsampled, _ = dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0est, y)
        # Y = Yupsampled[::int(delta/self.sampling_time)]
        # Yt = np.zeros([N, self.dim_output])
        # Yt[0, :] = y
        # x = self.x_sys
        # for k in range(1, Nactor):
        #     x = x + delta * self.sys_rhs([], x, myU[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sys_out(x)           
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
        # dataRow = []
        # for k in range(dim_output):
        #     dataRow.append( np.mean(Y[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ==================================================================     
        
        return U[:self.dim_input]    # Return first action
                    
    def compute_action(self, t, y):
        """
        Main method. See class documentation
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~controllers.ctrl_opt_pred._actor_cost`, here

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
                    u = self._actor(y)

                elif self.mode=='MPC':
                    u = self._actor(y)
                    
            elif self.mode in ['RQL', 'SQL']:
                # Critic
                timeInCriticPeriod = t - self.critic_clock
                
                # Update data buffers
                self.ubuffer = push_vec(self.ubuffer, self.uCurr)
                self.ybuffer = push_vec(self.ybuffer, y)
                
                if timeInCriticPeriod >= self.critic_period:
                    # Update critic's internal clock
                    self.critic_clock = t
                    
                    self.w_critic = self._critic()
                    self.w_critic_prev = self.w_critic
                    
                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.w_critic_init = self.w_critic_prev
                    
                else:
                    self.w_critic = self.w_critic_prev
                    
                # Actor. Apply control when model estimation phase is over
                if self.is_prob_noise and self.is_est_model:
                    u = self.prob_noise_pow * (rand(self.dim_input) - 0.5)
                elif not self.is_prob_noise and self.is_est_model:
                    u = self._actor(y)
                    
                elif self.mode in ['RQL', 'SQL']:
                    u = self._actor(y) 
            
            self.uCurr = u
            
            return u    
    
        else:
            return self.uCurr
        
class CtrlNominal3WRobot:
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
            # Update internal clock
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

            # DEBUG ===================================================================   
            # ================================LF debugger
            R  = '\033[31m'
            Bl  = '\033[30m'
            headerRow = ['L']  
            dataRow = [self.compute_LF(y)]
            rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
            table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
            print(R+table+Bl)
            # /DEBUG ===================================================================             

            return u    
    
        else:
            return self.uCurr

    def compute_action_vanila(self, y):
        """
        Same as :func:`~ctrl_nominal_3wrobot.compute_action`, but without invoking the internal clock

        """
        
        xNI, eta = self._Cart2NH( y ) 
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = - self.ctrl_gain * z
        u = self._NH2ctrl_Cart(xNI, eta, uNI)
        
        self.uCurr = u
        
        return u

    def compute_LF(self, y):
        
        xNI, eta = self._Cart2NH( y ) 
        theta_star = self._minimizer_theta(xNI, eta)
        
        return self._Fc(xNI, eta, theta_star)
    
class CtrlNominal3WRobotNI:
    """
    Nominal parking controller for NI using disassembled subgradients
    
    """
    
    def __init__(self, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
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
    
    def _zeta(self, xNI):
        """
        Analytic disassembled subgradient, without finding minimizer theta

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
        Stabilizing controller for NI-part

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
        Marginal function for NI

        """
        
        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        
        z = eta - self._kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)
      
    def _Cart2NH(self, coords_Cart): 
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates

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
        Get control for Cartesian NI from NH coordinates       

        """

        uCart = np.zeros(2)
        
        uCart[0] = uNI[1] + 1/2 * uNI[0] * ( xNI[2] + xNI[0] * xNI[1] )
        uCart[1] = uNI[0]
        
        return uCart

    def compute_action(self, t, y):
        """
        
        """
        
        time_in_sample = t - self.ctrl_clock
        
        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t
            
            xNI = self._Cart2NH( y ) 
            kappa_val = self._kappa(xNI)
            uNI = self.ctrl_gain * kappa_val
            u = self._NH2ctrl_Cart(xNI, uNI)
            
            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])           
            
            self.uCurr = u
            
            # DEBUG ===================================================================   
            # ================================LF debugger
            # R  = '\033[31m'
            # Bl  = '\033[30m'
            # headerRow = ['L']  
            # dataRow = [self.compute_LF(y)]
            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
            # print(R+table+Bl)
            # /DEBUG ===================================================================            
            
            return u    
    
        else:
            return self.uCurr

    def compute_action_vanila(self, y):
        """
        Same as :func:`~ctrl_nominal_3wrobot_NI.compute_action`, but without invoking the internal clock

        """
        
        xNI = self._Cart2NH( y ) 
        kappa_val = self._kappa(xNI)
        uNI = self.ctrl_gain * kappa_val
        u = self._NH2ctrl_Cart(xNI, uNI)
        
        self.uCurr = u
        
        return u

    def compute_LF(self, y):
        
        xNI = self._Cart2NH( y ) 
        
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt( np.abs(xNI[2]) )
        
        return xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma**2