#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains a generic interface for systems (environments) as well as concrete systems as realizations of the former

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
from numpy.random import randn

class System:
    """
    Interface class of dynamical systems a.k.a. environments.
    Concrete systems should be built upon this class.
    To design a concrete system: inherit this class, override:
        | :func:`~systems.system._state_dyn` :
        | right-hand side of system description (required)
        | :func:`~systems.system._disturb_dyn` :
        | right-hand side of disturbance model (if necessary)
        | :func:`~systems.system._ctrl_dyn` :
        | right-hand side of controller dynamical model (if necessary)
        | :func:`~systems.system.out` :
        | system out (if not overridden, output is identical to state)
      
    Attributes
    ----------
    sys_type : : string
        Type of system by description:
            
        | ``diff_eqn`` : differential equation :math:`\mathcal D x = f(x, u, q)`
        | ``discr_fnc`` : difference equation :math:`x^+ = f(x, u, q)`
        | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(x^+| x, u, q)`
    
    where:
        
        | :math:`x` : state
        | :math:`u` : input
        | :math:`q` : disturbance
        
    The time variable ``t`` is commonly used by ODE solvers, and you shouldn't have it explicitly referenced in the definition, unless your system is non-autonomous.
    For the latter case, however, you already have the input and disturbance at your disposal.
    
    Parameters of the system are contained in ``pars`` attribute.
    
    dim_state, dim_input, dim_output, dim_disturb : : integer
        System dimensions 
    pars : : list
        List of fixed parameters of the system
    ctrl_bnds : : array of shape ``[dim_input, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    is_dyn_ctrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    is_disturb : : 0 or 1
        If 0, no disturbance is fed into the system
    pars_disturb : : list
        Parameters of the disturbance model
        
    """
    def __init__(self, sys_type, dim_state, dim_input, dim_output, dim_disturb, pars=[], ctrl_bnds=[], is_dyn_ctrl=0, is_disturb=0, pars_disturb=[]):
        self.sys_type = sys_type
        
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_disturb = dim_disturb   
        self.pars = pars
        self.ctrl_bnds = ctrl_bnds
        self.is_dyn_ctrl = is_dyn_ctrl
        self.is_disturb = is_disturb
        self.pars_disturb = pars_disturb
        
        # Track system's state
        self._x = np.zeros(dim_state)
        
        # Current input (a.k.a. action)
        self.u = np.zeros(dim_input)
        
        if is_dyn_ctrl:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb + self.dim_input
            else:
                self._dim_full_state = self.dim_state
        else:
            if is_disturb:
                self._dim_full_state = self.dim_state + self.dim_disturb
            else:
                self._dim_full_state = self.dim_state
            
    def _state_dyn(self, t, x, u, q):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, ``_state_dyn`` should return a number in :math:`[0,1]`
        
        """
        pass

    def _disturb_dyn(self, t, q):
        """
        Dynamical disturbance model depending on the system type:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D q = f_q(q)`    
        | ``sys_type = "discr_fnc"`` : :math:`q^+ = f_q(q)`
        | ``sys_type = "discr_prob"`` : :math:`q^+ \sim P_Q(q^+|q)`
        
        """       
        pass

    def _ctrl_dyn(self, t, u, y):
        """
        Dynamical controller. When ``is_dyn_ctrl=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.
        
        Depending on the system type, can be:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D u = f_u(u, y)`    
        | ``sys_type = "discr_fnc"`` : :math:`u^+ = f_u(u, y)`  
        | ``sys_type = "discr_prob"`` : :math:`u^+ \sim P_U(u^+|u, y)`        
        
        """
        Du = np.zeros(self.dim_input)
    
        return Du 

    def out(self, x, u=[]):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``x`` since no physical processes transmit input to output instantly.       
        
        See also
        --------
        :func:`~systems.system._state_dyn`
        
        """
        # Trivial case: output identical to state
        y = x
        return y
    
    def receive_action(self, u):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~systems.system.out`. 

        Parameters
        ----------
        u : : array of shape ``[dim_input, ]``
            Action
            
        """
        self.u = u
        
    def closed_loop_rhs(self, t, ksi):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.
        
        Attributes
        ----------
        ksi : : vector
            Current closed-loop system state        
        
        """
        rhs_full_state = np.zeros(self._dim_full_state)
        
        x = ksi[0:self.dim_state]
        
        if self.is_disturb:
            q = ksi[self.dim_state:]
        else:
            q = []
        
        if self.is_dyn_ctrl:
            u = ksi[-self.dim_input:]
            y = self.out(x)
            rhs_full_state[-self.dim_input:] = self._ctrlDyn(t, u, y)
        else:
            # Fetch the control action stored in the system
            u = self.u
        
        if self.ctrl_bnds.any():
            for k in range(self.dim_input):
                u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])
        
        rhs_full_state[0:self.dim_state] = self._state_dyn(t, x, u, q)
        
        if self.is_disturb:
            rhs_full_state[self.dim_state:] = self._disturb_dyn(t, q)
        
        # Track system's state
        self._x = x
        
        return rhs_full_state    
    
class Sys3WRobot(System):
    """
    System class: 3-wheel robot with dynamical actuators.
    
    Description
    -----------
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
    def _state_dyn(self, t, x, u, q=[]):   
        m, I = self.pars[0], self.pars[1]

        Dx = np.zeros(self.dim_state)
        Dx[0] = x[3] * np.cos( x[2] )
        Dx[1] = x[3] * np.sin( x[2] )
        Dx[2] = x[4]
        
        if self.is_disturb and (q != []):
            Dx[3] = 1/m * (u[0] + q[0])
            Dx[4] = 1/I * (u[1] + q[1])
        else:
            Dx[3] = 1/m * u[0]
            Dx[4] = 1/I * u[1] 
            
        return Dx    
 
    def _disturb_dyn(self, t, q):
        """
        Description
        -----------
        
        We use here a 1st-order stochastic linear system of the type
        
        .. math:: \mathrm d Q_t = - \\frac{1}{\\tau_q} \\left( Q_t \\mathrm d t + \\sigma_q ( \\mathrm d B_t + \\mu_q ) \\right) ,
        
        where :math:`B` is the standard Brownian motion, :math:`Q` is the stochastic process whose realization is :math:`q`, and
        :math:`\\tau_q, \\sigma_q, \\mu_q` are the time constant, standard deviation and mean, resp.
        
        ``pars_disturb = [sigma_q, mu_q, tau_q]``, with each being an array of shape ``[dim_disturb, ]``
        
        """       
        Dq = np.zeros(self.dim_disturb)
        
        if self.is_disturb:
            sigma_q = self.pars_disturb[0]
            mu_q = self.pars_disturb[1]
            tau_q = self.pars_disturb[2]
            
            for k in range(0, self.dim_disturb):
                Dq[k] = - tau_q[k] * ( q[k] + sigma_q[k] * (randn() + mu_q[k]) )
                
        return Dq   
    
    def out(self, x, u=[]):
        y = np.zeros(self.dim_output)
        # y = x[:3] + measNoise # <-- Measure only position and orientation
        y = x  # <-- Position, force and torque sensors on
        return y

class Sys3WRobotNI(System):
    """
    System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).
    
    
    """        
    def _state_dyn(self, t, x, u, q=[]):   
        Dx = np.zeros(self.dim_state)
        Dx[0] = u[0] * np.cos( x[2] )
        Dx[1] = u[0] * np.sin( x[2] )
        Dx[2] = u[1]
             
        return Dx    
 
    def _disturb_dyn(self, t, q):
        """
        
        
        """       
        Dq = np.zeros(self.dim_disturb)
        
        if self.is_disturb:
            sigma_q = self.pars_disturb[0]
            mu_q = self.pars_disturb[1]
            tau_q = self.pars_disturb[2]
            
            for k in range(0, self.dim_disturb):
                Dq[k] = - tau_q[k] * ( q[k] + sigma_q[k] * (randn() + mu_q[k]) )
                
        return Dq   
    
    def out(self, x, u=[]):
        y = np.zeros(self.dim_output)
        y = x
        return y

class Sys2Tank(System):
    """
    Two-tank system with nonlinearity
    """
    def _state_dyn(self, t, x, u, q=[]):     
        tau1, tau2, K1, K2, K3 = self.pars

        Dx = np.zeros(self.dim_state)
        Dx[0] = 1/(tau1) * ( -x[0] + K1 * u)
        Dx[1] = 1/(tau2) * ( -x[1] + K2 * x[0] + K3 * x[1]**2)
            
        return Dx    
 
    def _disturb_dyn(self, t, q):   
        Dq = np.zeros(self.dim_disturb)
                
        return Dq   
    
    def out(self, x, u=[]):
        y = x
        return y   