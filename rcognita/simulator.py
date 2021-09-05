#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains one single class that simulates controller-system (agent-environment) loops.
The system can be of three types:
- discrete-time deterministic
- continuous-time deterministic or stochastic
- discrete-time stochastic (to model Markov decision processes)

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import scipy as sp

from .utilities import rej_sampling_rvs

class Simulator:
    """
    Class for simulating closed loops (system-controllers)
      
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
        
    closed_loop_rhs : : function
        Right-hand side description of the closed-loop system.
        Say, if you instantiated a concrete system (i.e., as an instance of a subclass of ``system`` class with concrete ``closed_loop_rhs`` method) as ``my_sys``,
        this could be just ``my_sys.closed_loop_rhs``
        
    sys_out : : function
        System output function.
        Same as above, this could be, say, ``my_sys.out``        
        
    is_dyn_ctrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector

    x0, q0, u0 : : vectors
        Initial values of the (open-loop) system state, disturbance and input
        
    t0, t1, dt : : numbers
        Initial, final times and time step size
        
    max_step, first_step, atol, rtol : : numbers
        Parameters for an ODE solver (used if ``sys_type`` is ``diff_eqn``)
        
    See also
    --------

    ``systems`` module    
   
    """    
    
    def __init__(self, sys_type, closed_loop_rhs, sys_out, x0, q0=[], u0=[], t0=0, t1=1, dt=1e-2, max_step=0.5e-2, first_step=1e-6, atol=1e-5, rtol=1e-3,
                 is_disturb=0, is_dyn_ctrl=0):
        self.sys_type = sys_type
        self.closed_loop_rhs = closed_loop_rhs
        self.sys_out = sys_out
        self.dt = dt
        
        # Build full state of the closed-loop
        if is_dyn_ctrl:
            if is_disturb:
                ksi0 = np.concatenate([x0, q0, u0])
            else:
                ksi0 = np.concatenate([x0, u0])
        else:
            if is_disturb:
                ksi0 = np.concatenate([x0, q0])
            else:
                ksi0 = x0
            
        self.ksi = ksi0
            
        self.t = t0
        self.x = x0
        self.dim_state = x0.shape[0]
        self.y = self.sys_out(x0)
        
        if sys_type == "diff_eqn":
            self.ODE_solver = sp.integrate.RK45(closed_loop_rhs, t0, ksi0, t1, max_step = dt/2, first_step=first_step, atol=atol, rtol=rtol) 
            
        # Store these for reset purposes
        self.ksi0 = ksi0
        self.t0 = t0
    
    def sim_step(self):
        """
        Do one simulation step and update current simulation data (time, system state and output) 

        """
        if self.sys_type == "diff_eqn":
            self.ODE_solver.step()
            
            self.t = self.ODE_solver.t
            self.ksi = self.ODE_solver.y 
            
            self.x = self.ksi[0:self.dim_state]
            self.y = self.sys_out(self.x)
            
        elif self.sys_type == "discr_fnc":
            self.t = self.t + self.dt
            self.ksi = self.closed_loop_rhs(self.t, self.ksi)
            
            self.x = self.ksi[0:self.dim_state]
            self.y = self.sys_out(self.x)
            
        elif self.sys_type == "discr_prob":
            self.ksi = rej_sampling_rvs(self.dim_state, self.closed_loop_rhs, 10)
            
            self.t = self.t + self.dt
            
            self.x = self.ksi[0:self.dim_state]
            self.y = self.sys_out(self.x)           
        else:
            raise ValueError('Invalid system description')
            
    def get_sim_step_data(self):
        """
        Collect current simulation data: time, system state and output, and, for completeness, full closed-loop state

        """
        
        t, x, y, ksi = self.t, self.x, self.y, self.ksi
        
        return t, x, y, ksi
    
    def reset(self):
        if self.sys_type == "diff_eqn":
            self.ODE_solver.status = 'running'
            self.ODE_solver.t = self.t0
            self.ODE_solver.y = self.ksi0
        else:
            self.t = self.t0
            self.ksi = self.ksi0