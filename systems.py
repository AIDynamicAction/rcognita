##!/usr/bin/env python3
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
from scipy.optimize import fsolve
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
            
        | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
        | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
        | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`
    
    where:
        
        | :math:`state` : state
        | :math:`action` : input
        | :math:`disturb` : disturbance
        
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
        
   Each concrete system must realize ``System`` and define ``name`` attribute.   
        
    """
    def __init__(self,
                 sys_type,
                 dim_state,
                 dim_input,
                 dim_output,
                 dim_disturb,
                 pars=[],
                 ctrl_bnds=[],
                 is_dyn_ctrl=0,
                 is_disturb=0,
                 pars_disturb=[]):
        
        """
        Parameters
        ----------
        sys_type : : string
            Type of system by description:
                
            | ``diff_eqn`` : differential equation :math:`\mathcal D state = f(state, action, disturb)`
            | ``discr_fnc`` : difference equation :math:`state^+ = f(state, action, disturb)`
            | ``discr_prob`` :  by probability distribution :math:`X^+ \sim P_X(state^+| state, action, disturb)`
        
        where:
            
            | :math:`state` : state
            | :math:`action` : input
            | :math:`disturb` : disturbance
            
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
        self._state = np.zeros(dim_state)
        
        # Current input (a.k.a. action)
        self.action = np.zeros(dim_input)
        
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
            
    def _state_dyn(self, t, state, action, disturb):
        """
        Description of the system internal dynamics.
        Depending on the system type, may be either the right-hand side of the respective differential or difference equation, or a probability distribution.
        As a probability disitribution, ``_state_dyn`` should return a number in :math:`[0,1]`
        
        """
        pass

    def _disturb_dyn(self, t, disturb):
        """
        Dynamical disturbance model depending on the system type:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D disturb = f_q(disturb)`    
        | ``sys_type = "discr_fnc"`` : :math:`disturb^+ = f_q(disturb)`
        | ``sys_type = "discr_prob"`` : :math:`disturb^+ \sim P_Q(disturb^+|disturb)`
        
        """       
        pass

    def _ctrl_dyn(self, t, action, observation):
        """
        Dynamical controller. When ``is_dyn_ctrl=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.
        
        Depending on the system type, can be:
            
        | ``sys_type = "diff_eqn"`` : :math:`\mathcal D action = f_u(action, observation)`    
        | ``sys_type = "discr_fnc"`` : :math:`action^+ = f_u(action, observation)`  
        | ``sys_type = "discr_prob"`` : :math:`action^+ \sim P_U(action^+|action, observation)`        
        
        """
        Daction = np.zeros(self.dim_input)
    
        return Daction 

    def out(self, state, action=[]):
        """
        System output.
        This is commonly associated with signals that are measured in the system.
        Normally, output depends only on state ``state`` since no physical processes transmit input to output instantly.       
        
        See also
        --------
        :func:`~systems.system._state_dyn`
        
        """
        # Trivial case: output identical to state
        observation = state
        return observation
    
    def receive_action(self, action):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~systems.system.out`. 
        Parameters
        ----------
        action : : array of shape ``[dim_input, ]``
            Action
            
        """
        self.action = action
        
    def closed_loop_rhs(self, t, state_full):
        """
        Right-hand side of the closed-loop system description.
        Combines everything into a single vector that corresponds to the right-hand side of the closed-loop system description for further use by simulators.
        
        Attributes
        ----------
        state_full : : vector
            Current closed-loop system state        
        
        """
        rhs_full_state = np.zeros(self._dim_full_state)
        
        state = state_full[0:self.dim_state]
        
        if self.is_disturb:
            disturb = state_full[self.dim_state:]
        else:
            disturb = []
        
        if self.is_dyn_ctrl:
            action = state_full[-self.dim_input:]
            observation = self.out(state)
            rhs_full_state[-self.dim_input:] = self._ctrlDyn(t, action, observation)
        else:
            # Fetch the control action stored in the system
            action = self.action
        
        if self.ctrl_bnds.any():
            for k in range(self.dim_input):
                action[k] = np.clip(action[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])
        
        rhs_full_state[0:self.dim_state] = self._state_dyn(t, state, action, disturb)
        
        if self.is_disturb:
            rhs_full_state[self.dim_state:] = self._disturb_dyn(t, disturb)
        
        # Track system's state
        self._state = state
        
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
        
    | :math:`x_с` : state-coordinate [m]
    | :math:`y_с` : observation-coordinate [m]
    | :math:`\\alpha` : turning angle [rad]
    | :math:`v` : speed [m/s]
    | :math:`\\omega` : revolution speed [rad/s]
    | :math:`F` : pushing force [N]          
    | :math:`M` : steering torque [Nm]
    | :math:`m` : robot mass [kg]
    | :math:`I` : robot moment of inertia around vertical axis [kg m\ :sup:`2`]
    | :math:`disturb` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``is_disturb = 0``
    
    :math:`state = [x_c, y_c, \\alpha, v, \\omega]`
    
    :math:`action = [F, M]`
    
    ``pars`` = :math:`[m, I]`
    
    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
        nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594
    
    """ 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.name = '3wrobot'
        
        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]
    
    def _state_dyn(self, t, state, action, disturb=[]):   
        m, I = self.pars[0], self.pars[1]

        Dstate = np.zeros(self.dim_state)
        Dstate[0] = state[3] * np.cos( state[2] )
        Dstate[1] = state[3] * np.sin( state[2] )
        Dstate[2] = state[4]
        
        if self.is_disturb and (disturb != []):
            Dstate[3] = 1/m * (action[0] + disturb[0])
            Dstate[4] = 1/I * (action[1] + disturb[1])
        else:
            Dstate[3] = 1/m * action[0]
            Dstate[4] = 1/I * action[1] 
            
        return Dstate    
 
    def _disturb_dyn(self, t, disturb):
        """
        Description
        -----------
        
        We use here a 1st-order stochastic linear system of the type
        
        .. math:: \mathrm d Q_t = - \\frac{1}{\\tau_disturb} \\left( Q_t \\mathrm d t + \\sigma_disturb ( \\mathrm d B_t + \\mu_disturb ) \\right) ,
        
        where :math:`B` is the standard Brownian motion, :math:`Q` is the stochastic process whose realization is :math:`disturb`, and
        :math:`\\tau_disturb, \\sigma_disturb, \\mu_disturb` are the time constant, standard deviation and mean, resp.
        
        ``pars_disturb = [sigma_disturb, mu_disturb, tau_disturb]``, with each being an array of shape ``[dim_disturb, ]``
        
        """       
        Ddisturb = np.zeros(self.dim_disturb)
   
        for k in range(0, self.dim_disturb):
            Ddisturb[k] = - self.tau_disturb[k] * ( disturb[k] + self.sigma_disturb[k] * (randn() + self.mu_disturb[k]) )
                
        return Ddisturb   
    
    def out(self, state, action=[]):
        observation = np.zeros(self.dim_output)
        # observation = state[:3] + measNoise # <-- Measure only position and orientation
        observation = state  # <-- Position, force and torque sensors on
        return observation

class Sys3WRobotNI(System):
    """
    System class: 3-wheel robot with static actuators (the NI - non-holonomic integrator).
    
    
    """ 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.name = '3wrobotNI'
        
        if self.is_disturb:
            self.sigma_disturb = self.pars_disturb[0]
            self.mu_disturb = self.pars_disturb[1]
            self.tau_disturb = self.pars_disturb[2]
    
    def _state_dyn(self, t, state, action, disturb=[]):   
        Dstate = np.zeros(self.dim_state)
        
        if self.is_disturb and (disturb != []):
            Dstate[0] = action[0] * np.cos( state[2] ) + disturb[0]
            Dstate[1] = action[0] * np.sin( state[2] ) + disturb[0]
            Dstate[2] = action[1] + disturb[1]
        else:
            Dstate[0] = action[0] * np.cos( state[2] )
            Dstate[1] = action[0] * np.sin( state[2] )
            Dstate[2] = action[1]           
             
        return Dstate    
 
    def _disturb_dyn(self, t, disturb):
        """
        
        
        """       
        Ddisturb = np.zeros(self.dim_disturb)
        
        for k in range(0, self.dim_disturb):
            Ddisturb[k] = - self.tau_disturb[k] * ( disturb[k] + self.sigma_disturb[k] * (randn() + self.mu_disturb[k]) )
                
        return Ddisturb   
    
    def out(self, state, action=[]):
        observation = np.zeros(self.dim_output)
        observation = state
        return observation

class Sys2Tank(System):
    """
    Two-tank system with nonlinearity.
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.name = '2tank'    
    
    def _state_dyn(self, t, state, action, disturb=[]):     
        tau1, tau2, K1, K2, K3 = self.pars

        Dstate = np.zeros(self.dim_state)
        Dstate[0] = 1/(tau1) * ( -state[0] + K1 * action)
        Dstate[1] = 1/(tau2) * ( -state[1] + K2 * state[0] + K3 * state[1]**2)
            
        return Dstate    
 
    def _disturb_dyn(self, t, disturb):   
        Ddisturb = np.zeros(self.dim_disturb)
                
        return Ddisturb   
    
    def out(self, state, action=[]):
        observation = state
        return observation 


class SFC_System(System):
    """
    Economic system .
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.name = 'SFC_economics'
        self.inflation = 0.0
        self.output_growth  = 0.0
    
    def _state_dyn(self, t, state, action, disturb=[]): 

        
        #Initial parametres
        a0= 0.5658628
        a1= 0.83
        a2= 0.04
        k0= 0.1086334242
        k1= 0.35
        k2= 0.025
        k3= 0.1
        k4 = 0.5
        k5 = 0.1
        v0= 0.22382378 
        v1= 0.2
        v2= 0.2
        v3= 0.1
        w0= 0.38973415 #(Model 1)
        w1 = 0.01
        w2 = 0.02
        w3 = 0.02
        f0 = 0.09826265506
        f1 = 0.2
        f2 = 0.6
        g0 = 0.2352693030
        g1 = 0.3
        g2 = 0.04
        g3 = 0
        z0 = 0.3
        z1 = 0.5
        z2 = 0.45
        z3 = 0.033333
        theta = 0.1
        lambda_in = 0.050005
        lambda_0 = 0.159143
        delta = 0.0625
        r0 = 0.67652
        sf = 0.34097798866 
        theta_b = 0.2862767

        #Interest Rates
        #ib = 0.015 
        
        m1b= 0.005 
        m2b = 0.005
        
        ib = action
        dim_action = 1
        ib_1 = ib

        #Initial values
        Y_1 = 100
        C_1 = 60
        I_1= 25
        G_1= 15
        BD_1 = 45
        B_1 = 0
        BP_1 =  0.979955
        BT_1 = 0
        DIV_1 = 20
        DIVe_1 = 13.33  
        DIVh_1 = 6.66
        Vg_1 = 0
        E_1 = 3
        Ee_1 = 2
        Eh_1 = 1
        g_1 = 0.0625
        Hh_1 = 9.54858
        Hb_1 = 2.250225
        K_2 = K_1 = 400
        L_2 = L_1 = 100
        pe_1 = 35
        rl_1= 0.02
        r_1 = 0.02
        rb_1= 0.02
        TB_1= 0.393063 
        TCB_1 = 0.176982075  
        T_1 = 7.47687
        UP_1 = 23.6813
        Vh_1 = 89.54858  
        YHSh_1 = 67.2918
        YDh_1 =  67.2918
        W_1= 67.652
        H_1= 11.798805 
        RF_1= 11.798805
        pb_1= 50
        Ve_1=K_1+pe_1*Ee_1-L_1-pe_1*E_1
        
        CGh_1=YHSh_1-YDh_1
        id_1=ib_1-m2b
        re_1=pb_1*B_1/(Vh_1)-v0-v1*rb_1+v2*id_1

        #from equation 15 
        ree_1=(pe_1*Ee_1/(pe_1*Ee_1+K_1)-f0-f2*(UP_1/K_2))/f1
        
        initial_conditions=[G_1,Y_1,C_1,I_1,B_1, YDh_1,W_1,T_1,CGh_1, YHSh_1,Vg_1,
                            Eh_1,Vh_1,re_1,pe_1,BD_1,K_1,Ee_1, ree_1, L_1, UP_1, E_1, Ve_1, BT_1, RF_1]

        G_1, Y_1, C_1, I_1, B_1, YDh_1, W_1, T_1, CGh_1, YHSh_1,\
        Vg_1, Eh_1, Vh_1, re_1, pe_1, BD_1, K_1, Ee_1, ree_1, L_1,\
        UP_1, E_1, Ve_1, BT_1, RF_1, L_2, K_2 = state

        Ve_1=K_1+pe_1*Ee_1-L_1-pe_1*E_1
        Vb_1=K_1-Vh_1-Ve_1-Vg_1
        CGh_1=YHSh_1-YDh_1
        id_1=ib_1-m2b
        re_1=pb_1*B_1/(Vh_1)-v0-v1*rb_1+v2*id_1

    #from equation 15 
        ree_1=(pe_1*Ee_1/(pe_1*Ee_1+K_1)-f0-f2*(UP_1/K_2))/f1
        DIV=(1-sf)*(Y_1-W_1-rl_1*L_2)
        DIVe=DIV*(Ee_1/E_1)
        DIVh=DIV-DIVe
        #Hh=lambda_0*C :we use this fact:

        #Control ib
        rl=ib+m1b
        ideposit=ib-m2b

        r=rl
        rb=r
        pb=1/rb

        TB=theta_b*(rl*L_1+r*BT_1-ideposit*BD_1-ib*RF_1)
        BP=(1-theta_b)*(rl*L_1+r*BT_1-ideposit*BD_1-ib*RF_1)
        TCB=ib*RF_1
        Vb=Vb_1+BP

        #solve economic system:


        def economic_system(x):

            equations=[x[1] -x[2]-x[3]-x[0], #1
                   x[5] -x[6]-ideposit*BD_1 - B_1-DIVh + x[7], #2
                    x[9]-x[5]- x[8],#3
                       x[7]-theta*(x[6]+ideposit*BD_1+B_1+DIVh),#4
                   x[2]-a0-a1*x[9]-a2*Vh_1,#5
                   pb*x[4]-(x[12]*(v0+v1*rb-v2*ideposit-v3*x[13])),#6
                   x[14]*x[11]-x[12]*(w0-w1*rb-w2*ideposit+w3*x[13]),#7

                   x[15]-BD_1-x[5]+x[2]+pb*(x[4]-B_1)+x[14]*(x[11]-Eh_1)+(lambda_0*x[2]-Hh_1),#9

                   x[8]-B_1*(pb-pb_1)-Eh_1*(x[14]-pe_1), #10
                   x[12]-x[15]-pb*x[4]-x[14]*x[11]-lambda_0*x[2], #11
                   #save K_2 can save K_1 instead 
                   x[3]-(k0+k1*(UP_1/K_2)+k2*((x[1]-Y_1)/Y_1)-k3*(L_1/K_1)-k4*rl-k5*x[18])*K_1,#12,#13 
                   x[16]-K_1-x[3]+delta*K_1, #14
                   x[13]*x[17]-(x[16]+x[14]*x[17])*(f0+f1*x[18]+f2*(x[20]/K_1)), #15
                   x[19]-x[16]*(g0+g1*(UP_1/K_1)+g2*re_1-g3*rl), #16

                   x[3]+x[14]*(x[17]-Ee_1)-x[20]-x[14]*(x[21]-E_1)-(x[19]-L_1),#17

                   x[20]-x[1]+x[6]+rl*L_1+DIVh,#18

                   x[6]-r0*x[1], #19

                   x[13]-((x[14]-pe_1)/pe_1)-DIV/(pe_1*E_1),#20

                   x[11]+x[17]-x[21],#24

                   x[22]-x[16]-x[14]*x[17]+x[19]+x[14]*x[21],#26

                   x[23]-BT_1-x[0]-r*BT_1-B_1+x[7]+TB+TCB+pb*(x[4]-B_1),#27

                   x[10]+x[23]+pb*x[4],#29

                   x[24]-RF_1-(lambda_in*x[15]-Hb_1)-(x[19]-L_1)-(x[23]-BT_1)+BP+(x[15]-BD_1),#32

                   x[24]-lambda_in*x[15]-lambda_0*x[2], #36  and H=RF, Hh=lambda_0*C, Hb=lambda_in*BD

                   x[10]+x[22]+x[12]+Vb-x[16]] #last accounting of wealth and capital  
            return equations

        roots = fsolve(economic_system, initial_conditions)
        #transition
        G, Y, C, I, B, YDh, W, T, CGh, YHSh, Vg,\
        Eh, Vh, re, pe, BD, K, Ee, ree, L, UP, E, Ve, BT, RF=roots


        Dstate = [G, Y, C, I, B, YDh, W, T, CGh, YHSh, Vg,
        Eh, Vh, re, pe, BD, K, Ee, ree, L, UP, E, Ve, BT, RF,
        L_1, K_1]
        #update inflation
        self.inflation = (pe - pe_1)/pe_1
        self.output_growth = (Y-Y_1)/Y_1

        #Dstate - state
        return Dstate    
 
    def _disturb_dyn(self, t, disturb):   
        pass
    
    def out(self, state, action=[]):
        observation = state

        Y_output = state[1]

        Kapital = state[16]
        Labor = state[19]
        Investment = state[3]
        Consumption = state[2]

        inflation =  self.inflation
        output_growth  = self.output_growth

        observation = [Y_output, inflation]
        
        return observation  