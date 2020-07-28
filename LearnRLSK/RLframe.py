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

# LearnRLSK
from LearnRLSK.utilities import *

class system:  
    """
    Class of continuous-time dynamical systems with exogenous input and dynamical disturbance for use with ODE solvers.
    In RL, this is considered the *environment*.
    Normally, you should pass ``closedLoop``, which represents the right-hand side, to your solver.
    
    Attributes
    ----------
    dimState, dimInput, dimOutput, dimDisturb : : integer
        System dimensions 
    pars : : list
        List of fixed parameters of the system
    ctrlBnds : : array of shape ``[dimInput, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    isDynCtrl : : 0 or 1
        If 1, the controller (a.k.a. agent) is considered as a part of the full state vector
    isDisturb : : 0 or 1
        If 0, no disturbance is fed into the system
    parsDisturb : : list
        Parameters of the disturbance model
        
    Customization
    -------------        
       
    Change specification of ``stateDyn, out, disturbDyn``.
    Set up dimensions properly and use the parameters ``pars`` and ``parsDisturb`` in accordance with your system specification
    
    """  
    def __init__(self, dimState, dimInput, dimOutput, dimDisturb, pars=[], ctrlBnds=[], isDynCtrl=0, isDisturb=0, parsDisturb=[]):
        self.dimState = dimState
        self.dimInput = dimInput
        self.dimOutput = dimOutput
        self.dimDisturb = dimDisturb   
        self.pars = pars
        self.ctrlBnds = ctrlBnds
        self.isDynCtrl = isDynCtrl
        self.isDisturb = isDisturb
        self.parsDisturb = parsDisturb
        
        # Track system's state
        self._x = np.zeros(dimState)
        
        # Current input (a.k.a. action)
        self.u = np.zeros(dimInput)
        
        if isDynCtrl:
            self._dimFullState = self.dimState + self.dimDisturb + self.dimInput
        else:
            self._dimFullState = self.dimState + self.dimDisturb
        
    def _stateDyn(self, t, x, u, q):
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
        | :math:`q` : actuator disturbance (see :func:`~RLframe.system.disturbDyn`). Is zero if ``isDisturb = 0``
        
        :math:`x = [x_c, y_c, \\alpha, v, \\omega]`
        
        :math:`u = [F, M]`
        
        ``pars`` = :math:`[m, I]`
        
        References
        ----------
        .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
            nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594
        
        """       
        m, I = self.pars[0], self.pars[1]

        Dx = np.zeros(self.dimState)
        Dx[0] = x[3] * np.cos( x[2] )
        Dx[1] = x[3] * np.sin( x[2] )
        Dx[2] = x[4]
        
        if self.isDisturb:
            Dx[3] = 1/m * (u[0] + q[0])
            Dx[4] = 1/I * (u[1] + q[1])
        else:
            Dx[3] = 1/m * u[0]
            Dx[4] = 1/I * u[1]
            
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
        
        ``parsDisturb = [sigma_q, mu_q, tau_q]``, with each being an array of shape ``[dimDisturb, ]``
        
        """       
        
        Dq = np.zeros(self.dimDisturb)
        
        if self.isDisturb:
            sigma_q = self.parsDisturb[0]
            mu_q = self.parsDisturb[1]
            tau_q = self.parsDisturb[2]
            
            for k in range(0, self.dimDisturb):
                Dq[k] = - tau_q[k] * ( q[k] + sigma_q[k] * (randn() + mu_q[k]) )
                
        return Dq   
 
    def _ctrlDyn(t, u, y):
        """
        Dynamical controller. When ``isDynCtrl=0``, the controller is considered static, which is to say that the control actions are
        computed immediately from the system's output.
        In case of a dynamical controller, the system's state vector effectively gets extended.
        Dynamical controllers have some advantages compared to the static ones.
        
        Controller description
        ---------------------- 
        **Provide your specification of a dynamical controller here**
        
        Currently, left for future implementation    
        
        """
        
        Du = np.zeros(self.dimInput)
    
        return Du   
    
    def out(self, x, u=[]):
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
        :func:`~RLframe.system._stateDyn`
        
        """
        
        y = np.zeros(self.dimOutput)
        # y = x[:3] + measNoise # <-- Measure only position and orientation
        y = x  # <-- Position, force and torque sensors on
        return y

    def receiveAction(self, u):
        """
        Receive exogeneous control action to be fed into the system.
        This action is commonly computed by your controller (agent) using the system output :func:`~RLframe.system.sysOut` 

        Parameters
        ----------
        u : : array of shape ``[dimInput, ]``
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
                x = ksi[0:sys.dimState]
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
                x = ksi[0:sys.dimState]
                y = sys.out(x)
                u = myController(y)
                sys.receiveAction(u)
        
        """
        
        # DEBUG ===================================================================
        # print('INTERNAL t = {time:2.3f}'.format(time=t))
        # /DEBUG ==================================================================
        
        DfullState = np.zeros(self._dimFullState)
        
        x = ksi[0:self.dimState]
        q = ksi[self.dimState:]
        
        if self.isDynCtrl:
            u = ksi[-self.dimInput:]
            DfullState[-self.dimInput:] = self._ctrlDyn(t, u, y)
        else:
            # Fetch the control action stored in the system
            u = self.u
        
        if self.ctrlBnds.any():
            for k in range(self.dimInput):
                u[k] = np.clip(u[k], self.ctrlBnds[k, 0], self.ctrlBnds[k, 1])
        
        DfullState[0:self.dimState] = self._stateDyn(t, x, u, q)
        
        if self.isDisturb:
            DfullState[self.dimState:] = self._disturbDyn(t, q)
        
        # Track system's state
        self._x = x
        
        return DfullState           
  
class controller:  
    """
    Optimal controller (a.k.a. agent) class.
        
    Attributes
    ----------
    dimInput, dimOutput : : integer
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

    ctrlBnds : : array of shape ``[dimInput, 2]``
        Box control constraints.
        First element in each row is the lower bound, the second - the upper bound.
        If empty, control is unconstrained (default)
    t0 : : number
        Initial value of the controller's internal clock
    samplTime : : number
        Controller's sampling time (in seconds)
    Nactor : : natural number
        Size of prediction horizon :math:`N_a` 
    predStepSize : : number
        Prediction step size in :math:`J` as defined above (in seconds). Should be a multiple of ``samplTime``. Commonly, equals it, but here left adjustable for
        convenience. Larger prediction step size leads to longer factual horizon
    sysRHS, sysOut : : functions        
        Functions that represents the right-hand side, resp., the output of the exogenously passed model.
        The latter could be, for instance, the true model of the system.
        In turn, ``xSys`` represents the (true) current state of the system and should be updated accordingly.
        Parameters ``sysRHS, sysOut, xSys`` are used in controller modes which rely on them.
    probNoisePow : : number
        Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control      
    modEstPhase : : number
        Initial phase to fill the estimator's buffer before applying optimal control (in seconds)      
    modEstPeriod : : number
        Time between model estimate updates (in seconds)
    bufferSize : : natural number
        Size of the buffer to store data
    modelOrder : : natural number
        Order of the state-space estimation model
        
        .. math::
            \\begin{array}{ll}
                \\hat x^+ & = A \\hat x + B u \\newline
                y^+  & = C \\hat x + D u,
            \\end{array}             
        
        **See** :func:`~RLframe.controller._estimateModel` . **This is just a particular model estimator.
        When customizing,** :func:`~RLframe.controller._estimateModel`
        **may be changed and in turn the parameter** ``modelOrder`` **also. For instance, you might want to use an artifial
        neural net and specify its layers and numbers
        of neurons, in which case** ``modelOrder`` **could be substituted for, say,** ``Nlayers``, ``Nneurons`` 
    modEstChecks : : natural number
        Estimated model parameters can be stored in stacks and the best among the ``modEstChecks`` last ones is picked.
        May improve the prediction quality somewhat
    gamma : : number in (0, 1]
        Discounting factor.
        Characterizes fading of running costs along horizon
    Ncritic : : natural number
        Critic stack size :math:`N_c`. The critic optimizes the temporal error which is a measure of critic's ability to capture the
        optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer
    criticPeriod : : number
        The same meaning as ``modEstPeriod`` 
    criticStruct : : natural number
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
    rcostStruct : : natural number
        Choice of the running cost structure.
        
        Currently available:
           
        .. list-table:: Critic structures
           :widths: 10 90
           :header-rows: 1
    
           * - Mode
             - Structure
           * - 1
             - Quadratic :math:`\\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcostPars`` should be ``[R1]``
           * - 2
             - 4th order :math:`\\left( \\chi^\\top \\right)^2 R_2 \\left( \\chi \\right)^2 + \\chi^\\top R_1 \\chi`, where :math:`\\chi = [y, u]`, ``rcostPars``
               should be ``[R1, R2]``           
        
        **Pass correct running cost parameters in** ``rcostPars`` **(as a list)**
        
        **When customizing the running cost, add your specification into the table above**

    Examples
    ----------
    
    Assuming ``sys`` is a ``system``-object, ``t0, t1`` - start and stop times, and ``ksi0`` - a properly defined initial condition:
    
    >>> import scipy as sp
    >>> simulator = sp.integrate.RK45(sys.closedLoop, t0, ksi0, t1)
    >>> agent = controller(sys.dimInput, sys.dimOutput)

    >>> while t < t1:
            simulator.step()
            t = simulator.t
            ksi = simulator.y
            x = ksi[0:sys.dimState]
            y = sys.out(x)
            u = agent.computeAction(t, y)
            sys.receiveAction(u)
            agent.update_icost(y, u)
        
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155        
        
    """    
         
    def __init__(self, dimInput, dimOutput, mode=1, ctrlBnds=[], t0=0, samplTime=0.1, Nactor=1, predStepSize=0.1,
                 sysRHS=[], sysOut=[], xSys=[], probNoisePow = 1, modEstPhase=1, modEstPeriod=0.1, bufferSize=20, modelOrder=3, modEstChecks=0,
                 gamma=1, Ncritic=4, criticPeriod=0.1, criticStruct=1, rcostStruct=1, rcostPars=[]):
        
        self.dimInput = dimInput
        self.dimOutput = dimOutput
        
        self.mode = mode

        self.ctrlClock = t0
        self.samplTime = samplTime
        
        # Controller: common
        self.Nactor = Nactor 
        self.predStepSize = predStepSize
        
        self.uMin = np.array( ctrlBnds[:,0] )
        self.uMax = np.array( ctrlBnds[:,1] )
        self.Umin = repMat(self.uMin, 1, Nactor)
        self.Umax = repMat(self.uMax, 1, Nactor) 
        
        self.uCurr = self.uMin/10
        
        self.Uinit = repMat( self.uMin/10 , 1, self.Nactor)
        
        self.ubuffer = np.zeros([ bufferSize, dimInput] )
        self.ybuffer = np.zeros( [bufferSize, dimOutput] )        
        
        # Exogeneous model's things
        self.sysRHS = sysRHS
        self.sysOut = sysOut
        self.xSys = xSys
        
        # Model estimator's things
        self.estClock = t0
        self.isProbNoise = 1
        self.probNoisePow = probNoisePow
        self.modEstPhase = modEstPhase
        self.modEstPeriod = modEstPeriod
        self.bufferSize = bufferSize
        self.modelOrder = modelOrder
        self.modEstChecks = modEstChecks
        
        A = np.zeros( [self.modelOrder, self.modelOrder] )
        B = np.zeros( [self.modelOrder, self.dimInput] )
        C = np.zeros( [self.dimOutput, self.modelOrder] )
        D = np.zeros( [self.dimOutput, self.dimInput] )
        x0est = np.zeros( self.modelOrder )
        
        self.myModel = model(A, B, C, D, x0est)
        
        self.modelStack = []
        for k in range(self.modEstChecks):
            self.modelStack.append(self.myModel)        
        
        # RL elements
        self.criticClock = t0
        self.gamma = gamma
        self.Ncritic = Ncritic
        self.Ncritic = np.min([self.Ncritic, self.bufferSize-1]) # Clip critic buffer size
        self.criticPeriod = criticPeriod
        self.criticStruct = criticStruct
        self.rcostStruct = rcostStruct
        self.rcostPars = rcostPars
        
        self.icostVal = 0

        if self.criticStruct == 1:
            self.dimCrit = ( ( self.dimOutput + self.dimInput ) + 1 ) * ( self.dimOutput + self.dimInput )/2 + (self.dimOutput + self.dimInput)  
            self.Wmin = -1e3*np.ones(self.dimCrit) 
            self.Wmax = 1e3*np.ones(self.dimCrit) 
        elif self.criticStruct == 2:
            self.dimCrit = ( ( self.dimOutput + self.dimInput ) + 1 ) * ( self.dimOutput + self.dimInput )/2
            self.Wmin = np.zeros(self.dimCrit) 
            self.Wmax = 1e3*np.ones(self.dimCrit)    
        elif self.criticStruct == 3:
            self.dimCrit = self.dimOutput + self.dimInput
            self.Wmin = np.zeros(self.dimCrit) 
            self.Wmax = 1e3*np.ones(self.dimCrit)    
        elif self.criticStruct == 4:
            self.dimCrit = self.dimOutput + self.dimOutput * self.dimInput + self.dimInput
            self.Wmin = -1e3*np.ones(self.dimCrit) 
            self.Wmax = 1e3*np.ones(self.dimCrit)
            
        self.Wprev = np.ones(self.dimCrit) 
        
        self.Winit = self.Wprev


    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        Only internal clock and current actions are reset.
        All the learned parameters are retained
        
        """
        self.ctrlClock = t0
        self.uCurr = self.uMin/10
    
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
            ySqn = np.zeros( [ uSqn.shape[0], C.shape[0] ] )
            xSqn = np.zeros( [ uSqn.shape[0], A.shape[0] ] )
            x = x0
            ySqn[0, :] = y0
            xSqn[0, :] = x0 
            for k in range( 1, uSqn.shape[0] ):
                x = A @ x + B @ uSqn[k-1, :]
                xSqn[k, :] = x
                ySqn[k, :] = C @ x + D @ uSqn[k-1, :]
                
            return ySqn, xSqn    
  
    def rcost(self, y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)
        
        See class documentation
        """
        chi = np.concatenate([y, u])
        
        r = 0

        if self.rcostStruct == 1:
            R1 = self.rcostPars[0]
            r = chi @ R1 @ chi
        elif self.rcostStruct == 2:
            R1 = self.rcostPars[0]
            R2 = self.rcostPars[1]
            
            r = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
        
        return r
        
    def update_icost(self, y, u):
        """
        Sample-to-sample integrated running cost. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``icost`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)
        
        """
        self.icostVal += self.rcost(y, u)*self.samplTime
    
    def _estimateModel(self, t, y):
        """
        Estimate model parameters by accumulating data buffers ``ubuffer`` and ``ybuffer``
        
        """
        
        timeInSample = t - self.ctrlClock
        
        if timeInSample >= self.samplTime: # New sample
            # Update buffers when using RL or requiring estimated model
            if self.mode in (2,3,4,5,6):
                timeInEstPeriod = t - self.estClock
                
                # Estimate model if required by ctrlStatMode
                if (timeInEstPeriod >= modEstPeriod) and (self.mode in (2,4,6)):
                    # Update model estimator's internal clock
                    self.estClock = t
                    
                    try:
                        # Using ssid from Githug:AndyLamperski/pyN4SID
                        # Aid, Bid, Cid, Did, _ ,_ = ssid.N4SID(serf.ubuffer.T,  self.ybuffer.T, 
                        #                                       NumRows = self.dimInput + self.modelOrder,
                        #                                       NumCols = self.bufferSize - (self.dimInput + self.modelOrder)*2,
                        #                                       NSig = self.modelOrder,
                        #                                       require_stable=False) 
                        # self.myModel.updatePars(Aid, Bid, Cid, Did)
                        
                        # Using Github:CPCLAB-UNIPI/SIPPY 
                        # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                        SSest = sippy.system_identification(self.ybuffer, self.ubuffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.modelOrder,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            # SS_f=int(self.bufferSize/12),
                                                            # SS_p=int(self.bufferSize/10),
                                                            SS_PK_B_reval=False,
                                                            tsample=self.samplTime)
                        
                        self.myModel.updatePars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # [EXPERIMENTAL] Using MATLAB's system identification toolbox
                        # us_ml = eng.transpose(matlab.double(self.ubuffer.tolist()))
                        # ys_ml = eng.transpose(matlab.double(self.ybuffer.tolist()))
                        
                        # Aml, Bml, Cml, Dml = eng.mySSest_simple(ys_ml, us_ml, dt, modelOrder, nargout=4)
                        
                        # self.myModel.updatePars(np.asarray(Aml), np.asarray(Bml), np.asarray(Cml), np.asarray(Dml) )
                        
                    except:
                        print('Model estimation problem')
                        self.myModel.updatePars(np.zeros( [self.modelOrder, self.modelOrder] ),
                                                np.zeros( [self.modelOrder, self.dimInput] ),
                                                np.zeros( [self.dimOutput, self.modelOrder] ),
                                                np.zeros( [self.dimOutput, self.dimInput] ) )
                    
                    # Model checks
                    if self.modEstChecks > 0:
                        # Update estimated model parameter stacks
                        self.modelStack.pop(0)
                        self.modelStack.append(self.model)

                        # Perform check of stack of models and pick the best
                        totAbsErrCurr = 1e8
                        for k in range(self.modEstChecks):
                            A, B, C, D = self.modelStack[k].A, self.modelStack[k].B, self.modelStack[k].C, self.modelStack[k].D
                            x0est,_,_,_ = np.linalg.lstsq(C, y)
                            Yest,_ = self._dssSim(A, B, C, D, self.ubuffer, x0est, y)
                            meanErr = np.mean(Yest - self.ybuffer, axis=0)
                            
                            # DEBUG ===================================================================
                            # ================================Interm output of model prediction quality
                            # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                            # dataRow = []
                            # for k in range(dimOutput):
                            #     dataRow.append( meanErr[k] )
                            # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                            # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                            # print( table )
                            # /DEBUG ===================================================================
                            
                            totAbsErr = np.sum( np.abs( meanErr ) )
                            if totAbsErr <= totAbsErrCurr:
                                totAbsErrCurr = totAbsErr
                                self.myModel.updatePars(SSest.A, SSest.B, SSest.C, SSest.D)
                        
                        # DEBUG ===================================================================
                        # ==========================================Print quality of the best model
                        # R  = '\033[31m'
                        # Bl  = '\033[30m'
                        # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, y)
                        # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.ubuffer, x0est, y)
                        # meanErr = np.mean(Yest - ctrlStat.ybuffer, axis=0)
                        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                        # dataRow = []
                        # for k in range(dimOutput):
                        #     dataRow.append( meanErr[k] )
                        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                        # print(R+table+Bl)
                        # /DEBUG ===================================================================                       
            
            # Update initial state estimate
            x0est,_,_,_ = np.linalg.lstsq(self.myModel.C, y)
            self.myModel.updateIC(x0est)
     
            if t >= self.modEstPhase:
                    # Drop probing noise
                    self.isProbNoise = 0 

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
        
        if self.criticStruct == 1:
            return np.concatenate([ uptria2vec( np.kron(chi, chi) ), chi ])
        elif self.criticStruct == 2:
            return np.concatenate([ uptria2vec( np.kron(chi, chi) ) ])   
        elif self.criticStruct == 3:
            return chi * chi    
        elif self.criticStruct == 4:
            return np.concatenate([ y**2, np.kron(y, u), u**2 ]) 
    
    def _criticCost(self, W, U, Y):
        """
        Cost function of the critic
        
        Currently uses value-iteration-like method  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation 
       
        """        
        Jc = 0
        
        for k in range(self.dimCrit, 0, -1):
            yPrev = Y[k-1, :]
            yNext = Y[k, :]
            uPrev = U[k-1, :]
            uNext = U[k, :]
            
            # Temporal difference
            e = W @ self._Phi( yPrev, uPrev ) - self.gamma * self.Wprev @ self._Phi( yNext, uNext ) - self.rcost(yPrev, uPrev)
            
            Jc += 1/2 * e**2
            
        return Jc
        
        
    def _critic(self, Wprev, Winit, U, Y):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``predStepSize``
        
        Customization
        -------------
        
        This method normally should not be altered, adjust :func:`~RLframe.controller._criticCost` instead.
        The only customization you might want here is regarding the optimization algorithm

        """        
        
        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        criticOptMethod = 'SLSQP'
        if criticOptMethod == 'trust-constr':
            criticOptOptions = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            criticOptOptions = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
        
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)
    
        W = minimize(lambda W: self._criticCost(W, U, Y), Winit, method=criticOptMethod, tol=1e-7, bounds=bnds, options=criticOptOptions).x
        
        # DEBUG ===================================================================
        # print('-----------------------Critic parameters--------------------------')
        # print( W )
        # /DEBUG ==================================================================
        
        return W
    
    def _actorCost(self, U, y, N, W, delta, mode):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``predStepSize``
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor function in this method. Don't forget to provide description in the class documentation

        """
        
        myU = np.reshape(U, [N, self.dimInput])
        
        Y = np.zeros([N, self.dimOutput])
        
        # System output prediction
        if (mode==1) or (mode==3) or (mode==5):    # Via exogenously passed model
            Y[0, :] = y
            x = self.xSys
            for k in range(1, self.Nactor):
                x = x + delta * self.sysRHS([], x, myU[k-1, :], [])  # Euler scheme
                Y[k, :] = self.sysOut(x)

        elif (mode==2) or (mode==4) or (mode==6):    # Via estimated model
            myU_upsampled = myU.repeat(int(delta/self.samplTime), axis=0)
            Yupsampled, _ = self._dssSim(self.myModel.A, self.myModel.B, self.myModel.C, self.myModel.D, myU_upsampled, self.myModel.x0est, y)
            Y = Yupsampled[::int(delta/self.samplTime)]
        
        J = 0         
        if (mode==1) or (mode==2):     # MPC
            for k in range(N):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
        elif (mode==3) or (mode==4):     # RL: Q-learning with Ncritic-1 roll-outs of running cost
             for k in range(N-1):
                J += self.gamma**k * self.rcost(Y[k, :], myU[k, :])
             J += W @ self._Phi( Y[-1, :], myU[-1, :] )
        elif (mode==5) or (mode==6):     # RL: (normalized) stacked Q-learning
             for k in range(N):
                Q = W @ self._Phi( Y[k, :], myU[k, :] )
                J += 1/N * Q      
        
        return J
    
    def _actor(self, y, Uinit, N, W, delta, mode):
        """
        See class documentation. Parameter ``delta`` here is a shorthand for ``predStepSize``
        
        Customization
        -------------         
        
        This method normally should not be altered, adjust :func:`~RLframe.controller._actorCost`, :func:`~RLframe.controller._actor` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        actorOptMethod = 'SLSQP'
        if actorOptMethod == 'trust-constr':
            actorOptOptions = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            actorOptOptions = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 
       
        isGlobOpt = 0
        
        myUinit = np.reshape(Uinit, [N*self.dimInput,])
        
        bnds = sp.optimize.Bounds(self.Umin, self.Umax, keep_feasible=True)
        
        try:
            if isGlobOpt:
                minimizer_kwargs = {'method': actorOptMethod, 'bounds': bnds, 'tol': 1e-7, 'options': actorOptOptions}
                U = basinhopping(lambda U: self._actorCost(U, y, N, W, delta, mode), myUinit, minimizer_kwargs=minimizer_kwargs, niter = 10).x
            else:
                U = minimize(lambda U: self._actorCost(U, y, N, W, delta, mode), myUinit, method=actorOptMethod, tol=1e-7, bounds=bnds, options=actorOptOptions).x        
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myUinit
        
        # DEBUG ===================================================================
        # ================================Interm output of model prediction quality
        # R  = '\033[31m'
        # Bl  = '\033[30m'
        # myU = np.reshape(U, [N, self.dimInput])    
        # myU_upsampled = myU.repeat(int(delta/self.samplTime), axis=0)
        # Yupsampled, _ = self._dssSim(self.myModel.A, self.myModel.B, self.myModel.C, self.myModel.D, myU_upsampled, self.myModel.x0est, y)
        # Y = Yupsampled[::int(delta/self.samplTime)]
        # Yt = np.zeros([N, self.dimOutput])
        # Yt[0, :] = y
        # x = self.xSys
        # for k in range(1, Nactor):
        #     x = x + delta * self.sysRHS([], x, myU[k-1, :], [])  # Euler scheme
        #     Yt[k, :] = self.sysOut(x)           
        # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
        # dataRow = []
        # for k in range(dimOutput):
        #     dataRow.append( np.mean(Y[:,k] - Yt[:,k]) )
        # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
        # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
        # print(R+table+Bl)
        # /DEBUG ==================================================================     
        
        return U[:self.dimInput]    # Return first action
                    
    def computeAction(self, t, y):
        """
        Main method. See class documentation
        
        Customization
        -------------         
        
        Add your modes, that you introduced in :func:`~RLframe.controller._actorCost`, here

        """       
        
        timeInSample = t - self.ctrlClock
        
        if timeInSample >= self.samplTime: # New sample
            # Update controller's internal clock
            self.ctrlClock = t
            
            if self.mode in (1, 2):  
                
                # Apply control when model estimation phase is over  
                if self.isProbNoise and (self.mode==2):
                    return self.probNoisePow * (rand(self.dimInput) - 0.5)
                
                elif not self.isProbNoise and (self.mode==2):
                    u = self._actor(y, self.Uinit, self.Nactor, [], self.predStepSize, self.mode)

                elif (self.mode==1):
                    u = self._actor(y, self.Uinit, self.Nactor, [], self.predStepSize, self.mode)
                    
            elif self.mode in (3, 4, 5, 6):
                # Critic
                timeInCriticPeriod = t - self.criticClock
                
                # Update data buffers
                self.ubuffer = pushVec(self.ubuffer, self.uCurr)
                self.ybuffer = pushVec(self.ybuffer, y)
                
                if timeInCriticPeriod >= self.criticPeriod:
                    # Update critic's internal clock
                    self.criticClock = t
                    
                    W = self._critic(self.Wprev, self.Winit, self.ubuffer[-self.Ncritic:,:], self.ybuffer[-self.Ncritic:,:])
                    self.Wprev = W
                    
                    # Update initial critic weight for the optimizer. In general, this assignment is subject to tuning
                    # self.Winit = self.Wprev
                    
                else:
                    W = self.Wprev
                    
                # Actor. Apply control when model estimation phase is over
                if self.isProbNoise and (self.mode in (4, 6)):
                    u = self.probNoisePow * (rand(self.dimInput) - 0.5)
                elif not self.isProbNoise and (self.mode in (4, 6)):
                    u = self._actor(y, self.Uinit, self.Nactor, W, self.predStepSize, self.mode)
                    
                    # [EXPERIMENTAL] Call MATLAB's actor
                    # R1 = self.rcostPars[0]
                    # u = eng.optCtrl(eng.transpose(matlab.double(y.tolist())), eng.transpose(matlab.double(self.Uinit.tolist())), 
                    #                                   matlab.double(R1[:dimOutput,:dimOutput].tolist()), matlab.double(R1[dimOutput:,dimOutput:].tolist()), self.gamma,
                    #                                   self.Nactor,
                    #                                   eng.transpose(matlab.double(W.tolist())), 
                    #                                   matlab.double(self.myModel.A.tolist()), 
                    #                                   matlab.double(self.myModel.B.tolist()), 
                    #                                   matlab.double(self.myModel.C.tolist()), 
                    #                                   matlab.double(self.myModel.D.tolist()), 
                    #                                   eng.transpose(matlab.double(self.myModel.x0est.tolist())),
                    #                                   self.mode, 
                    #                                   eng.transpose(matlab.double(self.uMin.tolist())), 
                    #                                   eng.transpose(matlab.double(self.uMax.tolist())), 
                    #                                   dt, matlab.double(self.trueModelPars), self.criticStruct, nargout=1)
                    # u = np.squeeze(np.asarray(u)
                    
                elif self.mode in (3, 5):
                    u = self._actor(y, self.Uinit, self.Nactor, W, self.predStepSize, self.mode) 
            
            self.uCurr = u
            
            return u    
    
        else:
            return self.uCurr

    
class nominalController:
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
    
    def __init__(self, m, I, ctrlGain=10, ctrlBnds=[], t0=0, samplTime=0.1):
        self.m = m
        self.I = I
        self.ctrlGain = ctrlGain
        self.ctrlBnds = ctrlBnds
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
    
        sigmaTilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigmaTilde**3
        
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigmaTilde**3
        
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigmaTilde**3  
    
        return nablaF
    
    def _kappa(self, xNI, theta): 
        """
        Stabilizing controller for NI-part

        """
        kappaVal = np.zeros(2)
        
        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])
                         
        zetaVal = self._zeta(xNI, theta)
        
        kappaVal[0] = - np.abs( np.dot( zetaVal, G[:,0] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,0] ) )
        kappaVal[1] = - np.abs( np.dot( zetaVal, G[:,1] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,1] ) )
        
        return kappaVal
    
    def _Fc(self, xNI, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """
        
        sigmaTilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigmaTilde
        
        z = eta - self._kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)
    
    def _thetaMinimizer(self, xNI, eta):
        thetaInit = 0
        
        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
        
        options = {'maxiter': 50, 'disp': False}
        
        thetaVal = minimize(lambda theta: self._Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x
        
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
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )
        
        eta[0] = omega
        eta[1] = ( yc * np.cos(alpha) - xc * np.sin(alpha) ) * omega + v   
        
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
        
        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
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
        
        if timeInSample >= self.samplTime: # New sample

            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH( y ) 
            thetaStar = self._thetaMinimizer(xNI, eta)
            kappaVal = self._kappa(xNI, thetaStar)
            z = eta - kappaVal
            uNI = - self.ctrlGain * z
            u = self._NH2CartCtrl(xNI, eta, uNI)
            
            if self.ctrlBnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrlBnds[k, 0], self.ctrlBnds[k, 1])           
            
            self.uCurr = u
            
            return u    
    
        else:
            return self.uCurr