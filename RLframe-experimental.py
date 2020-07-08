#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:54:06 2020

@author: Pavel Osinenko
"""


# =============================================================================
# Reinforcement learning frame
# 
# This is a skeleton for reinforcement learning (RL) methods in Python ready for implementation of custom setups, e.g., value iteration, policy iteration, dual etc.
# 
# User settings: 
#       Initialization -- general settings
#       Simulation & visualization: setup
#           main sub
#           controller sub
#           digital elements sub
#
# =============================================================================
#
# Remark: 
#
# All vectors are trated as of type [n,]
# All buffers are trated as of type [L, n] where each row is a vector
# Buffers are updated from bottom


#%% Import packages

import csv
from datetime import datetime

from tabulate import tabulate
# !pip install tabulate <-- to install this

import scipy as sp
import numpy as np
import numpy.linalg as la
# from scipy.integrate import ode <- old, Fortran engine for IVPs
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from numpy.matlib import repmat
from numpy.random import rand
from numpy.random import randn

import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# !pip install mpldatacursor <-- to install this
from mpldatacursor import datacursor

from scipy import signal

# !pip install svgpath2mpl matplotlib <-- to install this
from svgpath2mpl import parse_path

from collections import namedtuple

# System identification packages
# import ssid  # Github:OsinenkoP/pyN4SID, fork of Githug:AndyLamperski/pyN4SID, with some errors fixed
import sippy  # Github:CPCLAB-UNIPI/SIPPY

import warnings

# [EXPERIMENTAL] Use MATLAB's system identification toolbox instead of ssid and sippy
# Compatible MATLAB Runtime and system identification toolobx must be installed
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'~/MATLAB/RL/ENDICart',nargout=0)

#%% Initialization 

#------------------------------------system
# System
dimState = 5
dimInput = 2
dimOutput = 5
dimDisturb = 2

dimFullStateStat = dimState + dimDisturb
dimFullStateDyn = dimState + dimDisturb + dimInput

# System parameters
m = 10 # [kg]
I = 1 # [kg m^2]

# Disturbance
sigma_q_DEF = 1e-3 * np.ones(dimDisturb)
mu_q_DEF = np.zeros(dimDisturb)
tau_q_DEF = np.ones(dimDisturb)

sigma_q = sigma_q_DEF
mu_q = mu_q_DEF
tau_q = tau_q_DEF

#------------------------------------simulation
t0 = 0
t1 = 200
Nruns = 1

x0 = np.zeros(dimState)
x0[0] = 5
x0[1] = 5
x0[2] = np.pi/2

u0 = 0 * np.ones(dimInput)

q0 = 0 * np.ones(dimDisturb)

# Solver
atol = 1e-8
rtol = 1e-8

# xy-plane
xMin = -10
xMax = 10
yMin = -10
yMax = 10

#------------------------------------digital elements
# Digital elements sampling time
dt = 0.01 # [s], controller sampling time
sampleFreq = 1/dt # [Hz]

# Parameters
cutoff = 1 # [Hz]

# Digital differentiator filter order
diffFiltOrd = 4

#------------------------------------model estimator
modEstPhase = 2 # [s]
modEstPeriod = 1*dt # [s]
modEstBufferSize = 200

modelOrder = 5

probNoisePow = 8

# Model estimator stores models in a stack and recall the best of modEstchecks
modEstchecks = 0

#------------------------------------controller
# u[0]: Pushing force F [N]
# u[1]: Steering torque M [N m]

# Manual control
Fman = -3
Nman = -1

# Control constraints
Fmin = -5
Fmax = 5
Mmin = -1
Mmax = 1

# Control horizon length
Nactor = 70

# Should be a multiple of dt
predStepSize = 1*dt # [s]

#------------------------------------RL elements
# Running cost structure and parameters
# Notation: chi = [y, u]
# 1     - quadratic chi.T R1 chi 
# 2     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
rcostStruct = 1

R1 = np.diag([10, 10, 1, 0, 0, 0, 0])  # No mixed terms, full-state measurement
# R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
# R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# Critic stack size, not greater than modEstBufferSize
Ncritic = 50

# Discounting factor
gamma = 1

# Critic is updated every criticPeriod seconds
criticPeriod = 5*dt # [s]

# Critic structure choice
# 1 - quadratic-linear
# 2 - quadratic
# 3 - quadratic, no mixed terms
# 4 - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...] u[0]^2 + ... 
criticStruct = 3

#------------------------------------main switches
isLogData = 1
isVisualization = 0
isPrintSimStep = 1

disturbOn = 0
ctrlConstraintOn = 1

# Static or dynamic controller
isDynCtrl = 0

# Static controller mode (see definition of ctrlStat)
# 0     - manual constant control
# 10    - nominal parking controller
# 1     - model-predictive control (MPC). Prediction via discretized true model
# 2     - adaptive MPC. Prediction via estimated model
# 3     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via discretized true model
# 4     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via estimated model
# 5     - RL: stacked Q-learning. Prediction via discretized true model
# 6     - RL: stacked Q-learning. Prediction via estimated model
ctrlStatMode = 4 

# Dynamic controller mode (see definition of ctrlDyn)
# 0 - PID-controller
ctrlDynMode = 0

# Use or not probing noise during modEstPhase seconds
isProbNoise = 1

# Use global optimization algorithm in actor (increases computation time significantly)
isGlobOpt = 0

#%% Service
def toColVec(argin):
    if argin.ndim < 2:
        return np.reshape(argin, (argin.size, 1))
    elif argin.ndim ==2:
        if argin.shape[0] < argin.shape[1]:
            return argin.T
        else:
            return argin

# To ensure 1D result
def repMat(argin, n, m):
    return np.squeeze(repmat(argin, n, m))

# def pushColRight(matrix, vec):
#     return np.hstack([matrix[:,1:], toColVec(vec)])

def pushVec(matrix, vec):
    return np.vstack([matrix[1:,:], vec])

class ZOH:
    def __init__(self, initTime=0, initVal=0, sampleTime=1):
        self.timeStep = initTime
        self.sampleTime = sampleTime
        self.currVal = initVal
        
    def hold(self, signalVal, t):
        timeInSample = t - self.timeStep
        if timeInSample >= self.sampleTime: # New sample
            self.timeStep = t
            self.currVal = signalVal

        return self.currVal
    
# Real-time digital filter
class dfilter:
    def __init__(self, filterNum, filterDen, bufferSize=16, initTime=0, initVal=0, sampleTime=1):
        self.Num = filterNum
        self.Den = filterDen
        self.zi = repMat( signal.lfilter_zi(filterNum, filterDen), 1, initVal.size)
        
        self.timeStep = initTime
        self.sampleTime = sampleTime
        self.buffer = repMat(initVal, 1, bufferSize)
        
    def filt(self, signalVal, t=None):
        # Sample only if time is specified
        if t is not None:
            timeInSample = t - self.timeStep
            if timeInSample >= self.sampleTime: # New sample
                self.timeStep = t
                self.buffer = pushVec(self.buffer, signalVal)
        else:
            self.buffer = pushVec(self.buffer, signalVal)
        
        bufferFiltered = np.zeros(self.buffer.shape)
        
        for k in range(0, signalVal.size):
                bufferFiltered[k,:], self.zi[k] = signal.lfilter(self.Num, self.Den, self.buffer[k,:], zi=self.zi[k, :])
        return bufferFiltered[-1,:]
    
def printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u):
    # alphaDeg = alpha/np.pi*180      
    
    headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]']  
    dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]  
    rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')   
    table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
    
    print(table)
    
def logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
    with open(dataFile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]])

#%% System

def sysStateDyn(t, x, u, q):
    global m, I, disturbOn
    # x[0] -- x [m]
    # x[1] -- y [m]
    # x[2] -- alpha [rad]
    # x[3] -- v [m/s]
    # x[4] -- omega [rad/s]
    Dx = np.zeros(dimState)
    Dx[0] = x[3] * np.cos( x[2] )
    Dx[1] = x[3] * np.sin( x[2] )
    Dx[2] = x[4]
    
    if disturbOn:
        Dx[3] = 1/m * (u[0] + q[0])
        Dx[4] = 1/I * (u[1] + q[1])
    else:
        Dx[3] = 1/m * u[0]
        Dx[4] = 1/I * u[1]
        
    return Dx

def sysOut(x, measNoise=np.zeros(dimOutput)):
    y = np.zeros(dimOutput)
    # y = x[:3] + measNoise # <-- Measure only position and orientation
    y = x  # <-- Position, force and torque sensors on
    return y    

#%% Controller

# Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)
def zeta(xNI, theta):
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

# Stabilizing controller for NI-part
def kappa(xNI, theta): 
    kappaVal = np.zeros(2)
    
    G = np.zeros([3, 2])
    G[:,0] = np.array([1, 0, xNI[1]])
    G[:,1] = np.array([0, 1, -xNI[0]])
                     
    zetaVal = zeta(xNI, theta)
    
    kappaVal[0] = - np.abs( np.dot( zetaVal, G[:,0] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,0] ) )
    kappaVal[1] = - np.abs( np.dot( zetaVal, G[:,1] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,1] ) )
    
    return kappaVal

# Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned above
def Fc(xNI, eta, theta):
    sigmaTilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
    
    F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigmaTilde
    
    z = eta - kappa(xNI, theta)
    
    return F + 1/2 * np.dot(z, z)

def thetaMinimizer(xNI, eta):
    thetaInit = 0
    
    bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
    
    options = {'maxiter': 50, 'disp': False}
    
    thetaVal = minimize(lambda theta: Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x
    
    return thetaVal
    
# Transformation from Cartesian coordinates to non-holonomic (NH) coordinates
# See Section VIII.A in Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
# integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
#
# The transformation is a bit different since the 3rd NI eqn reads for our case as: dot x3 = x2 u1 - x1 u2
def Cart2NH(CartCoords): 
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

# Get control for Cartesian NI from NH coordinates
# See Section VIII.A in Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
# integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)
#
# The transformation is a bit different since the 3rd NI eqn reads for our parking controller as: dot x3 = x2 u1 - x1 u2
#     
def NH2CartCtrl(xNI, eta, uNI): 
    global m, I
    
    uCart = np.zeros(2)
    
    uCart[0] = m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
    uCart[1] = I * uNI[0]
    
    return uCart

# Convert upper triangular square sub-matrix to column vector
def uptria2vec(mat):
    n = mat.shape[0]
    
    vec = np.zeros( n*(n+1)/2, 1 )
    
    k = 0
    for i in range(n):
        for j in range(n):
            vec[j] = mat[i, j]
            k += 1
    

# Simulate output response of discrete-time state-space model
def dssSim(A, B, C, D, uSqn, x0, y0):
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

# Feature vector of critic
def Phi(y, u):
    chi = np.concatenate([y, u])
    
    if criticStruct == 1:
        return np.concatenate([ uptria2vec( np.kron(chi, chi) ), chi ])
    elif criticStruct == 2:
        return np.concatenate([ uptria2vec( np.kron(chi, chi) ) ])   
    elif criticStruct == 3:
        return chi * chi    
    elif criticStruct == 4:
        return np.concatenate([ y**2, np.kron(y, u), u**2 ]) 

def criticCost(W, U, Y, Wprev):
    Jc = 0
    
    for k in range(dimCrit, 0, -1):
        yPrev = Y[k-1, :]
        yNext = Y[k, :]
        uPrev = U[k-1, :]
        uNext = U[k, :]
        
        # Temporal difference
        e = W @ Phi( yPrev, uPrev ) - gamma * Wprev @ Phi( yNext, uNext ) - rcost(yPrev, uPrev)
        
        Jc += 1/2 * e**2
        
    return Jc
    
    
def critic(Wprev, Winit, U, Y):  
    global Wmin, Wmax, criticOptMethod, criticOptOptions
    
    bnds = sp.optimize.Bounds(Wmin, Wmax, keep_feasible=True)

    W = minimize(lambda W: criticCost(W, U, Y, Wprev), Winit, method=criticOptMethod, tol=1e-7, bounds=bnds, options=criticOptOptions).x
    
    # DEBUG ===================================================================
    # print('-----------------------Critic parameters--------------------------')
    # print( W )
    # /DEBUG ==================================================================
    
    return W

def actorCost(U, y, N, W, delta, mode):
    myU = np.reshape(U, [N, dimInput])
    
    Y = np.zeros([N, dimOutput])
    
    # System output prediction
    if (mode==1) or (mode==3) or (mode==5):    # Via true model
        Y[0, :] = y
        x = closedLoopStat.x0true
        for k in range(1, Nactor):
            x = x + delta * sysStateDyn([], x, myU[k-1, :], [])  # Euler scheme
            Y[k, :] = sysOut(x)
            # Y[k, :] = Y[k-1, :] + dt * sysStateDyn([], Y[k-1, :], myU[k-1, :], [])  # Euler scheme
    elif (mode==2) or (mode==4) or (mode==6):    # Via estimated model
        myU_upsampled = myU.repeat(int(delta/dt), axis=0)
        Yupsampled, _ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, myU_upsampled, ctrlStat.x0est, y)
        Y = Yupsampled[::int(delta/dt)]
    
    J = 0         
    if (mode==1) or (mode==2):     # MPC
        for k in range(N):
            J += gamma**k * rcost(Y[k, :], myU[k, :])
    elif (mode==3) or (mode==4):     # RL: Q-learning with Ncritic roll-outs of running cost
         for k in range(N-1):
            J += gamma**k * rcost(Y[k, :], myU[k, :])
         J += W @ Phi( Y[-1, :], myU[-1, :] )
    elif (mode==5) or (mode==6):     # RL: (normalized) stacked Q-learning
         for k in range(N):
            Q = W @ Phi( Y[k, :], myU[k, :] )
            J += 1/N * Q
            
    # DEBUG ===================================================================
    # ================================Interm output of model prediction quality
    # R  = '\033[31m'
    # Bl  = '\033[30m'
    # Yt = np.zeros([N, dimOutput])
    # Yt[0, :] = y
    # x = closedLoopStat.x0true
    # for k in range(1, Nactor):
    #     x = x + dt * sysStateDyn([], x, myU[k-1, :], [])  # Euler scheme
    #     Yt[k, :] = sysOut(x)           
    # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
    # dataRow = []
    # for k in range(dimOutput):
    #     dataRow.append( np.mean(Y[:,k] - Yt[:,k]) )
    # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
    # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
    # print(R+table+Bl)
    # /DEBUG ==================================================================        
        
    return J

# Optimal controller a.k.a. actor in RL terminology
def actor(y, Uinit, N, W, A, B, C, D, x0, delta, mode):
    global actorOptMethod, actorOptOptions, Umin, Umax
    
    myUinit = np.reshape(Uinit, [N*dimInput,])
    
    bnds = sp.optimize.Bounds(Umin, Umax, keep_feasible=True)
    
    try:
        if isGlobOpt:
            minimizer_kwargs = {'method': actorOptMethod, 'bounds': bnds, 'tol': 1e-7, 'options': actorOptOptions}
            U = basinhopping(lambda U: actorCost(U, y, N, W, delta, mode), myUinit, minimizer_kwargs=minimizer_kwargs, niter = 10).x
        else:
            U = minimize(lambda U: actorCost(U, y, N, W, delta, mode), myUinit, method=actorOptMethod, tol=1e-7, bounds=bnds, options=actorOptOptions).x        
    except ValueError:
        print('Actor''s optimizer failed. Returning default action')
        U = myUinit
    
    # DEBUG ===================================================================
    # ================================Interm output of model prediction quality
    R  = '\033[31m'
    Bl  = '\033[30m'
    myU = np.reshape(U, [N, dimInput])    
    myU_upsampled = myU.repeat(int(delta/dt), axis=0)
    Yupsampled, _ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, myU_upsampled, ctrlStat.x0est, y)
    Y = Yupsampled[::int(delta/dt)]
    Yt = np.zeros([N, dimOutput])
    Yt[0, :] = y
    x = closedLoopStat.x0true
    for k in range(1, Nactor):
        x = x + delta * sysStateDyn([], x, myU[k-1, :], [])  # Euler scheme
        Yt[k, :] = sysOut(x)           
    headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
    dataRow = []
    for k in range(dimOutput):
        dataRow.append( np.mean(Y[:,k] - Yt[:,k]) )
    rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
    table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
    print(R+table+Bl)
    # /DEBUG ==================================================================     
    
    return U[:dimInput]    # Return first action

# Controller is called outside the simulator because the latter performs multiple iterations when doing one integration step which might distort the results.
# Therefore, the control action is stored as a static variable and passed to the closed loop routine which is in turn called by the simulator
def ctrlStat(y, t, sampleTime = dt):
    global uMin, uMax, Fman, Nman, ctrlStatMode, PID, isProbNoise, modelOrder
    
    # In ctrlStat, a ZOH is built-in
    timeInSample = t - ctrlStat.ctrlClock
    
    if timeInSample >= sampleTime: # New sample
        # Update controller's internal clock
        ctrlStat.ctrlClock = t
        
        #------------------------------------model update
        # Update buffers when using RL or requiring estimated model
        if ctrlStatMode in (2,3,4,5,6):
            timeInEstPeriod = t - ctrlStat.estClock
            
            ctrlStat.modEst_ubuffer = pushVec(ctrlStat.modEst_ubuffer, ctrlStat.sampled_u)
            ctrlStat.modEst_ybuffer = pushVec(ctrlStat.modEst_ybuffer, y)
        
            # Estimate model if required by ctrlStatMode
            if (timeInEstPeriod >= modEstPeriod) and (ctrlStatMode in (2,4,6)):
                # Update model estimator's internal clock
                ctrlStat.estClock = t
                
                try:
                    # Using ssid from Githug:AndyLamperski/pyN4SID
                    # Aid, Bid, Cid, Did, _ ,_ = ssid.N4SID(ctrlStat.modEst_ubuffer.T,  ctrlStat.modEst_ybuffer.T, 
                    #                                       NumRows = dimInput + modelOrder,
                    #                                       NumCols = modEstBufferSize - (dimInput + modelOrder)*2,
                    #                                       NSig = modelOrder,
                    #                                       require_stable=False) 
                    # ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D = Aid, Bid, Cid, Did
                    # ctrlStat.x0est = np.zeros(modelOrder)
                    
                    # Using Github:CPCLAB-UNIPI/SIPPY 
                    # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                    SSest = sippy.system_identification(ctrlStat.modEst_ybuffer, ctrlStat.modEst_ubuffer,
                                                        id_method='N4SID',
                                                        SS_fixed_order=modelOrder,
                                                        SS_D_required=False,
                                                        SS_A_stability=False,
                                                        # SS_f=int(modEstBufferSize/12),
                                                        # SS_p=int(modEstBufferSize/10),
                                                        SS_PK_B_reval=False,
                                                        tsample=dt)
                    ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D = SSest.A, SSest.B, SSest.C, SSest.D
                    # ctrlStat.x0est = SSest.x0[:,0] 
                    
                    # [EXPERIMENTAL] Using MATLAB's system identification toolbox
                    # us_ml = eng.transpose(matlab.double(ctrlStat.modEst_ubuffer.tolist()))
                    # ys_ml = eng.transpose(matlab.double(ctrlStat.modEst_ybuffer.tolist()))
                    
                    # Aml, Bml, Cml, Dml = eng.mySSest_simple(ys_ml, us_ml, dt, modelOrder, nargout=4)
                    
                    # ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D = np.asarray(Aml), np.asarray(Bml), np.asarray(Cml), np.asarray(Dml)
                    
                except:
                    print('Model estimation problem')
                    ctrlStat.A = np.zeros( [modelOrder, modelOrder] )
                    ctrlStat.B = np.zeros( [modelOrder, dimInput] )
                    ctrlStat.C = np.zeros( [dimOutput, modelOrder] )
                    ctrlStat.D = np.zeros( [dimOutput, dimInput] )
                
                #---model checks
                if modEstchecks > 0:
                    # Update estimated model parameter stacks
                    ctrlStat.modEstAstack.pop(0)
                    ctrlStat.modEstAstack.append(ctrlStat.A)
                    
                    ctrlStat.modEstBstack.pop(0)
                    ctrlStat.modEstBstack.append(ctrlStat.B)
                    
                    ctrlStat.modEstCstack.pop(0)
                    ctrlStat.modEstCstack.append(ctrlStat.C)
                    
                    ctrlStat.modEstDstack.pop(0)
                    ctrlStat.modEstDstack.append(ctrlStat.D)
                    
                    # Perform check of stack of models and pick the best
                    totAbsErrCurr = 1e8
                    for k in range(modEstchecks):
                        A, B, C, D = ctrlStat.modEstAstack[k], ctrlStat.modEstBstack[k], ctrlStat.modEstCstack[k], ctrlStat.modEstDstack[k]
                        x0est,_,_,_ = np.linalg.lstsq(C, y)
                        Yest,_ = dssSim(A, B, C, D, ctrlStat.modEst_ubuffer, x0est, y)
                        meanErr = np.mean(Yest - ctrlStat.modEst_ybuffer, axis=0)
                        
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
                            ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D = A, B, C, D
                    
                    # DEBUG ===================================================================
                    # ==========================================Print quality of the best model
                    # R  = '\033[31m'
                    # Bl  = '\033[30m'
                    # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, y)
                    # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.modEst_ubuffer, x0est, y)
                    # meanErr = np.mean(Yest - ctrlStat.modEst_ybuffer, axis=0)
                    # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']  
                    # dataRow = []
                    # for k in range(dimOutput):
                    #     dataRow.append( meanErr[k] )
                    # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                    # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                    # print(R+table+Bl)
                    # /DEBUG ===================================================================                       
        
        # Update initial state estimate        
        ctrlStat.x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, y)
        
        if t >= modEstPhase:
                # Drop probing noise
                isProbNoise = 0  
        
        #------------------------------------control: manual
        if ctrlStatMode==0:         
            ctrlStat.sampled_u[0] = Fman
            ctrlStat.sampled_u[1] = Nman
         
        #------------------------------------control: nominal    
        elif ctrlStatMode==10:
            # For the algorithm, refer to Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
            # via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887
            
            # Gain for nominal controller
            kNom = 50
            
            # This controller needs full-state measurement
            xNI, eta = Cart2NH( closedLoopStat.x0true ) 
            thetaStar = thetaMinimizer(xNI, eta)
            kappaVal = kappa(xNI, thetaStar)
            z = eta - kappaVal
            uNI = - kNom * z
            ctrlStat.sampled_u = NH2CartCtrl(xNI, eta, uNI)
            
        #------------------------------------control: MPC    
        elif ctrlStatMode in (1, 2):           
            Uinit = repMat( uMin/10 , 1, Nactor )
            
            # Apply control when model estimation phase is over
            if isProbNoise and (ctrlStatMode==2):
                ctrlStat.sampled_u = probNoisePow * (rand(dimInput) - 0.5)
            elif not isProbNoise and (ctrlStatMode==2):
                ctrlStat.sampled_u = actor(y, Uinit, Nactor, [], ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, ctrlStatMode)
                # DEBUG ===================================================================
                # =================================Comparison with control using true model
                # tmp_u = actor(y, Uinit, uMin, uMax, Nactor, W, ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, 1)
                # headerRow = ['u1', 'u2', 'ut1', 'ut2']  
                # dataRow = []
                # dataRow.append( ctrlStat.sampled_u[0] )
                # dataRow.append( ctrlStat.sampled_u[1] )
                # dataRow.append( tmp_u[0] )
                # dataRow.append( tmp_u[1] )
                # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f')   
                # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                # print( table )
                # /DEBUG ==================================================================
            elif (ctrlStatMode==1):
                ctrlStat.sampled_u = actor(y, Uinit, Nactor, [], ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, ctrlStatMode)
                
        #------------------------------------control: RL
        elif ctrlStatMode in (3, 4, 5, 6):
            # Critic
            timeInCriticPeriod = t - ctrlStat.criticClock
            if timeInCriticPeriod >= criticPeriod:
                W = critic(ctrlStat.Wprev, Winit, ctrlStat.modEst_ubuffer[-Ncritic:,:], ctrlStat.modEst_ybuffer[-Ncritic:,:])
                ctrlStat.Wprev = W
                # Update critic's internal clock
                ctrlStat.criticClock = t
            else:
                W = ctrlStat.Wprev
                
            # Actor. Apply control when model estimation phase is over
            if isProbNoise and (ctrlStatMode in (4, 6)):
                ctrlStat.sampled_u = probNoisePow * (rand(dimInput) - 0.5)
            elif not isProbNoise and (ctrlStatMode in (4, 6)):
                Uinit = repMat(uMin/10, Nactor, 1)
                ctrlStat.sampled_u = actor(y, Uinit, Nactor, W, ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, ctrlStatMode)
                
                # [EXPERIMENTAL] Call MATLAB's actor
                # ctrlStat.sampled_u = eng.optCtrl(eng.transpose(matlab.double(y.tolist())), eng.transpose(matlab.double(Uinit.tolist())), 
                #                                   matlab.double(R1[:dimOutput,:dimOutput].tolist()), matlab.double(R1[dimOutput:,dimOutput:].tolist()), gamma,
                #                                   Nactor,
                #                                   eng.transpose(matlab.double(W.tolist())), 
                #                                   matlab.double(ctrlStat.A.tolist()), 
                #                                   matlab.double(ctrlStat.B.tolist()), 
                #                                   matlab.double(ctrlStat.C.tolist()), 
                #                                   matlab.double(ctrlStat.D.tolist()), 
                #                                   eng.transpose(matlab.double(ctrlStat.x0est.tolist())),
                #                                   ctrlStatMode, 
                #                                   eng.transpose(matlab.double(uMin.tolist())), 
                #                                   eng.transpose(matlab.double(uMax.tolist())), 
                #                                   dt, matlab.double([m,I]), criticStruct, nargout=1)
                # ctrlStat.sampled_u = np.squeeze(np.asarray(ctrlStat.sampled_u))
                
                # DEBUG ===================================================================
                # =================================Comparison with control using true model
                # tmp_u = actor(y, Uinit, uMin, uMax, Nactor, W, ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, 5)
                # headerRow = ['u1', 'u2', 'ut1', 'ut2', 'diff u1', 'diff u2']  
                # dataRow = []
                # dataRow.append( ctrlStat.sampled_u[0] )
                # dataRow.append( ctrlStat.sampled_u[1] )
                # dataRow.append( tmp_u[0] )
                # dataRow.append( tmp_u[1] )
                # dataRow.append( ctrlStat.sampled_u[0]-tmp_u[0] )
                # dataRow.append( ctrlStat.sampled_u[1]-tmp_u[1] )
                # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f', '8.5f')   
                # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')  
                # print( table )
                # /DEBUG ==================================================================
            elif ctrlStatMode in (3, 5):
                Uinit = repMat( uMin/10 , Nactor, 1)
                ctrlStat.sampled_u = actor(y, Uinit, Nactor, W, ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.x0est, predStepSize, ctrlStatMode)            

# ===> ToDo: buffer update to move outside the simulator <===
def ctrlDyn(t, u, y):
    global PID, ctrlDynMode, diffFilters
    
    ctrlDyn.ybuffer = pushVec(ctrlDyn.ybuffer, y)
    
    Du = np.zeros(dimInput)
    if ctrlDynMode==0:
        
        # 1st difference
        if t - ctrlDyn.itime > 0:
            Dy = ( ctrlDyn.ybuffer[1,:] - ctrlDyn.ybuffer[0,:] )/(t - ctrlDyn.itime)
        else:
            Dy = y
        
        # Using differentiator filer. Warning: lfilter is slow
        # Dy = diffFilters.y.filt(y, t=t)
        # DDy = diffFilters.Dy.filt(Dy, t=t)
        
        trajNrm = la.norm(y[:2])
        DtrajNrm = np.dot( 1/trajNrm * np.array([y[0], y[1]]), Dy[:2] )
        alpha = y[2]
        Dalpha = Dy[2]
        
        Du[0] = -PID.P[0] * DtrajNrm - PID.I[0] * trajNrm
        Du[1] = -PID.P[1] * Dalpha - PID.I[1] * alpha

    ctrlDyn.itime = t

    return Du

#%% Disturbance dynamics

# Simple 1st order filter of white Gaussian noise
def disturbDyn(t, q, sigma_q=sigma_q_DEF, mu_q=mu_q_DEF, tau_q=tau_q_DEF):
    Dq = np.zeros(dimDisturb)
    for k in range(0, dimDisturb):
        Dq[k] = - tau_q[k] * ( q[k] + sigma_q[k] * randn() + mu_q[k])
    return Dq

#%% Closed loop

def closedLoopStat(t, ksi, sigma_q=sigma_q_DEF, mu_q=mu_q_DEF, tau_q=tau_q_DEF, sampleTime = dt):
    global ctrlConstraintOn
   
    # print('INTERNAL t = {time:2.3f}'.format(time=t))
    
    DfullState = np.zeros(dimFullStateStat)
    
    x = ksi[0:dimState]
    q = ksi[dimState:]
    
    # Get the control action
    u = ctrlStat.sampled_u
    
    if ctrlConstraintOn:
        u[0] = np.clip(u[0], Fmin, Fmax)
        u[1] = np.clip(u[1], Mmin, Mmax)
    
    DfullState[0:dimState] = sysStateDyn(t, x, u, q)
    DfullState[dimState:] = disturbDyn(t, q, sigma_q = sigma_q, mu_q = mu_q, tau_q = tau_q)
    
    # Track system's state for some controllers
    closedLoopStat.x0true = x
    
    return DfullState

def closedLoopDyn(t, ksi, sigma_q=sigma_q_DEF, mu_q=mu_q_DEF, tau_q=tau_q_DEF):
    global ctrlConstraintOn, ZOHs
    
    DfullState = np.zeros(dimFullStateDyn)
    
    x = ksi[0:dimState]
    q = ksi[dimState:dimState+dimDisturb]
    u = ksi[-dimInput:]
    
    u = ZOHs.u.hold(u, t)
    
    y = sysOut(x)
    
    if ctrlConstraintOn:
        u[0] = np.clip(u[0], Fmin, Fmax)
        u[1] = np.clip(u[1], Mmin, Mmax)
    
    DfullState[0:dimState] = sysStateDyn(t, x, u, q)
    DfullState[dimState:dimState+dimDisturb] = disturbDyn(t, q, sigma_q = sigma_q, mu_q = mu_q, tau_q = tau_q)
    DfullState[-dimInput:] = ctrlDyn(t, u, y)
    
    return DfullState

#%% Cost 

# Running cost
def rcost(y, u):
    global R1, R2, rcostStruct
    chi = np.concatenate([y, u])
    
    r = 0
    
    if rcostStruct == 1:
        r = chi @ R1 @ chi
    elif rcostStruct == 2:
        r = chi**2 @ R2 @ chi**2 + chi @ R1 @ chi
    
    return r
    
# Integrated cost
def icost(r, t):
    icost.val += r*(t - icost.itime)
    icost.itime = t
    return icost.val      

#%% Visualization: utilities
    
def updateLine(line, newX, newY):
    line.set_xdata( np.append( line.get_xdata(), newX) )
    line.set_ydata( np.append( line.get_ydata(), newY) )  
    
def resetLine(line):
    line.set_data([], [])     
 
def updateScatter(scatter, newX, newY):
    scatter.set_offsets( np.vstack( [ scatter.get_offsets().data, np.c_[newX, newY] ] ) )
    
def updateText(textHandle, newText):
    textHandle.set_text(newText)
    
class pltMarker:
    def __init__(self, angle=None, pathString=None):
        self.angle = angle or []
        self.pathString = pathString or """m 66.893258,227.10128 h 5.37899 v 0.91881 h 1.65571 l 1e-5,-3.8513 3.68556,-1e-5 v -1.43933
        l -2.23863,10e-6 v -2.73937 l 5.379,-1e-5 v 2.73938 h -2.23862 v 1.43933 h 3.68556 v 8.60486 l -3.68556,1e-5 v 1.43158
        h 2.23862 v 2.73989 h -5.37899 l -1e-5,-2.73989 h 2.23863 v -1.43159 h -3.68556 v -3.8513 h -1.65573 l 1e-5,0.91881 h -5.379 z"""
        self.path = parse_path( self.pathString )
        self.path.vertices -= self.path.vertices.mean( axis=0 )
        self.marker = mpl.markers.MarkerStyle( marker=self.path )
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(angle-self.angle)
        self.angle = angle

def onKeyPress(event):
    global lines    

    if event.key==' ':
        if anm.running:
            anm.event_source.stop()
            
        else:
            anm.event_source.start()
        anm.running ^= True
    elif event.key=='q':
        plt.close('all')
        raise Exception('exit')
        
#%% Simulation & visualization: setup

#------------------------------------main

y0 = sysOut(x0)

icost.val = 0
icost.itime = t0

xCoord0 = x0[0]
yCoord0 = x0[1]
alpha0 = x0[2]
alphaDeg0 = alpha0/2/np.pi

# Data logging init
dataFolder = 'data'

if isLogData:
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H.%M.%S")
    dataFiles = [None] * Nruns
    for k in range(0, Nruns):
        dataFiles[k] = dataFolder + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
        with open(dataFiles[k], 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]'] )
    dataFile = dataFiles[0]

# Do not display annoying warnings when print is on
if isPrintSimStep:
    warnings.filterwarnings('ignore')

# Track system's state for some controllers
closedLoopStat.x0true = x0    

#------------------------------------model estimator
if isDynCtrl:
    pass
else:
    ctrlStat.estClock = t0
    ctrlStat.modEst_ubuffer = np.zeros([ modEstBufferSize, dimInput] )
    ctrlStat.modEst_ybuffer = np.zeros( [modEstBufferSize, dimOutput] )
    
    # Initial model estimate is subject to tuning
    ctrlStat.A = np.zeros( [modelOrder, modelOrder] )
    ctrlStat.B = np.zeros( [modelOrder, dimInput] )
    ctrlStat.C = np.zeros( [dimOutput, modelOrder] )
    ctrlStat.D = np.zeros( [dimOutput, dimInput] )
    ctrlStat.x0est = np.zeros( modelOrder )
    
ctrlStat.modEstAstack = [ctrlStat.A] * modEstchecks
ctrlStat.modEstBstack = [ctrlStat.B] * modEstchecks
ctrlStat.modEstCstack = [ctrlStat.C] * modEstchecks
ctrlStat.modEstDstack = [ctrlStat.D] * modEstchecks

#------------------------------------controller
if isDynCtrl:
    ctrlDyn.ybuffer = repMat(y0, 1, 2)
    ctrlDyn.itime = t0
else:
    ctrlStat.ctrlClock = t0
    ctrlStat.sampled_u = u0
  
# Stack constraints    
uMin = np.array([Fmin, Mmin])
uMax = np.array([Fmax, Mmax])
Umin = repMat(uMin, 1, Nactor)
Umax = repMat(uMax, 1, Nactor)
    
# Optimization method of actor    
# Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
actorOptMethod = 'SLSQP'
if actorOptMethod == 'trust-constr':
    actorOptOptions = {'maxiter': 300, 'disp': False} #'disp': True, 'verbose': 2}
else:
    actorOptOptions = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2}    

cPID = namedtuple('PID', ['P', 'I', 'D'])
PID = cPID(P=[1, 1], I=[.5, .5], D=[0, 0])

#------------------------------------RL elements
    
if criticStruct == 1:
    dimCrit = ( ( dimOutput + dimInput ) + 1 ) * ( dimOutput + dimInput )/2 + (dimOutput + dimInput)  
    Wmin = -1e3*np.ones(dimCrit) 
    Wmax = 1e3*np.ones(dimCrit) 
elif criticStruct == 2:
    dimCrit = ( ( dimOutput + dimInput ) + 1 ) * ( dimOutput + dimInput )/2
    Wmin = np.zeros(dimCrit) 
    Wmax = 1e3*np.ones(dimCrit)    
elif criticStruct == 3:
    dimCrit = dimOutput + dimInput
    Wmin = np.zeros(dimCrit) 
    Wmax = 1e3*np.ones(dimCrit)    
elif criticStruct == 4:
    dimCrit = dimOutput + dimOutput * dimInput + dimInput
    Wmin = -1e3*np.ones(dimCrit) 
    Wmax = 1e3*np.ones(dimCrit)    
  
# Critic weights
Winit = rand(dimCrit)

if isDynCtrl:
    pass
else:
    ctrlStat.Wprev = Winit
    ctrlStat.criticClock = 0

# Optimization method of critic    
# Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
criticOptMethod = 'SLSQP'
if actorOptMethod == 'trust-constr':
    criticOptOptions = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
else:
    criticOptOptions = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 

# Clip critic buffer size
Ncritic = np.min([Ncritic, modEstBufferSize-1])

#------------------------------------digital elements

# Differentiator filters
diffFilterNum = signal.remez(diffFiltOrd+1, [0, cutoff], [1], Hz=sampleFreq, type='differentiator')
diffFilterDen = np.array([1.0])

cdiffFilters = namedtuple('diffFilter', ['y', 'Dy'])
diffFilters = cdiffFilters(y = dfilter(diffFilterNum, diffFilterDen, initVal=y0, initTime=t0, sampleTime=dt, bufferSize = 4),
                           Dy = dfilter(diffFilterNum, diffFilterDen, initVal=y0, initTime=t0, sampleTime=dt, bufferSize = 4))

# Zero-order holds
cZOHs = namedtuple('ZOH', ['u', 'y'])
ZOHs = cZOHs(u = ZOH(initTime=t0, initVal=u0, sampleTime=dt),
             y = ZOH(initTime=t0, initVal=y0, sampleTime=dt))

#------------------------------------simulator
if isDynCtrl:
    ksi0 = np.concatenate([x0, q0, u0])
    simulator = sp.integrate.RK45(lambda t, ksi: closedLoopDyn(t, ksi, sigma_q, mu_q, tau_q, dt), 
                                  t0, ksi0, t1, first_step=1e-6, atol=atol, rtol=rtol)
else:
    ksi0 = np.concatenate([x0, q0])
    simulator = sp.integrate.RK45(lambda t, ksi: closedLoopStat(t, ksi, sigma_q, mu_q, tau_q, dt), 
                                  t0, ksi0, t1, first_step=1e-6, atol=atol, rtol=rtol)

#------------------------------------visuals
if isVisualization:  
   
    plt.close('all')
     
    simFig = plt.figure(figsize=(10,10))    
        
    # xy plane  
    xyPlaneAxs = simFig.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax), xlabel='x [m]', ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')
    xyPlaneAxs.set_aspect('equal', adjustable='box')
    xyPlaneAxs.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
    xyPlaneAxs.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
    trajLine, = xyPlaneAxs.plot(xCoord0, yCoord0, 'b--', lw=0.5)
    robotMarker = pltMarker(angle=alphaDeg0)
    textTime = 't = {time:2.3f}'.format(time = t0)
    textTimeHandle = xyPlaneAxs.text(0.05, 0.95, textTime, horizontalalignment='left', verticalalignment='center', transform=xyPlaneAxs.transAxes)
    xyPlaneAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
    
    # Solution
    solAxs = simFig.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
    solAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
    normLine, = solAxs.plot(t0, la.norm([xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
    alphaLine, = solAxs.plot(t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
    solAxs.legend(fancybox=True, loc='upper right')
    solAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
    
    # Cost
    costAxs = simFig.add_subplot(223, autoscale_on=False,
                                 xlim=(t0,t1),
                                 ylim=(0, 1e3*rcost( y0, u0 ) ),
                                 yscale='symlog', xlabel='t [s]')
    r = rcost(y0, u0)
    # textRcost = 'r = {r:2.3f}'.format(r = r)
    # textRcostHandle = simFig.text(0.05, 0.05, textRcost, horizontalalignment='left', verticalalignment='center')
    textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = icost.val)
    textIcostHandle = simFig.text(0.05, 0.5, textIcost, horizontalalignment='left', verticalalignment='center')
    rcostLine, = costAxs.plot(t0, r, 'r-', lw=0.5, label='r')
    icostLine, = costAxs.plot(t0, icost.val, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
    costAxs.legend(fancybox=True, loc='upper right')
    
    # Control
    ctrlAxs = simFig.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
    ctrlAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
    ctrlLines = ctrlAxs.plot(t0, toColVec(u0).T, lw=0.5)
    ctrlAxs.legend(iter(ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
    
    # Pack all lines together
    cLines = namedtuple('lines', ['trajLine', 'normLine', 'alphaLine', 'rcostLine', 'icostLine', 'ctrlLines'])
    lines = cLines(trajLine=trajLine, normLine=normLine, alphaLine=alphaLine, rcostLine=rcostLine, icostLine=icostLine, ctrlLines=ctrlLines)
    
    # Enable data cursor
    for item in lines:
        if isinstance(item, list):
            for subitem in item:
                datacursor(subitem)
        else:
            datacursor(item)

#%% Simulation & visualization: init & animate

def initAnim():
    animate.solScatter = xyPlaneAxs.scatter(xCoord0, yCoord0, marker=robotMarker.marker, s=400, c='b')
    animate.currRun = 1
    return animate.solScatter, animate.currRun, 
    
def animate(k):
    global dataFile
    
    #------------------------------------simStep
    simulator.step()
    
    t = simulator.t
    ksi = simulator.y
    
    x = ksi[0:dimState]
    y = sysOut(x)
    
    if isDynCtrl:
        u = ksi[-dimInput:]
    else:
        ctrlStat(y, t, sampleTime=dt)   # Updates ctrlStat.sampled_u
        u = ctrlStat.sampled_u
    
    xCoord = ksi[0]
    yCoord = ksi[1]
    alpha = ksi[2]
    alphaDeg = alpha/np.pi*180
    v = ksi[3]
    omega = ksi[4]
    
    r = rcost(y, u)
    icost.val = icost(r, t)
    
    if isPrintSimStep:
        printSimStep(t, xCoord, yCoord, alpha, v, omega, icost.val, u)
        
    if isLogData:
        logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, icost.val, u)
    
    #------------------------------------visuals     
    # xy plane    
    textTime = 't = {time:2.3f}'.format(time = t)
    updateText(textTimeHandle, textTime)
    updateLine(trajLine, *ksi[:2])  # Update the robot's track on the plot
    
    robotMarker.rotate(alphaDeg)    # Rotate the robot on the plot  
    animate.solScatter.remove()
    animate.solScatter = xyPlaneAxs.scatter(xCoord, yCoord, marker=robotMarker.marker, s=400, c='b')
    
    # Solution
    updateLine(normLine, t, la.norm([xCoord, yCoord]))
    updateLine(alphaLine, t, alpha)

    # Cost
    updateLine(rcostLine, t, r)
    updateLine(icostLine, t, icost.val)
    textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost.val)
    updateText(textIcostHandle, textIcost)
    # Control
    for (line, uSingle) in zip(ctrlLines, u):
        updateLine(line, t, uSingle)

    #------------------------------------run done
    if t >= t1:  
        if isPrintSimStep:
                print('.....................................Run {run:2d} done.....................................'.format(run = animate.currRun))
            
        animate.currRun += 1
        
        if animate.currRun > Nruns:
            anm.event_source.stop()
            return
        
        if isLogData:
            dataFile = dataFiles[animate.currRun-1]
        
        # Reset simulator
        simulator.status = 'running'
        simulator.t = t0
        simulator.y = ksi0
        
        # Reset controller
        if isDynCtrl:
            ctrlDyn.ybuffer = repMat(y0, 2, 1)
            ctrlDyn.itime = t0
        else:
            ctrlStat.ctrlClock = t0
            ctrlStat.sampled_u = u0
        
        icost.val = 0      
        
        for item in lines:
            if item != trajLine:
                if isinstance(item, list):
                    for subitem in item:
                        resetLine(subitem)
                else:
                    resetLine(item)

        updateLine(trajLine, np.nan, np.nan)
    
    return animate.solScatter

#%% Simulation & visualization: main loop

if isVisualization:
    cId = simFig.canvas.mpl_connect('key_press_event', onKeyPress)
       
    anm = animation.FuncAnimation(simFig, animate, init_func=initAnim, blit=False, interval=dt/1e3, repeat=False)
    anm.running = True
    
    simFig.tight_layout()
    
    plt.show()
    
else:   
    t = simulator.t
    
    animate.currRun = 1
    
    while True:
        simulator.step()
        
        t = simulator.t
        ksi = simulator.y
        
        x = ksi[0:dimState]
        y = sysOut(x)
        
        if isDynCtrl:
            u = ksi[-dimInput:]
        else:
            ctrlStat(y, t, sampleTime=dt)   # Updates ctrlStat.sampled_u
            u = ctrlStat.sampled_u
        
        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        v = ksi[3]
        omega = ksi[4]
        
        r = rcost(y, u)
        icost.val = icost(r, t)
        
        if isPrintSimStep:
            printSimStep(t, xCoord, yCoord, alpha, v, omega, icost.val, u)
            
        if isLogData:
            logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, icost.val, u)
        
        if t >= t1:  
            if isPrintSimStep:
                print('.....................................Run {run:2d} done.....................................'.format(run = animate.currRun))
                
            animate.currRun += 1
            
            if animate.currRun > Nruns:
                break
                
            if isLogData:
                dataFile = dataFiles[animate.currRun-1]
            
            # Reset simulator
            simulator.status = 'running'
            simulator.t = t0
            simulator.y = ksi0
            
            # Reset controller
            if isDynCtrl:
                ctrlDyn.ybuffer = repMat(y0, 2, 1)
                ctrlDyn.itime = t0
            else:
                ctrlStat.ctrlClock = t0
                ctrlStat.sampled_u = u0
            
            icost.val = 0      