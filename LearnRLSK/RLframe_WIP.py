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
import os
import pathlib
import warnings

# [EXPERIMENTAL] Use MATLAB's system identification toolbox instead of ssid and sippy
# Compatible MATLAB Runtime and system identification toolobx must be installed
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'~/MATLAB/RL/ENDICart',nargout=0)

#%% Initialization

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

class LearnRLSK:
    def __init__(self, 
                dimState = 5, 
                dimInput = 2, 
                dimOutput = 5, 
                dimDisturb = 2,
                m = 10,
                I = 1,
                t0 = 0,
                t1 = 100,
                Nruns = 1,
                atol = 1e-5,
                rtol = 1e-5,
                xMin = -10,
                xMax = 10,
                yMin = -10,
                yMax = 10,
                dt = 0.05,
                cutoff = 1,
                diffFiltOrd = 4,
                modEstPhase = 2,
                modEstBufferSize = 200,
                modelOrder = 5,
                probNoisePow = 8,
                modEstchecks = 0,
                Fman = -3,
                Nman = -1,
                Fmin = -5,
                Fmax = 5,
                Mmin = -1,
                Mmax = 1,
                Nactor = 6,
                rcostStruct = 1,
                Ncritic = 50,
                gamma = 1,
                criticStruct = 3,
                disturbOn = 0,
                isLogData = 1,
                isVisualization = 1,
                isPrintSimStep = 1,
                ctrlConstraintOn = 1,
                isDynCtrl = 0,
                ctrlStatMode = 5,
                ctrlDynMode = 0,
                isProbNoise = 1,
                isGlobOpt = 0):



        self.dimState = dimState
        self.dimInput = dimInput
        self.dimOutput = dimOutput
        self.dimDisturb = dimDisturb

        self.dimFullStateStat = dimState + dimDisturb
        self.dimFullStateDyn = dimState + dimDisturb + dimInput

        # System parameters
        self.m = m # [kg]
        self.I = I # [kg m^2]

        # Disturbance
        self.sigma_q = self.sigma_q_DEF = 1e-3 * np.ones(dimDisturb)
        self.mu_q = self.mu_q_DEF = np.zeros(dimDisturb)
        self.tau_q = self.tau_q_DEF = np.ones(dimDisturb)


        #------------------------------------simulation
        self.t0 = t0
        self.t1 = t1
        self.Nruns = Nruns

        self.x0 = np.zeros(dimState)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi/2

        self.u0 = np.zeros(dimInput)
        self.q0 = np.zeros(dimDisturb)

        # Solver
        self.atol = atol
        self.rtol = rtol

        # xy-plane
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax

        #------------------------------------digital elements
        # Digital elements sampling time
        self.dt = dt # [s], controller sampling time
        self.sampleFreq = 1/dt # [Hz]
        self.cutoff = cutoff # [Hz]

        # Digital differentiator filter order
        self.diffFiltOrd = diffFiltOrd

        #------------------------------------model estimator
        self.modEstPhase = modEstPhase # [s]
        self.modEstPeriod = 1*dt # [s]
        self.modEstBufferSize = modEstBufferSize

        self.modelOrder = modelOrder
        self.probNoisePow = probNoisePow

        # Model estimator stores models in a stack and recall the best of modEstchecks
        self.modEstchecks = modEstchecks

        #------------------------------------controller
        # u[0]: Pushing force F [N]
        # u[1]: Steering torque M [N m]

        # Manual control
        self.Fman = Fman
        self.Nman = Nman

        # Control constraints
        self.Fmin = Fmin
        self.Fmax = Fmax
        self.Mmin = Mmin
        self.Mmax = Mmax

        # Control horizon length
        self.Nactor = Nactor

        # Should be a multiple of dt
        self.predStepSize = 5*dt # [s]
        self.rcostStruct = rcostStruct

        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) 
        self.Ncritic = Ncritic

        # Discounting factor
        self.gamma = gamma

        # Critic is updated every criticPeriod seconds
        self.criticPeriod = 5*dt # [s]
        self.criticStruct = criticStruct

        #------------------------------------main switches
        self.isLogData = 1
        self.isVisualization = 1
        self.isPrintSimStep = 1

        self.disturbOn = 0
        self.ctrlConstraintOn = 1

        # Static or dynamic controller
        self.isDynCtrl = isDynCtrl
        self.ctrlStatMode = ctrlStatMode 
        self.ctrlDynMode = ctrlDynMode
        self.isProbNoise = isProbNoise
        self.isGlobOpt = isGlobOpt


        # main
        self.y0 = self.x0

        self.icost_val = 0
        self.icost_itime = t0

        self.xCoord0 = self.x0[0]
        self.yCoord0 = self.x0[1]
        self.alpha0 = self.x0[2]
        self.alphaDeg0 = self.alpha0/2/np.pi

        # Data logging init
        cwd = os.getcwd()
        datafolder = '/data'
        self.dataFolder_path = cwd + datafolder



        #------------------------------------RL elements
            
        if self.criticStruct == 1:
            self.dimCrit = ( ( dimOutput + dimInput ) + 1 ) * ( dimOutput + dimInput )/2 + (dimOutput + dimInput)  
            self.Wmin = -1e3*np.ones(dimCrit) 
            self.Wmax = 1e3*np.ones(dimCrit) 
        elif self.criticStruct == 2:
            self.dimCrit = ( ( dimOutput + dimInput ) + 1 ) * ( dimOutput + dimInput )/2
            self.Wmin = np.zeros(dimCrit) 
            self.Wmax = 1e3*np.ones(dimCrit)    
        elif self.criticStruct == 3:
            self.dimCrit = dimOutput + dimInput
            self.Wmin = np.zeros(dimCrit) 
            self.Wmax = 1e3*np.ones(dimCrit)    
        elif self.criticStruct == 4:
            self.dimCrit = dimOutput + dimOutput * dimInput + dimInput
            self.Wmin = -1e3*np.ones(dimCrit) 
            self.Wmax = 1e3*np.ones(dimCrit)
          
        # Critic weights
        self.Winit = rand(self.dimCrit)

        if isDynCtrl:
            pass
        else:
            self.ctrlStat_Wprev = Winit
            self.ctrlStat_criticClock = 0

        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        self.actorOptMethod = 'SLSQP'
        if self.actorOptMethod == 'trust-constr':
            self.criticOptOptions = {'maxiter': 200, 'disp': False} #'disp': True, 'verbose': 2}
        else:
            self.criticOptOptions = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7, 'fatol': 1e-7} # 'disp': True, 'verbose': 2} 

        # Clip critic buffer size
        self.Ncritic = np.min([Ncritic, modEstBufferSize-1])
            

        self.closedLoopStat_x0true = self.x0





    def create_datafile(self):
        if isLogData:
            # create data dir
            pathlib.Path(self.dataFolder_path).mkdir(parents=True, exist_ok=True) 

            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%Hh%Mm%Ss")
            dataFiles = [None] * self.Nruns
            for k in range(0, self.Nruns):
                dataFiles[k] = self.dataFolder_path + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
                with open(dataFiles[k], 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]'] )
            dataFile = dataFiles[0]

            return dataFile


    def train_this(self):
        dataFile = create_datafile()
        
        if isPrintSimStep:
            warnings.filterwarnings('ignore')

            



    #%% Service
    def toColVec(self, argin):
        if argin.ndim < 2:
            return np.reshape(argin, (argin.size, 1))
        elif argin.ndim ==2:
            if argin.shape[0] < argin.shape[1]:
                return argin.T
            else:
                return argin

    # To ensure 1D result
    def repMat(self, argin, n, m):
        return np.squeeze(repmat(argin, n, m))

    def pushVec(self, matrix, vec):
        return np.vstack([matrix[1:,:], vec])


    def printSimStep(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180      
        
        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]']  
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]  
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')   
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
        
        print(table)

    def logDataRow(self, dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
        with open(dataFile, 'a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow([t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]])

    #%% System
    def sysStateDyn(self, t, x, u, q):
        # x[0] -- x [m]
        # x[1] -- y [m]
        # x[2] -- alpha [rad]
        # x[3] -- v [m/s]
        # x[4] -- omega [rad/s]
        Dx = np.zeros(self.dimState)
        Dx[0] = x[3] * np.cos( x[2] )
        Dx[1] = x[3] * np.sin( x[2] )
        Dx[2] = x[4]
        
        if self.disturbOn:
            Dx[3] = 1/self.m * (u[0] + q[0])
            Dx[4] = 1/self.I * (u[1] + q[1])
        else:
            Dx[3] = 1/self.m * u[0]
            Dx[4] = 1/self.I * u[1]
            
        return Dx

    #%% Controller

    # Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)
    def zeta(self, xNI, theta):
        sigmaTilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        nablaF = np.zeros(3)
        
        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigmaTilde**3
        
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigmaTilde**3
        
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigmaTilde**3  

        return nablaF


    # Stabilizing controller for NI-part
    def kappa(self, xNI, theta): 
        kappaVal = np.zeros(2)
        
        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])
                         
        zetaVal = zeta(xNI, theta)
        
        kappaVal[0] = - np.abs( np.dot( zetaVal, G[:,0] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,0] ) )
        kappaVal[1] = - np.abs( np.dot( zetaVal, G[:,1] ) )**(1/3) * np.sign( np.dot( zetaVal, G[:,1] ) )
        
        return kappaVal

    # Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned above
    def Fc(self, xNI, eta, theta):
        sigmaTilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigmaTilde
        
        z = eta - kappa(xNI, theta)
        
        return F + 1/2 * np.dot(z, z)

    def thetaMinimizer(self, xNI, eta):
        thetaInit = 0
        
        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)
        
        options = {'maxiter': 50, 'disp': False}
        
        thetaVal = minimize(lambda theta: Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x
        
        return thetaVal


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
 
    def NH2CartCtrl(self, xNI, eta, uNI): 
        uCart = np.zeros(2)
        
        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
        uCart[1] = self.I * uNI[0]
        
        return uCart


    # Simulate output response of discrete-time state-space model
    def dssSim(self, A, B, C, D, uSqn, x0, y0):
        if uSqn.ndim == 1:
            return y0, x0
        else:
            ySqn = np.zeros([uSqn.shape[0], C.shape[0]])
            xSqn = np.zeros([uSqn.shape[0], A.shape[0]])
            x = x0
            ySqn[0, :] = y0
            xSqn[0, :] = x0 
            for k in range(1, uSqn.shape[0]):
                x = A @ x + B @ uSqn[k-1, :]
                xSqn[k, :] = x
                ySqn[k, :] = C @ x + D @ uSqn[k-1, :]
                
            return ySqn, xSqn


    # Feature vector of critic
    def Phi(self, y, u):
        chi = np.concatenate([y, u])
        
        if self.criticStruct == 3:
            return chi * chi    
        elif self.criticStruct == 4:
            return np.concatenate([ y**2, np.kron(y, u), u**2 ])


    # Running cost
    def rcost(self, y, u):
        chi = np.concatenate([y, u])
        
        r = 0
        
        if self.rcostStruct == 1:
            r = chi @ self.R1 @ chi
        elif self.rcostStruct == 2:
            r = chi**2 @ self.R2 @ chi**2 + chi @ self.R1 @ chi
        
        return r
        
    # Integrated cost
    def icost(self, r, t):
        icost_val += r*(t - self.icost_itime)
        self.icost_itime = t
        return icost_val   

    def criticCost(self, W, U, Y, Wprev):
        Jc = 0
        
        for k in range(self.dimCrit, 0, -1):
            yPrev = Y[k-1, :]
            yNext = Y[k, :]
            uPrev = U[k-1, :]
            uNext = U[k, :]
            
            # Temporal difference
            e = W @ self.Phi( yPrev, uPrev ) - self.gamma * Wprev @ self.Phi( yNext, uNext ) - self.rcost(yPrev, uPrev)
            
            Jc += 1/2 * e**2
            
        return Jc

        
    def critic(self, Wprev, Winit, U, Y):              
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)

        W = minimize(lambda W: criticCost(W, U, Y, Wprev), Winit, method=self.criticOptMethod, tol=1e-7, bounds=bnds, options=self.criticOptOptions).x

        
        return W

    def actorCost(self, U, y, N, W, delta, mode):
        myU = np.reshape(U, [N, self.dimInput])
        
        Y = np.zeros([N, self.dimOutput])
        
        # System output prediction
        if (mode==1) or (mode==3) or (mode==5):    # Via true model
            Y[0, :] = y
            x = self.closedLoopStat_x0true
            for k in range(1, self.Nactor):
                x = x + delta * sysStateDyn([], x, myU[k-1, :], [])  # Euler scheme
                Y[k, :] = sysOut(x)
                # Y[k, :] = Y[k-1, :] + dt * sysStateDyn([], Y[k-1, :], myU[k-1, :], [])  # Euler scheme
        elif (mode==2) or (mode==4) or (mode==6):    # Via estimated model
            myU_upsampled = myU.repeat(int(delta/self.dt), axis=0)
            Yupsampled, _ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, myU_upsampled, ctrlStat.x0est, y)
            Y = Yupsampled[::int(delta/self.dt)]
        
        J = 0         
        if (mode==1) or (mode==2):     # MPC
            for k in range(N):
                J += self.gamma**k * rcost(Y[k, :], myU[k, :])
        elif (mode==3) or (mode==4):     # RL: Q-learning with Ncritic roll-outs of running cost
             for k in range(N-1):
                J += gamma**k * rcost(Y[k, :], myU[k, :])
             J += W @ Phi( Y[-1, :], myU[-1, :] )
        elif (mode==5) or (mode==6):     # RL: (normalized) stacked Q-learning
             for k in range(N):
                Q = W @ Phi( Y[k, :], myU[k, :] )
                J += 1/N * Q
       
            
        return J

    # Optimal controller a.k.a. actor in RL terminology
    def actor(self, y, Uinit, N, W, A, B, C, D, x0, delta, mode):
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
        