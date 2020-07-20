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


# %% Import packages

import csv
import os
import pathlib
import warnings
from collections import namedtuple
from datetime import datetime

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
# System identification packages
import sippy
from mpldatacursor import datacursor
from numpy.random import rand
from numpy.random import randn
from scipy import signal
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from svgpath2mpl import parse_path
from tabulate import tabulate


# !pip install tabulate <-- to install this

# [EXPERIMENTAL] Use MATLAB's system identification toolbox instead of ssid and sippy
# Compatible MATLAB Runtime and system identification toolobx must be installed
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'~/MATLAB/RL/ENDICart',nargout=0)

# %% Initialization

class ZOH:
    def __init__(self, initTime=0, initVal=0, sampleTime=1):
        self.timeStep = initTime
        self.sampleTime = sampleTime
        self.currVal = initVal

    def hold(self, signalVal, t):
        timeInSample = t - self.timeStep
        if timeInSample >= self.sampleTime:  # New sample
            self.timeStep = t
            self.currVal = signalVal

        return self.currVal


# Real-time digital filter
class dfilter:
    def __init__(self, filterNum, filterDen, bufferSize=16, initTime=0, initVal=0, sampleTime=1):
        self.Num = filterNum
        self.Den = filterDen
        self.zi = repMatCustom(signal.lfilter_zi(filterNum, filterDen), 1, initVal)

        self.timeStep = initTime
        self.sampleTime = sampleTime
        self.buffer = repMatCustom(initVal, 1, bufferSize)

    def filt(self, signalVal, t=None):
        # Sample only if time is specified
        if t is not None:
            timeInSample = t - self.timeStep
            if timeInSample >= self.sampleTime:  # New sample
                self.timeStep = t
                self.buffer = pushVec(self.buffer, signalVal)
        else:
            self.buffer = pushVec(self.buffer, signalVal)

        bufferFiltered = np.zeros(self.buffer.shape)

        for k in range(0, signalVal.size):
            bufferFiltered[k, :], self.zi[k] = signal.lfilter(self.Num, self.Den, self.buffer[k, :], zi=self.zi[k, :])
        return bufferFiltered[-1, :]


class pltMarker:
    def __init__(self, angle=None, pathString=None):
        self.angle = angle or []
        self.pathString = pathString or []
        self.path = parse_path(self.pathString)
        self.path.vertices -= self.path.vertices.mean(axis=0)
        self.marker = mpl.markers.MarkerStyle(marker=self.path)
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(angle - self.angle)
        self.angle = angle


# To ensure 1D result
def repMatCustom(argin, n, m):
    return np.squeeze(repMatCustom(argin, n, m))


def pushVec(matrix, vec):
    return np.vstack([matrix[1:, :], vec])


class LearnRLSK:
    def __init__(self,
                 dimState=5,
                 dimInput=2,
                 dimOutput=5,
                 dimDisturb=2,
                 m=10,
                 I=1,
                 t0=0,
                 t1=100,
                 Nruns=1,
                 atol=1e-5,
                 rtol=1e-5,
                 xMin=-10,
                 xMax=10,
                 yMin=-10,
                 yMax=10,
                 dt=0.05,
                 cutoff=1,
                 diffFiltOrd=4,
                 modEstPhase=2,
                 modEstBufferSize=200,
                 modelOrder=5,
                 probNoisePow=8,
                 modEstchecks=0,
                 Fman=-3,
                 Nman=-1,
                 Fmin=-5,
                 Fmax=5,
                 Mmin=-1,
                 Mmax=1,
                 Nactor=6,
                 rcostStruct=1,
                 Ncritic=50,
                 gamma=1,
                 criticStruct=3,
                 disturbOn=0,
                 isLogData=1,
                 isVisualization=1,
                 isPrintSimStep=1,
                 ctrlConstraintOn=1,
                 isDynCtrl=0,
                 ctrlStatMode=5,
                 ctrlDynMode=0,
                 isProbNoise=1,
                 isGlobOpt=0):

        self.isPrintSimStep = isPrintSimStep
        self.Nactor = Nactor
        self.isVisualization = isVisualization
        self.isLogData = isLogData
        self.disturbOn = disturbOn
        self.ctrlConstraintOn = ctrlConstraintOn
        self.dimState = dimState
        self.dimInput = dimInput
        self.dimOutput = dimOutput
        self.dimDisturb = dimDisturb

        self.dimFullStateStat = dimState + dimDisturb
        self.dimFullStateDyn = dimState + dimDisturb + dimInput

        # System parameters
        self.m = m  # [kg]
        self.I = I  # [kg m^2]

        # Disturbance
        self.sigma_q = self.sigma_q_DEF = 1e-3 * np.ones(dimDisturb)
        self.mu_q = self.mu_q_DEF = np.zeros(dimDisturb)
        self.tau_q = self.tau_q_DEF = np.ones(dimDisturb)

        # ------------------------------------simulation
        self.t0 = t0
        self.t1 = t1
        self.Nruns = Nruns

        self.x0 = np.zeros(dimState)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2

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

        # ------------------------------------digital elements
        # Digital elements sampling time
        self.dt = dt  # [s], controller sampling time
        self.sampleFreq = 1 / dt  # [Hz]
        self.cutoff = cutoff  # [Hz]

        # Digital differentiator filter order
        self.diffFiltOrd = diffFiltOrd

        # ------------------------------------model estimator
        self.modEstPhase = modEstPhase  # [s]
        self.modEstPeriod = 1 * dt  # [s]
        self.modEstBufferSize = modEstBufferSize

        self.modelOrder = modelOrder
        self.probNoisePow = probNoisePow

        # Model estimator stores models in a stack and recall the best of modEstchecks
        self.modEstchecks = modEstchecks

        # ------------------------------------controller
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
        self.Nactor = self.Nactor

        # Should be a multiple of dt
        self.predStepSize = 5 * dt  # [s]
        self.rcostStruct = rcostStruct

        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.Ncritic = Ncritic

        # Discounting factor
        self.gamma = gamma

        # Critic is updated every criticPeriod seconds
        self.criticPeriod = 5 * dt  # [s]
        self.criticStruct = criticStruct

        # ------------------------------------main switches
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
        self.alphaDeg0 = self.alpha0 / 2 / np.pi

        self.closedLoopStat_x0true = self.x0

        # Critic weights
        self.Winit = rand(self.dimCrit)

        # controller, model estimator
        if isDynCtrl:
            self.ctrlDyn_ybuffer = repMatCustom(self.y0, 1, 2)
            self.ctrlDyn_itime = t0
            self.ksi0 = np.concatenate([x0, q0, u0])
            self.simulator = sp.integrate.RK45(lambda t, ksi: closedLoopDyn(t, ksi, sigma_q, mu_q, tau_q, dt), t0, ksi0,
                                               t1, first_step=1e-6, atol=atol, rtol=rtol)
        else:
            self.ctrlStat_Wprev = Winit
            self.ctrlStat_CriticClock = 0
            self.ctrlStat_estClock = t0
            self.ctrlStat_modEst_ubuffer = np.zeros([modEstBufferSize, dimInput])
            self.ctrlStat_modEst_ybuffer = np.zeros([modEstBufferSize, dimOutput])
            self.ctrlDyn_ybuffer = repMatCustom(self.y0, 1, 2)
            self.ctrlDyn_itime = t0

            # Initial model estimate is subject to tuning
            self.ctrlStat_A = np.zeros([modelOrder, modelOrder])
            self.ctrlStat_B = np.zeros([modelOrder, dimInput])
            self.ctrlStat_C = np.zeros([dimOutput, modelOrder])
            self.ctrlStat_D = np.zeros([dimOutput, dimInput])
            self.ctrlStat_x0est = np.zeros(modelOrder)
            self.ctrlStat_CtrlClock = t0
            self.ctrlStat_sampled_u = u0
            self.ksi0 = np.concatenate([x0, q0])
            self.simulator = sp.integrate.RK45(lambda t, ksi: closedLoopStat(t, ksi, sigma_q, mu_q, tau_q, dt),
                                               t0, ksi0, t1, first_step=1e-6, atol=atol, rtol=rtol)

        self.ctrlStat_modEstAstack = [self.ctrlStat_A] * modEstchecks
        self.ctrlStat_modEstBstack = [self.ctrlStat_B] * modEstchecks
        self.ctrlStat_modEstCstack = [self.ctrlStat_C] * modEstchecks
        self.ctrlStat_modEstDstack = [self.ctrlStat_D] * modEstchecks

        # Stack constraints    
        self.uMin = np.array([Fmin, Mmin])
        self.uMax = np.array([Fmax, Mmax])
        self.uMin = repMatCustom(self.uMin, 1, self.Nactor)
        self.Umax = repMatCustom(self.uMax, 1, self.Nactor)

        # Optimization method of critic    
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell
        self.actorOptMethod = 'SLSQP'
        self.criticOptMethod = 'SLSQP'
        if self.actorOptMethod == 'trust-constr':
            self.criticOptOptions = {'maxiter': 200, 'disp': False}
            self.actorOptOptions = {'maxiter': 300, 'disp': False}  # 'disp': True, 'verbose': 2}
        else:
            self.criticOptOptions = {'maxiter': 200, 'maxfev': 1500, 'disp': False, 'adaptive': True, 'xatol': 1e-7,
                                     'fatol': 1e-7}  # 'disp': True, 'verbose': 2}
            self.actorOptOptions = {'maxiter': 300, 'maxfev': 5000, 'disp': False, 'adaptive': True, 'xatol': 1e-7,
                                    'fatol': 1e-7}  # 'disp': True, 'verbose': 2}

        self.cPID = namedtuple('PID', ['P', 'I', 'D'])
        self.PID = cPID(P=[1, 1], I=[.5, .5], D=[0, 0])
        # Clip critic buffer size
        self.Ncritic = np.min([Ncritic, modEstBufferSize - 1])

        # ------------------------------------RL elements

        if self.criticStruct == 1:
            self.dimCrit = ((dimOutput + dimInput) + 1) * (dimOutput + dimInput) / 2 + (dimOutput + dimInput)
            self.Wmin = -1e3 * np.ones(self.dimCrit)
            self.Wmax = 1e3 * np.ones(self.dimCrit)
        elif self.criticStruct == 2:
            self.dimCrit = ((dimOutput + dimInput) + 1) * (dimOutput + dimInput) / 2
            self.Wmin = np.zeros(self.dimCrit)
            self.Wmax = 1e3 * np.ones(self.dimCrit)
        elif self.criticStruct == 3:
            self.dimCrit = dimOutput + dimInput
            self.Wmin = np.zeros(self.dimCrit)
            self.Wmax = 1e3 * np.ones(self.dimCrit)
        elif self.criticStruct == 4:
            self.dimCrit = dimOutput + dimOutput * dimInput + dimInput
            self.Wmin = -1e3 * np.ones(self.dimCrit)
            self.Wmax = 1e3 * np.ones(self.dimCrit)

        # ------------------------------------digital elements

        # Differentiator filters
        self.diffFilterNum = signal.remez(self.diffFiltOrd + 1, [0, cutoff], [1], type='differentiator')
        diffFilterDen = np.array([1.0])

        self.cdiffFilters = namedtuple('diffFilter', ['y', 'Dy'])
        self.diffFilters = cdiffFilters(
            Dy=dfilter(self.diffFilterNum, diffFilterDen, bufferSize=4, initTime=t0, initVal=self.y0, sampleTime=dt),
            y=dfilter(self.diffFilterNum, diffFilterDen, bufferSize=4, initTime=t0, initVal=self.y0, sampleTime=dt))

        # Zero-order holds
        self.cZOHs = namedtuple('ZOH', ['u', 'y'])
        self.ZOHs = cZOHs(u=ZOH(initTime=t0, initVal=u0, sampleTime=dt),
                          y=ZOH(initTime=t0, initVal=self.y0, sampleTime=dt))

    # %% Service
    def toColVec(self, argin):
        if argin.ndim < 2:
            return np.reshape(argin, (argin.size, 1))
        elif argin.ndim == 2:
            if argin.shape[0] < argin.shape[1]:
                return argin.T
            else:
                return argin

    def printSimStep(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180      

        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]',
                     'M [N m]']
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')

        print(table)

    def logDataRow(self, dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
        with open(dataFile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]])

    # %% System
    def sysStateDyn(self, t, x, u, q):
        # x[0] -- x [m]
        # x[1] -- y [m]
        # x[2] -- alpha [rad]
        # x[3] -- v [m/s]
        # x[4] -- omega [rad/s]
        Dx = np.zeros(self.dimState)
        Dx[0] = x[3] * np.cos(x[2])
        Dx[1] = x[3] * np.sin(x[2])
        Dx[2] = x[4]

        if self.disturbOn:
            Dx[3] = 1 / self.m * (u[0] + q[0])
            Dx[4] = 1 / self.I * (u[1] + q[1])
        else:
            Dx[3] = 1 / self.m * u[0]
            Dx[4] = 1 / self.I * u[1]

        return Dx

    # %% Controller

    # Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)
    def zeta(self, xNI, theta):
        sigmaTilde = xNI[0] * np.cos(theta) + xNI[1] * np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4 * xNI[0] ** 3 - 2 * np.abs(xNI[2]) ** 3 * np.cos(theta) / sigmaTilde ** 3

        nablaF[1] = 4 * xNI[1] ** 3 - 2 * np.abs(xNI[2]) ** 3 * np.sin(theta) / sigmaTilde ** 3

        nablaF[2] = (3 * xNI[0] * np.cos(theta) + 3 * xNI[1] * np.sin(theta) + 2 * np.sqrt(np.abs(xNI[2]))) * xNI[
            2] ** 2 * np.sign(xNI[2]) / sigmaTilde ** 3

        return nablaF

    # Stabilizing controller for NI-part
    def kappa(self, xNI, theta):
        kappaVal = np.zeros(2)

        G = np.zeros([3, 2])
        G[:, 0] = np.array([1, 0, xNI[1]])
        G[:, 1] = np.array([0, 1, -xNI[0]])

        zetaVal = zeta(xNI, theta)

        kappaVal[0] = - np.abs(np.dot(zetaVal, G[:, 0])) ** (1 / 3) * np.sign(np.dot(zetaVal, G[:, 0]))
        kappaVal[1] = - np.abs(np.dot(zetaVal, G[:, 1])) ** (1 / 3) * np.sign(np.dot(zetaVal, G[:, 1]))

        return kappaVal

    # Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned above
    def Fc(self, xNI, eta, theta):
        sigmaTilde = xNI[0] * np.cos(theta) + xNI[1] * np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        F = xNI[0] ** 4 + xNI[1] ** 4 + np.abs(xNI[2]) ** 3 / sigmaTilde

        z = eta - kappa(xNI, theta)

        return F + 1 / 2 * np.dot(z, z)

    def thetaMinimizer(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        thetaVal = minimize(lambda theta: Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds,
                            options=options).x

        return thetaVal

    def Cart2NH(self, CartCoords):
        xNI = np.zeros(3)
        eta = np.zeros(2)

        xc = CartCoords[0]
        yc = CartCoords[1]
        alpha = CartCoords[2]
        v = CartCoords[3]
        omega = CartCoords[4]

        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * (yc * np.cos(alpha) - xc * np.sin(alpha)) - alpha * (xc * np.cos(alpha) + yc * np.sin(alpha))

        eta[0] = omega
        eta[1] = (yc * np.cos(alpha) - xc * np.sin(alpha)) * omega + v

        return [xNI, eta]

    def NH2CartCtrl(self, xNI, eta, uNI):
        uCart = np.zeros(2)

        uCart[0] = self.m * (uNI[1] + xNI[1] * eta[0] ** 2 + 1 / 2 * (xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2]))
        uCart[1] = self.I * uNI[0]

        return uCart

    # Simulate output response of discrete-time state-space model
    def dssSim(self, A, B, C, D, uSqn, x0, self.y0):
        if uSqn.ndim == 1:
            return self.y0, x0
        else:
            ySqn = np.zeros([uSqn.shape[0], C.shape[0]])
            xSqn = np.zeros([uSqn.shape[0], A.shape[0]])
            x = x0
            ySqn[0, :] = self.y0
            xSqn[0, :] = x0
            for k in range(1, uSqn.shape[0]):
                x = A @ x + B @ uSqn[k - 1, :]
                xSqn[k, :] = x
                ySqn[k, :] = C @ x + D @ uSqn[k - 1, :]

            return ySqn, xSqn

    # Feature vector of critic
    def Phi(self, y, u):
        chi = np.concatenate([y, u])

        if self.criticStruct == 3:
            return chi * chi
        elif self.criticStruct == 4:
            return np.concatenate([y ** 2, np.kron(y, u), u ** 2])

    # Running cost
    def rcost(self, y, u):
        chi = np.concatenate([y, u])

        r = 0

        if self.rcostStruct == 1:
            r = chi @ self.R1 @ chi
        elif self.rcostStruct == 2:
            r = chi ** 2 @ self.R2 @ chi ** 2 + chi @ self.R1 @ chi

        return r

    # Integrated cost
    def icost(self, r, t):
        icost_val += r * (t - self.icost_itime)
        self.icost_itime = t
        return icost_val

    def criticCost(self, W, U, Y, Wprev):
        Jc = 0

        for k in range(self.dimCrit, 0, -1):
            yPrev = Y[k - 1, :]
            yNext = Y[k, :]
            uPrev = U[k - 1, :]
            uNext = U[k, :]

            # Temporal difference
            e = W @ self.Phi(yPrev, uPrev) - self.gamma * Wprev @ self.Phi(yNext, uNext) - self.rcost(yPrev, uPrev)

            Jc += 1 / 2 * e ** 2

        return Jc

    def critic(self, Wprev, Winit, U, Y):
        bnds = sp.optimize.Bounds(self.Wmin, self.Wmax, keep_feasible=True)

        W = minimize(lambda W: criticCost(W, U, Y, Wprev), Winit, method=self.criticOptMethod, tol=1e-7, bounds=bnds,
                     options=self.criticOptOptions).x

        return W

    def actorCost(self, U, y, N, W, delta, mode):
        myU = np.reshape(U, [N, self.dimInput])

        Y = np.zeros([N, self.dimOutput])

        # System output prediction
        if (mode == 1) or (mode == 3) or (mode == 5):  # Via true model
            Y[0, :] = y
            x = self.closedLoopStat_x0true
            for k in range(1, self.Nactor):
                x = x + delta * sysStateDyn([], x, myU[k - 1, :], [])  # Euler scheme
                Y[k, :] = self.sysOut(x)
                # Y[k, :] = Y[k-1, :] + dt * sysStateDyn([], Y[k-1, :], myU[k-1, :], [])  # Euler scheme
        elif (mode == 2) or (mode == 4) or (mode == 6):  # Via estimated model
            myU_upsampled = myU.repeat(int(delta / self.dt), axis=0)
            Yupsampled, _ = self.dssSim(self.ctrlStat_A, self.ctrlStat_B, self.ctrlStat_C, self.ctrlStat_D,
                                        myU_upsampled, self.ctrlStat_x0est, y)
            Y = Yupsampled[::int(delta / self.dt)]

        J = 0
        if (mode == 1) or (mode == 2):  # MPC
            for k in range(N):
                J += self.gamma ** k * self.rcost(Y[k, :], myU[k, :])
        elif (mode == 3) or (mode == 4):  # RL: Q-learning with Ncritic roll-outs of running cost
            for k in range(N - 1):
                J += self.gamma ** k * self.rcost(Y[k, :], myU[k, :])
            J += W @ self.Phi(Y[-1, :], myU[-1, :])
        elif (mode == 5) or (mode == 6):  # RL: (normalized) stacked Q-learning
            for k in range(N):
                Q = W @ self.Phi(Y[k, :], myU[k, :])
                J += 1 / N * Q

        return J

    # Optimal controller a.k.a. actor in RL terminology
    def actor(self, y, Uinit, N, W, A, B, C, D, x0, delta, mode):
        myUinit = np.reshape(Uinit, [N * self.dimInput, ])

        bnds = sp.optimize.Bounds(self.uMin, self.Umax, keep_feasible=True)

        try:
            if self.isGlobOpt:
                minimizer_kwargs = {'method': self.actorOptMethod, 'bounds': bnds, 'tol': 1e-7,
                                    'options': self.actorOptOptions}
                U = basinhopping(lambda U: self.actorCost(U, y, N, W, delta, mode), myUinit,
                                 minimizer_kwargs=minimizer_kwargs, niter=10).x
            else:
                U = minimize(lambda U: self.actorCost(U, y, N, W, delta, mode), myUinit, method=self.actorOptMethod,
                             tol=1e-7, bounds=bnds, options=self.actorOptOptions).x
        except ValueError:
            print('Actor''s optimizer failed. Returning default action')
            U = myUinit

        R = '\033[31m'
        Bl = '\033[30m'
        myU = np.reshape(U, [N, dimInput])
        myU_upsampled = myU.repeat(int(delta / dt), axis=0)
        Yupsampled, _ = self.dssSim(self.ctrlStat_A, self.ctrlStat_B, self.ctrlStat_C, self.ctrlStat_D, myU_upsampled,
                                    self.ctrlStat_x0est, y)
        Y = Yupsampled[::int(delta / self.dt)]
        Yt = np.zeros([N, self.dimOutput])
        Yt[0, :] = y
        x = self.closedLoopStat_x0true
        for k in range(1, self.Nactor):
            x = x + delta * sysStateDyn([], x, myU[k - 1, :], [])  # Euler scheme
            Yt[k, :] = self.sysOut(x)
        headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
        dataRow = []
        for k in range(dimOutput):
            dataRow.append(np.mean(Y[:, k] - Yt[:, k]))
        rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
        print(R + table + Bl)
        # /DEBUG ==================================================================     

        return U[:self.dimInput]

    def ctrlStat(self, y, t):
        # In ctrlStat, a ZOH is built-in
        timeInSample = t - self.ctrlStat_CtrlClock

        if timeInSample >= self.dt:  # New sample
            # Update controller's internal clock
            self.ctrlStat_CtrlClock = t

            # ------------------------------------model update
            # Update buffers when using RL or requiring estimated model
            if self.ctrlStatMode in (2, 3, 4, 5, 6):
                timeInEstPeriod = t - self.ctrlStat_estClock

                self.ctrlStat_modEst_ubuffer = self.pushVec(self.ctrlStat_modEst_ubuffer, self.ctrlStat_sampled_u)
                self.ctrlStat_modEst_ubuffer = self.pushVec(self.ctrlStat_modEst_ybuffer, y)

                # Estimate model if required by self.ctrlStatMode
                if (timeInEstPeriod >= self.modEstPeriod) and (self.ctrlStatMode in (2, 4, 6)):
                    # Update model estimator's internal clock
                    self.ctrlStat_estClock = t

                    try:

                        SSest = sippy.system_identification(
                            self.ctrlStat_modEst_ybuffer,
                            self.ctrlStat_modEst_ubuffer,
                            id_method='N4SID',
                            SS_fixed_order=self.modelOrder,
                            SS_D_required=False,
                            SS_A_stability=False,
                            SS_PK_B_reval=False,
                            tsample=dt)

                        self.ctrlStat_A, self.ctrlStat_B, self.ctrlStat_C, self.ctrlStat_D = SSest.A, SSest.B, SSest.C, SSest.D


                    except:
                        print('Model estimation problem')
                        self.ctrlStat_A = np.zeros([modelOrder, modelOrder])
                        self.ctrlStat_B = np.zeros([modelOrder, dimInput])
                        self.ctrlStat_C = np.zeros([dimOutput, modelOrder])
                        self.ctrlStat_D = np.zeros([dimOutput, dimInput])

                    # ---model checks
                    if self.modEstchecks > 0:
                        # Update estimated model parameter stacks
                        self.ctrlStat_modEstAstack.pop(0)
                        self.ctrlStat_modEstAstack.append(self.ctrlStat_A)

                        self.ctrlStat_modEstBstack.pop(0)
                        self.ctrlStat_modEstBstack.append(self.ctrlStat_B)

                        self.ctrlStat_modEstCstack.pop(0)
                        self.ctrlStat_modEstCstack.append(self.ctrlStat_C)

                        self.ctrlStat_modEstDstack.pop(0)
                        self.ctrlStat_modEstDstack.append(self.ctrlStat_D)

                        # Perform check of stack of models and pick the best
                        totAbsErrCurr = 1e8
                        for k in range(self.modEstchecks):
                            A, B, C, D = self.ctrlStat_modEstAstack[k], self.ctrlStat_modEstBstack[k], \
                                         self.ctrlStat_modEstCstack[k], self.ctrlStat_modEstDstack[k]
                            x0est, _, _, _ = np.linalg.lstsq(C, y)
                            Yest, _ = self.dssSim(A, B, C, D, self.ctrlStat_modEst_ubuffer, x0est, y)
                            meanErr = np.mean(Yest - self.ctrlStat_modEst_ybuffer, axis=0)

                            totAbsErr = np.sum(np.abs(meanErr))
                            if totAbsErr <= totAbsErrCurr:
                                totAbsErrCurr = totAbsErr
                                self.ctrlStat_A, self.ctrlStat_B, self.ctrlStat_C, self.ctrlStat_D = A, B, C, D

            # Update initial state estimate        
            self.ctrlStat_x0est, _, _, _ = np.linalg.lstsq(self.ctrlStat_C, y)

            if t >= self.modEstPhase:
                # Drop probing noise
                self.isProbNoise = 0

                # ------------------------------------control: manual
            if self.ctrlStatMode == 0:
                self.ctrlStat_sampled_u[0] = Fman
                self.ctrlStat_sampled_u[1] = Nman

            # ------------------------------------control: nominal    
            elif self.ctrlStatMode == 10:

                kNom = 50

                # This controller needs full-state measurement
                xNI, eta = Cart2NH(self.closedLoopStat_x0true)
                thetaStar = thetaMinimizer(xNI, eta)
                kappaVal = kappa(xNI, thetaStar)
                z = eta - kappaVal
                uNI = - kNom * z
                self.ctrlStat_sampled_u = NH2CartCtrl(xNI, eta, uNI)

            # ------------------------------------control: MPC    
            elif self.ctrlStatMode in (1, 2):
                Uinit = repMatCustom(self.uMin / 10, 1, self.Nactor)

                # Apply control when model estimation phase is over
                if isProbNoise and (self.ctrlStatMode == 2):
                    self.ctrlStat_sampled_u = self.probNoisePow * (rand(self.dimInput) - 0.5)
                elif not self.isProbNoise and (self.ctrlStatMode == 2):
                    self.ctrlStat_sampled_u = actor(y, Uinit, self.Nactor, [], self.ctrlStat_A, self.ctrlStat_B,
                                                    self.ctrlStat_C, self.ctrlStat_D, self.ctrlStat_x0est, predStepSize,
                                                    self.ctrlStatMode)

                elif (self.ctrlStatMode == 1):
                    self.ctrlStat_sampled_u = actor(y, Uinit, self.Nactor, [], self.ctrlStat_A, self.ctrlStat_B,
                                                    self.ctrlStat_C, self.ctrlStat_D, self.ctrlStat_x0est, predStepSize,
                                                    self.ctrlStatMode)

            # ------------------------------------control: RL
            elif self.ctrlStatMode in (3, 4, 5, 6):
                # Critic
                timeInCriticPeriod = t - self.ctrlStat_CriticClock
                if timeInCriticPeriod >= criticPeriod:
                    W = critic(self.ctrlStat_Wprev, Winit, self.ctrlStat_modEst_ubuffer[-Ncritic:, :],
                               self.ctrlStat_modEst_ybuffer[-Ncritic:, :])
                    self.ctrlStat_Wprev = W
                    # Update critic's internal clock
                    self.ctrlStat_CriticClock = t
                else:
                    W = self.ctrlStat_Wprev

                # Actor. Apply control when model estimation phase is over
                if isProbNoise and (self.ctrlStatMode in (4, 6)):
                    self.ctrlStat_sampled_u = probNoisePow * (rand(dimInput) - 0.5)
                elif not isProbNoise and (self.ctrlStatMode in (4, 6)):
                    Uinit = repMatCustom(self.uMin / 10, self.Nactor, 1)
                    self.ctrlStat_sampled_u = actor(y, Uinit, self.Nactor, W, self.ctrlStat_A, self.ctrlStat_B,
                                                    self.ctrlStat_C, self.ctrlStat_D, self.ctrlStat_x0est, predStepSize,
                                                    self.ctrlStatMode)

                elif self.ctrlStatMode in (3, 5):
                    Uinit = repMatCustom(self.uMin / 10, self.Nactor, 1)
                    self.ctrlStat_sampled_u = actor(y, Uinit, self.Nactor, W, self.ctrlStat_A, self.ctrlStat_B,
                                                    self.ctrlStat_C, self.ctrlStat_D, self.ctrlStat_x0est, predStepSize,
                                                    self.ctrlStatMode)

    # ===> ToDo: buffer update to move outside the simulator <===
    def ctrlDyn(self, t, u, y):
        self.ctrlDyn_ybuffer = pushVec(self.ctrlDyn_ybuffer, y)

        Du = np.zeros(dimInput)
        if self.ctrlDynMode == 0:

            # 1st difference
            if t - self.ctrlDyn_itime > 0:
                Dy = (self.ctrlDyn_ybuffer[1, :] - self.ctrlDyn_ybuffer[0, :]) / (t - self.ctrlDyn_itime)
            else:
                Dy = y

            trajNrm = la.norm(y[:2])
            DtrajNrm = np.dot(1 / trajNrm * np.array([y[0], y[1]]), Dy[:2])
            alpha = y[2]
            Dalpha = Dy[2]

            Du[0] = - self.PID.P[0] * DtrajNrm - self.PID.I[0] * trajNrm
            Du[1] = - self.PID.P[1] * Dalpha - self.PID.I[1] * alpha

        self.ctrlDyn_itime = t

        return Du

    # %% Disturbance dynamics

    # Simple 1st order filter of white Gaussian noise
    def disturbDyn(self, t, q):
        Dq = np.zeros(self.dimDisturb)
        for k in range(0, self.dimDisturb):
            Dq[k] = - self.tau_q_DEF[k] * (q[k] + self.sigma_q_DEF[k] * randn() + self.mu_q_DEF[k])
        return Dq

    # %% Closed loop

    def closedLoopStat(self, t, ksi):
        DfullState = np.zeros(self.dimFullStateStat)

        x = ksi[0:self.dimState]
        q = ksi[self.dimState:]

        # Get the control action
        u = self.ctrlStat_sampled_u

        if self.ctrlConstraintOn:
            u[0] = np.clip(u[0], self.Fmin, self.Fmax)
            u[1] = np.clip(u[1], self.Mmin, self.Mmax)

        DfullState[0:dimState] = self.sysStateDyn(t, x, u, q)
        DfullState[dimState:] = self.disturbDyn(t, q)

        # Track system's state for some controllers
        self.closedLoopStat_x0true = x

        return DfullState

    def closedLoopDyn(self, t, ksi):

        DfullState = np.zeros(dimFullStateDyn)

        x = ksi[0:self.dimState]
        q = ksi[self.dimState:self.dimState + self.dimDisturb]
        u = ksi[-self.dimInput:]

        u = self.ZOHs.u.hold(u, t)

        y = self.sysOut(x)

        if ctrlConstraintOn:
            u[0] = np.clip(u[0], Fmin, Fmax)
            u[1] = np.clip(u[1], Mmin, Mmax)

        DfullState[0:self.dimState] = self.sysStateDyn(t, x, u, q)
        DfullState[self.dimState:self.dimState + self.dimDisturb] = self.disturbDyn(t, q)
        DfullState[-self.dimInput:] = self.ctrlDyn(t, u, y)

        return DfullState

    # %% Cost 
    def updateLine(self, line, newX, newY):
        line.set_xdata(np.append(line.get_xdata(), newX))
        line.set_ydata(np.append(line.get_ydata(), newY))

    def resetLine(self, line):
        line.set_data([], [])

    def updateScatter(self, scatter, newX, newY):
        scatter.set_offsets(np.vstack([scatter.get_offsets().data, np.c_[newX, newY]]))

    def updateText(self, textHandle, newText):
        textHandle.set_text(newText)

    def create_datafile(self):
        if isLogData:
            # Data logging init
            cwd = os.getcwd()
            datafolder = '/data'
            self.dataFolder_path = cwd + datafolder

            # create data dir
            pathlib.Path(self.dataFolder_path).mkdir(parents=True, exist_ok=True)

            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%Hh%Mm%Ss")
            dataFiles = [None] * self.Nruns
            for k in range(0, self.Nruns):
                dataFiles[k] = self.dataFolder_path + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(
                    run=k + 1)
                with open(dataFiles[k], 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(
                        ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]',
                         'M [N m]'])
            dataFile = dataFiles[0]

            return dataFile

    def self.sysOut(self, x):
        y = np.zeros(dimOutput)
        # y = x[:3] + measNoise # <-- Measure only position and orientation
        y = x  # <-- Position, force and torque sensors on
        return y

    def enable_viz(self):
        if self.isVisualization:
            plt.close('all')

            self.simFig = plt.figure(figsize=(10, 10))

            # xy plane  
            self.xyPlaneAxs = self.simFig.add_subplot(221, autoscale_on=False, xlim=(self.xMin, self.xMax),
                                                      ylim=(self.yMin, self.yMax), xlabel='x [m]', ylabel='y [m]',
                                                      title='Pause - space, q - quit, click - data cursor')
            self.xyPlaneAxs.set_aspect('equal', adjustable='box')
            self.xyPlaneAxs.plot([self.xMin, self.xMax], [0, 0], 'k--', lw=0.75)  # Help line
            self.xyPlaneAxs.plot([0, 0], [self.yMin, self.yMax], 'k--', lw=0.75)  # Help line
            trajLine, = self.xyPlaneAxs.plot(xCoord0, yCoord0, 'b--', lw=0.5)
            self.robotMarker = pltMarker(angle=alphaDeg0)
            self.textTime = 't = {time:2.3f}'.format(time=t0)
            self.textTimeHandle = self.xyPlaneAxs.text(0.05, 0.95, self.textTime, horizontalalignment='left',
                                                       verticalalignment='center', transform=self.xyPlaneAxs.transAxes)
            self.xyPlaneAxs.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

            # Solution
            self.solAxs = self.simFig.add_subplot(222, autoscale_on=False, xlim=(t0, t1), ylim=(
            2 * np.min([self.xMin, self.yMin]), 2 * np.max([self.xMax, self.yMax])), xlabel='t [s]')
            self.solAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)  # Help line
            self.normLine, = solAxs.plot(t0, la.norm([self.xCoord0, self.yCoord0]), 'b-', lw=0.5,
                                         label=r'$\Vert(x,y)\Vert$ [m]')
            self.alphaLine, = solAxs.plot(self.t0, self.alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]')
            self.solAxs.legend(fancybox=True, loc='upper right')
            self.solAxs.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

            # Cost
            self.costAxs = self.simFig.add_subplot(223,
                                                   autoscale_on=False,
                                                   xlim=(self.t0, self.t1),
                                                   ylim=(0, 1e3 * self.rcost(self.y0, self.u0)),
                                                   yscale='symlog',
                                                   xlabel='t [s]')
            r = self.rcost(self.y0, self.u0)
            # textRcost = 'r = {r:2.3f}'.format(r = r)
            # textRcostHandle = self.simFig.text(0.05, 0.05, textRcost, horizontalalignment='left', verticalalignment='center')
            self.textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=self.icost_val)
            self.textIcostHandle = self.simFig.text(0.05, 0.5, textIcost, horizontalalignment='left',
                                                    verticalalignment='center')
            self.rcostLine, = self.costAxs.plot(self.t0, r, 'r-', lw=0.5, label='r')
            self.icostLine, = self.costAxs.plot(self.t0, self.icost_val, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
            costAxs.legend(fancybox=True, loc='upper right')

            # Control
            self.ctrlAxs = self.simFig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([self.Fmin, self.Mmin]), 1.1 * np.max([self.Fmax, self.Mmax])), xlabel='t [s]')
            self.ctrlAxs.plot([self.t0, self.t1], [0, 0], 'k--', lw=0.75)

            # Help line
            self.ctrlLines = self.ctrlAxs.plot(self.t0, toColVec(self.u0).T, lw=0.5)
            self.ctrlAxs.legend(iter(self.ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

            # Pack all lines together
            cLines = namedtuple('lines', ['trajLine', 'normLine', 'alphaLine', 'rcostLine', 'icostLine', 'ctrlLines'])
            self.lines = cLines(trajLine=trajLine, normLine=normLine, alphaLine=alphaLine, rcostLine=self.rcostLine,
                                icostLine=self.icostLine, ctrlLines=self.ctrlLines)

            # Enable data cursor
            for item in self.lines:
                if isinstance(item, list):
                    for subitem in item:
                        datacursor(subitem)
                else:
                    datacursor(item)

    def initAnim(self):
        self.animate_solScatter = self.xyPlaneAxs.scatter(self.xCoord0, self.yCoord0, marker=self.robotMarker.marker,
                                                          s=400, c='b')
        self.animate_currRun = 1
        return self.animate_solScatter, self.animate_currRun,

    def animate(self, k):
        # ------------------------------------simStep
        simulator.step()

        t = simulator.t
        ksi = simulator.y

        x = ksi[0:self.dimState]
        y = self.sysOut(x)

        if self.isDynCtrl:
            u = ksi[-self.dimInput:]
        else:
            self.ctrlStat(y, t)  # Updates self.ctrlStat_sampled_u
            u = self.ctrlStat_sampled_u

        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        alphaDeg = alpha / np.pi * 180
        v = ksi[3]
        omega = ksi[4]

        r = self.rcost(y, u)
        self.icost_val = self.icost(r, t)

        if isPrintSimStep:
            printSimStep(t, xCoord, yCoord, alpha, v, omega, self.icost_val, u)

        if isLogData:
            logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, self.icost_val, u)

        # ------------------------------------visuals     
        # xy plane    
        self.textTime = 't = {time:2.3f}'.format(time=t)
        self.updateText(self.textTimeHandle, self.textTime)
        self.updateLine(self.trajLine, *ksi[:2])  # Update the robot's track on the plot

        self.robotMarker.rotate(alphaDeg)  # Rotate the robot on the plot  
        self.animate_solScatter.remove()
        self.animate_solScatter = self.xyPlaneAxs.scatter(xCoord, yCoord, marker=self.robotMarker.marker, s=400, c='b')

        # Solution
        self.updateLine(self.normLine, t, la.norm([xCoord, yCoord]))
        self.updateLine(self.alphaLine, t, alpha)

        # Cost
        self.updateLine(self.rcostLine, t, r)
        self.updateLine(self.icostLine, t, self.icost_val)
        self.textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost=self.icost_val)
        self.updateText(self.textIcostHandle, self.textIcost)
        # Control
        for (line, uSingle) in zip(ctrlLines, u):
            self.updateLine(line, t, uSingle)

        # ------------------------------------run done
        if t >= t1:
            if isPrintSimStep:
                print(
                    '.....................................Run {run:2d} done.....................................'.format(
                        run=self.animate_currRun))

            self.animate_currRun += 1

            if self.animate_currRun > Nruns:
                anm.event_source.stop()
                return

            if isLogData:
                dataFile = dataFiles[self.animate_currRun - 1]

            # Reset simulator
            simulator.status = 'running'
            simulator.t = self.t0
            simulator.y = self.ksi0

            # Reset controller
            if isDynCtrl:
                ctrlDyn.ybuffer = repMatCustom(self.y0, 2, 1)
                ctrlDyn.itime = self.t0
            else:
                self.ctrlStat_ctrlClock = self.t0
                self.ctrlStat_sampled_u = self.u0

            self.icost_val = 0

            for item in lines:
                if item != trajLine:
                    if isinstance(item, list):
                        for subitem in item:
                            resetLine(subitem)
                    else:
                        resetLine(item)

            updateLine(trajLine, np.nan, np.nan)

        return self.animate_solScatter

    def run_sim(self):
        dataFile = create_datafile()

        if isPrintSimStep:
            warnings.filterwarnings('ignore')

        enable_viz()

        if self.isVisualization:
            anm = animation.FuncAnimation(self.simFig, self.animate, init_func=self.initAnim, blit=False,
                                          interval=self.dt / 1e3, repeat=False)

            anm.running = True

            self.simFig.tight_layout()

            plt.show()

        else:
            t = simulator.t

            self.animate_currRun = 1

            while True:
                simulator.step()

                t = simulator.t
                ksi = simulator.y

                x = ksi[0:dimState]
                y = self.sysOut(x)

                if isDynCtrl:
                    u = ksi[-dimInput:]
                else:
                    self.ctrlStat(y, t)  # Updates self.ctrlStat_sampled_u
                    u = self.ctrlStat_sampled_u

                xCoord = ksi[0]
                yCoord = ksi[1]
                alpha = ksi[2]
                v = ksi[3]
                omega = ksi[4]

                r = self.rcost(y, u)
                self.icost_val = self.icost(r, t)

                if isPrintSimStep:
                    printSimStep(t, xCoord, yCoord, alpha, v, omega, self.icost_val, u)

                if isLogData:
                    logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, self.icost_val, u)

                if t >= t1:
                    if self.isPrintSimStep:
                        print(
                            '.....................................Run {run:2d} done.....................................'.format(
                                run=self.animate_currRun))

                    self.animate_currRun += 1

                    if self.animate_currRun > Nruns:
                        break

                    if isLogData:
                        dataFile = dataFiles[self.animate_currRun - 1]

                    # Reset simulator
                    simulator.status = 'running'
                    simulator.t = self.t0
                    simulator.y = self.ksi0

                    # Reset controller
                    if isDynCtrl:
                        ctrlDyn.ybuffer = repMatCustom(self.y0, 2, 1)
                        ctrlDyn.itime = self.t0
                    else:
                        self.ctrlStat_ctrlClock = self.t0
                        self.ctrlStat_sampled_u = self.u0

                    self.icost_val = 0
