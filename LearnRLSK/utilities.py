import os
import pathlib

from svgpath2mpl import parse_path
from collections import namedtuple
from mpldatacursor import datacursor
from datetime import datetime
from tabulate import tabulate

import numpy as np
from numpy.matlib import repmat
import numpy.linalg as la

import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%% Utilities
def toColVec(argin):
    if argin.ndim < 2:
        return np.reshape(argin, (argin.size, 1))
    elif argin.ndim ==2:
        if argin.shape[0] < argin.shape[1]:
            return argin.T
        else:
            return argin

def repMat(argin, n, m):
    """
    Ensures 1D result
    
    """
    return np.squeeze(repmat(argin, n, m))

# def pushColRight(matrix, vec):
#     return np.hstack([matrix[:,1:], toColVec(vec)])

def pushVec(matrix, vec):
    return np.vstack([matrix[1:,:], vec])

def uptria2vec(mat):
    """
    Convert upper triangular square sub-matrix to column vector
    
    """    
    n = mat.shape[0]
    
    vec = np.zeros( n*(n+1)/2, 1 )
    
    k = 0
    for i in range(n):
        for j in range(n):
            vec[j] = mat[i, j]
            k += 1

def logdata(Nruns, save=False):
    dataFiles = [None] * Nruns

    if save:
        cwd = os.getcwd()
        datafolder = '/data'
        dataFolder_path = cwd + datafolder
        
        # create data dir
        pathlib.Path(dataFolder_path).mkdir(parents=True, exist_ok=True) 

        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%Hh%Mm%Ss")
        dataFiles = [None] * Nruns
        for k in range(0, Nruns):
            dataFiles[k] = dataFolder_path + '/RLsim__' + date + '__' + time + '__run{run:02d}.csv'.format(run=k+1)
            with open(dataFiles[k], 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]'] )

    return dataFiles        

def onKeyPress(event, anm):
    if event.key == ' ':
        if anm.running is True:
            anm.event_source.stop()
            anm.running = False
            
        elif anm.running is False:
            anm.event_source.start()
            anm.running = True
        
    elif event.key == 'q':
        plt.close('all')
        print("Program exit")
        os._exit(1)

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


def ctrlSelector(t, y, uMan, nominalCtrl, agent, mode):
    """
    Main interface for different agents

    Parameters
    ----------
    mode : : integer
        Agent mode, see ``user settings`` section

    Returns
    -------
    u : : array of shape ``[dimInput, ]``
        Control action
        
    Customization
    -------
    Include your controller modes in this method    

    """
    
    if mode==0: # Manual control
        u = uMan
    elif mode==-1: # Nominal controller
        u = nominalCtrl.compute_action(t, y)
    elif mode > 0: # Optimal controller
        u = agent.compute_action(t, y)
        
    return u


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
        
    def updatePars(self, Anew, Bnew, Cnew, Dnew):
        self.A = Anew
        self.B = Bnew
        self.C = Cnew
        self.D = Dnew
        
    def updateIC(self, x0setNew):
        self.x0set = x0setNew

class ZOH:
    """
    Zero-order hold
    
    """    
    def __init__(self, initTime=0, initVal=0, samplTime=1):
        self.timeStep = initTime
        self.samplTime = samplTime
        self.currVal = initVal
        
    def hold(self, signalVal, t):
        timeInSample = t - self.timeStep
        if timeInSample >= self.samplTime: # New sample
            self.timeStep = t
            self.currVal = signalVal

        return self.currVal


class dfilter:
    """
    Real-time digital filter
    
    """
    def __init__(self, filterNum, filterDen, bufferSize=16, initTime=0, initVal=0, samplTime=1):
        self.Num = filterNum
        self.Den = filterDen
        self.zi = repMat( signal.lfilter_zi(filterNum, filterDen), 1, initVal.size)
        
        self.timeStep = initTime
        self.samplTime = samplTime
        self.buffer = repMat(initVal, 1, bufferSize)
        
    def filt(self, signalVal, t=None):
        # Sample only if time is specified
        if t is not None:
            timeInSample = t - self.timeStep
            if timeInSample >= self.samplTime: # New sample
                self.timeStep = t
                self.buffer = pushVec(self.buffer, signalVal)
        else:
            self.buffer = pushVec(self.buffer, signalVal)
        
        bufferFiltered = np.zeros(self.buffer.shape)
        
        for k in range(0, signalVal.size):
                bufferFiltered[k,:], self.zi[k] = signal.lfilter(self.Num, self.Den, self.buffer[k,:], zi=self.zi[k, :])
        return bufferFiltered[-1,:]


class pltMarker:
    """
    Robot marker for visualization
    
    """    
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
    
