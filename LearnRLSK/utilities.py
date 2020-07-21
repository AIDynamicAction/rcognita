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
    if event.key==' ':
        if anm.running:
            anm.event_source.stop()
            
        else:
            anm.event_source.start()
        anm.running ^= True
    elif event.key=='q':
        plt.close('all')
        raise Exception('exit')


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
        u = nominalCtrl.computeAction(t, y)
    elif mode > 0: # Optimal controller
        u = agent.computeAction(t, y)
        
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

class animator:
    """
    Class containing machinery for the use in ``animation.FuncAnimation``
    
    Attributes
    ----------
    objects : : tuple
        Objects to be updated within animation cycle
    pars : : tuple
        Fixed parameters of objects and visual elements      
    
    Customization
    -------------    
    
    This is a quick, *ad hoc* implementation. Entities, in particular visual elements, and methods should be adjusted for each particular application
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities (adjust for custom use)
        self.simulator, self.sys, self.nominalCtrl, self.agent, self.dataFiles, self.ctrlSelector, self.printSimStep, self.logDataRow = self.objects
        
        self.dimState, self.x0, u0, self.t0, self.t1, self.ksi0, xMin, xMax, yMin, yMax, Fmin, Fmax, Mmin, Mmax, ctrlMode, uMan, Nruns, isPrintSimStep, isLogData = self.pars 
        
        self.ctrlMode = ctrlMode
        self.uMan = uMan
        self.Nruns = Nruns
        self.isPrintSimStep = isPrintSimStep
        self.isLogData = isLogData
        
        y0 = self.sys.out(self.x0)
        xCoord0 = self.x0[0]
        yCoord0 = self.x0[1]
        alpha0 = self.x0[2]
        alphaDeg0 = alpha0/2/np.pi
        
        plt.close('all')
     
        self.simFig = plt.figure(figsize=(10,10))    
            
        # xy plane  
        self.xyPlaneAxs = self.simFig.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax),
                                                  xlabel='x [m]', ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')
        self.xyPlaneAxs.set_aspect('equal', adjustable='box')
        self.xyPlaneAxs.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
        self.xyPlaneAxs.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
        self.trajLine, = self.xyPlaneAxs.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robotMarker = pltMarker(angle=alphaDeg0)
        textTime = 't = {time:2.3f}'.format(time = self.t0)
        self.textTimeHandle = self.xyPlaneAxs.text(0.05, 0.95, textTime,
                                                   horizontalalignment='left', verticalalignment='center', transform=self.xyPlaneAxs.transAxes)
        self.xyPlaneAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Solution
        self.solAxs = self.simFig.add_subplot(222, autoscale_on=False, xlim=(self.t0,self.t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
        self.solAxs.plot([self.t0, self.t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.normLine, = self.solAxs.plot(self.t0, la.norm([xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.alphaLine, = self.solAxs.plot(self.t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
        self.solAxs.legend(fancybox=True, loc='upper right')
        self.solAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Cost
        self.costAxs = self.simFig.add_subplot(223, autoscale_on=False, xlim=(self.t0,self.t1), ylim=(0, 1e4*self.agent.rcost( y0, u0 ) ), yscale='symlog', xlabel='t [s]')
        
        r = self.agent.rcost(y0, u0)
        textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = 0)
        self.textIcostHandle = self.simFig.text(0.05, 0.5, textIcost, horizontalalignment='left', verticalalignment='center')
        self.rcostLine, = self.costAxs.plot(self.t0, r, 'r-', lw=0.5, label='r')
        self.icostLine, = self.costAxs.plot(self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.costAxs.legend(fancybox=True, loc='upper right')
        
        # Control
        self.ctrlAxs = self.simFig.add_subplot(224, autoscale_on=False, xlim=(self.t0,self.t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
        self.ctrlAxs.plot([self.t0, self.t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.ctrlLines = self.ctrlAxs.plot(self.t0, toColVec(u0).T, lw=0.5)
        self.ctrlAxs.legend(iter(self.ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['trajLine', 'normLine', 'alphaLine', 'rcostLine', 'icostLine', 'ctrlLines'])
        self.lines = cLines(trajLine=self.trajLine,
                            normLine=self.normLine,
                            alphaLine=self.alphaLine,
                            rcostLine=self.rcostLine,
                            icostLine=self.icostLine,
                            ctrlLines=self.ctrlLines)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)
    
    def initAnim(self):
        xCoord0 = self.x0[0]
        yCoord0 = self.x0[1]       
        
        self.solScatter = self.xyPlaneAxs.scatter(xCoord0, yCoord0, marker=self.robotMarker.marker, s=400, c='b')
        self.currRun = 1
        self.currDataFile = self.dataFiles[0]
     
    def updateLine(self, line, newX, newY):
        line.set_xdata( np.append( line.get_xdata(), newX) )
        line.set_ydata( np.append( line.get_ydata(), newY) )  
        
    def resetLine(self, line):
        line.set_data([], [])     
     
    def updateScatter(self, scatter, newX, newY):
        scatter.set_offsets( np.vstack( [ scatter.get_offsets().data, np.c_[newX, newY] ] ) )
        
    def updateText(self, textHandle, newText):
        textHandle.set_text(newText)
   
    def animate(self, k):

        self.simulator.step()
        
        t = self.simulator.t
        ksi = self.simulator.y
        
        x = ksi[0:self.dimState]
        y = self.sys.out(x)
        
        u = self.ctrlSelector(t, y, self.uMan, self.nominalCtrl, self.agent, self.ctrlMode)

        self.sys.receiveAction(u)
        self.agent.receiveSysState(self.sys._x) 
        self.agent.update_icost(y, u)
        
        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        alphaDeg = alpha/np.pi*180
        v = ksi[3]
        omega = ksi[4]
        
        r = self.agent.rcost(y, u)
        icost = self.agent.icostVal
        
        if self.isPrintSimStep:
            self.printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u)
            
        if self.isLogData:
            self.logDataRow(self.currDataFile, t, xCoord, yCoord, alpha, v, omega, icost.val, u)
        
        # xy plane    
        textTime = 't = {time:2.3f}'.format(time = t)
        self.updateText(self.textTimeHandle, textTime)
        self.updateLine(self.trajLine, *ksi[:2])  # Update the robot's track on the plot
        
        self.robotMarker.rotate(alphaDeg)    # Rotate the robot on the plot  
        self.solScatter.remove()
        self.solScatter = self.xyPlaneAxs.scatter(xCoord, yCoord, marker=self.robotMarker.marker, s=400, c='b')
        
        # Solution
        self.updateLine(self.normLine, t, la.norm([xCoord, yCoord]))
        self.updateLine(self.alphaLine, t, alpha)
    
        # Cost
        self.updateLine(self.rcostLine, t, r)
        self.updateLine(self.icostLine, t, icost)
        textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost)
        self.updateText(self.textIcostHandle, textIcost)
        # Control
        for (line, uSingle) in zip(self.ctrlLines, u):
            self.updateLine(line, t, uSingle)
    
        # Run done
        if t >= self.t1:  
            if self.isPrintSimStep:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.currRun))
                
            self.currRun += 1
            
            if self.currRun > self.Nruns:
                return
            
            if isLogData:
                self.currDataFile = self.dataFiles[self.currRun-1]
            
            # Reset simulator
            self.simulator.status = 'running'
            self.simulator.t = self.t0
            self.simulator.y = self.ksi0
            
            # Reset controller
            if self.ctrlMode > 0:
                self.agent.reset(self.t0)
            else:
                self.nominalCtrl.reset(self.t0)
            
            icost = 0      
            
            for item in self.lines:
                if item != self.trajLine:
                    if isinstance(item, list):
                        for subitem in item:
                            self.resetLine(subitem)
                    else:
                        self.resetLine(item)
    
            self.updateLine(self.trajLine, np.nan, np.nan)
        
        return self.solScatter