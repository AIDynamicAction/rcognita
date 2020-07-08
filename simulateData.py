#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:36:32 2020

@author: pavel
"""

#%% Import packages
from collections import namedtuple

import numpy as np
import numpy.linalg as la

import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpldatacursor import datacursor

from svgpath2mpl import parse_path

from tabulate import tabulate
# !pip install tabulate <-- to install this

#%% Initialization

# Specify yours here
dataFile = 'data/data-experimental.csv'

# isStaticVisualization: just plot the data at once
# isAnimate: plot the data dynamically as if the simulator were running
# Set either one to 1 and another one to 0
isAnimate = 1
isStaticVisualization = 0

isPrintSimStep = 1

#%% Service

def printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u):
    # alphaDeg = alpha/np.pi*180      
    
    headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'int r dt', 'F [N]', 'M [N m]']  
    dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]  
    rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f', '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')   
    table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
    
    print(table)

#%% Read data
    
rawData = np.loadtxt(dataFile, delimiter=",", skiprows=1)

ts = rawData[:,0]
xCoords = rawData[:,1]
yCoords = rawData[:,2]
alphas = rawData[:,3]
vs = rawData[:,4]
omegas = rawData[:,5]
icosts = rawData[:,6]
Fs = rawData[:,7]
Ms = rawData[:,8]

# Initial data
t0, t1 = ts[0], ts[-1]
xCoord0, yCoord0 = xCoords[0], yCoords[0]
alpha0, alphaDeg0 = alphas[0], alphas[0]/np.pi*180
icost0 = icosts[0]
F0, M0 = Fs[0], Ms[0]

norms = np.zeros(np.size(ts))
for k in range(np.size(ts)):
    norms[k] = la.norm([xCoords[k], yCoords[k]])

xMin = - np.max( [ np.abs(np.min([np.min(xCoords), np.min(yCoords)])), np.abs(np.max([np.max(xCoords), np.max(yCoords)])) ] ) - 1
xMax = -xMin
yMin = xMin
yMax = xMax

Fmin, Fmax = np.min(Fs), np.max(Fs)
Mmin, Mmax = np.min(Ms), np.max(Ms)

simStep = 0
     
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

#%% Visuals: setup

if isStaticVisualization:
    plt.close('all')
     
    simFig = plt.figure(figsize=(10,10))    
        
    # xy plane  
    xyPlaneAxs = simFig.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax), xlabel='x [m]', ylabel='y [m]',
                                    title='Pause - space, q - quit, click - data cursor')
    xyPlaneAxs.set_aspect('equal', adjustable='box')
    xyPlaneAxs.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
    xyPlaneAxs.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
    trajLine, = xyPlaneAxs.plot(xCoords, yCoords, 'b--', lw=0.5)
    robotMarker = pltMarker(angle=alphaDeg0)
    textTime = 't = {time:2.3f}'.format(time = t1)
    textTimeHandle = xyPlaneAxs.text(0.05, 0.95, textTime, horizontalalignment='left', verticalalignment='center', transform=xyPlaneAxs.transAxes)
    xyPlaneAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
    
    # Solution
    solAxs = simFig.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
    solAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
    normLine, = solAxs.plot(ts, norms, 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
    alphaLine, = solAxs.plot(ts, alphas, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
    solAxs.legend(fancybox=True, loc='upper right')
    solAxs.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
    
    # Cost
    costAxs = simFig.add_subplot(223, autoscale_on=False,
                                 xlim=(t0,t1),
                                 ylim=(0, 2*np.max(icosts) ),
                                 yscale='symlog', xlabel='t [s]')
    
    textIcost = r'$\int_{{t0}}^{{t1}} r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = icosts[-1])
    textIcostHandle = simFig.text(0.05, 0.5, textIcost, horizontalalignment='left', verticalalignment='center')
    icostLine, = costAxs.plot(ts, icosts, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
    costAxs.legend(fancybox=True, loc='upper right')
    
    # Control
    ctrlAxs = simFig.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
    ctrlAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
    ctrlLines = ctrlAxs.plot(ts, np.array([Fs, Ms]).T, lw=0.5)
    ctrlAxs.legend(iter(ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
    
    # Pack all lines together
    cLines = namedtuple('lines', ['trajLine', 'normLine', 'alphaLine', 'icostLine', 'ctrlLines'])
    lines = cLines(trajLine=trajLine, normLine=normLine, alphaLine=alphaLine, icostLine=icostLine, ctrlLines=ctrlLines)
    
    # Enable data cursor
    for item in lines:
        if isinstance(item, list):
            for subitem in item:
                datacursor(subitem)
        else:
            datacursor(item)
            
elif isAnimate:  
    # For animation, data samples are limited to 500 for speed
    # animInterval is the interval between animation frames and should be adjusted depending on gear for better view
    if np.size(ts) > 500:
        simStepSize = int(np.size(ts)/500)
        animInterval = 1e-4
    else:
        simStepSize = 1
        animInterval = np.size(ts)/1e6
    
    plt.close('all')
     
    simFig = plt.figure(figsize=(10,10))    
        
    # xy plane  
    xyPlaneAxs = simFig.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax), xlabel='x [m]', ylabel='y [m]',
                                    title='Pause - space, q - quit, click - data cursor')
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
                                 ylim=(0, 2*np.max(icosts) ),
                                 yscale='symlog', xlabel='t [s]')
    
    textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = icosts[0])
    textIcostHandle = simFig.text(0.05, 0.5, textIcost, horizontalalignment='left', verticalalignment='center')
    icostLine, = costAxs.plot(t0, icost0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
    costAxs.legend(fancybox=True, loc='upper right')
    
    # Control
    ctrlAxs = simFig.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
    ctrlAxs.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
    ctrlLines = ctrlAxs.plot(t0, np.array([[F0], [M0]]).T, lw=0.5)
    ctrlAxs.legend(iter(ctrlLines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
    
    # Pack all lines together
    cLines = namedtuple('lines', ['trajLine', 'normLine', 'alphaLine', 'icostLine', 'ctrlLines'])
    lines = cLines(trajLine=trajLine, normLine=normLine, alphaLine=alphaLine, icostLine=icostLine, ctrlLines=ctrlLines)
    
    # Enable data cursor
    for item in lines:
        if isinstance(item, list):
            for subitem in item:
                datacursor(subitem)
        else:
            datacursor(item)
        
#%% Visuals: init & animate

def initAnim():
    animate.solScatter = xyPlaneAxs.scatter(xCoord0, yCoord0, marker=robotMarker.marker, s=400, c='b')
    animate.currRun = 1
    return animate.solScatter, animate.currRun, 
    
def animate(k):
    global simStep
    
    simStep += simStepSize
    
    # Get data row
    t, xCoord, yCoord, alpha, v, omega, icost, F, M = ts[simStep], xCoords[simStep], yCoords[simStep], alphas[simStep], vs[simStep], omegas[simStep], icosts[simStep], Fs[simStep], Ms[simStep]
    
    alphaDeg = alpha/np.pi*180
    
    if isPrintSimStep:
        printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, [F, M])
    
    # xy-plane    
    textTime = 't = {time:2.3f}'.format(time = t)
    updateText(textTimeHandle, textTime)
    updateLine(trajLine, xCoord, yCoord)  # Update the robot's track on the plot
    
    robotMarker.rotate(alphaDeg)    # Rotate the robot on the plot  
    animate.solScatter.remove()
    animate.solScatter = xyPlaneAxs.scatter(xCoord, yCoord, marker=robotMarker.marker, s=400, c='b')
    
    # Solution
    updateLine(normLine, t, la.norm([xCoord, yCoord]))
    updateLine(alphaLine, t, alpha)

    # Cost
    updateLine(icostLine, t, icost)
    textIcost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost)
    updateText(textIcostHandle, textIcost)
    
    # Control
    updateLine(ctrlLines[0], t, F)
    updateLine(ctrlLines[1], t, M)
    
    if t >= t1:  
        anm.event_source.stop()
        return
    
    return animate.solScatter

#%% Main loop

if isAnimate:
    cId = simFig.canvas.mpl_connect('key_press_event', onKeyPress)
       
    anm = animation.FuncAnimation(simFig, animate, init_func=initAnim, blit=False, interval=animInterval, repeat=False)
    anm.running = True
    
    simFig.tight_layout()
    
    plt.show()
    
elif isPrintSimStep:   
    while True:
        # Get data row
        t, xCoord, yCoord, alpha, v, omega, icost, F, M = ts[simStep], xCoords[simStep], yCoords[simStep], alphas[simStep], vs[simStep], omegas[simStep], icosts[simStep], Fs[simStep], Ms[simStep]
        
        printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, [F, M])
        
        simStep += 1
        
        if t >= t1:  
            break