#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:36:32 2020

@author: Pavel Osinenko
"""

"""
=============================================================================
rcognita

https://github.com/AIDynamicAction/rcognita

Python framework for hybrid simulation of predictive reinforcement learning agents and classical controllers

=============================================================================

This module:

play back a single run of a saved simulation of a 3-wheel robot

=============================================================================

Remark:

All vectors are treated as of type [n,]
All buffers are treated as of type [L, n] where each row is a vector
Buffers are updated from bottom to top
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import visuals
import loggers

import numpy.linalg as la

from utilities import on_key_press
from argparse import ArgumentParser

# Specify yours here
# datafile = 'data/data-experimental.csv'
parser = ArgumentParser()
parser.add_argument('--datafile', type=str, default='data/RLsim__2021-01-11__14h29m57s__run01.csv')

args = parser.parse_args()
datafile = args.datafile

#------------------------------------user settings
# Save animation to an mp4-file
is_save_anim = 0

# Speed-up factor. Use to increase playback speed
speedup = 1
#------------------------------------read data
rawData = np.loadtxt(datafile, delimiter=",", skiprows=1)

ts = rawData[:,0]
xCoords = rawData[:,1]
yCoords = rawData[:,2]
alphas = rawData[:,3]
vs = rawData[:,4]
omegas = rawData[:,5]
rs = rawData[:,6]
icosts = rawData[:,7]
Fs = rawData[:,8]
Ms = rawData[:,9]

# Initial data
t0, t1 = ts[0], ts[-1]
xCoord0, yCoord0 = xCoords[0], yCoords[0]
alpha0, alphaDeg0 = alphas[0], alphas[0]/np.pi*180
v0, omega0 = vs[0], omegas[0]
icost0 = icosts[0]
F0, M0 = Fs[0], Ms[0]

x0 = np.array([xCoord0, yCoord0, alpha0, v0, omega0])
u0 = np.array([F0, M0])
ksi0 = x0

r0 = rs[0]

norms = np.zeros(np.size(ts))
for k in range(np.size(ts)):
    norms[k] = la.norm([xCoords[k], yCoords[k]])

xMin = - np.max( [ np.abs(np.min([np.min(xCoords), np.min(yCoords)])), np.abs(np.max([np.max(xCoords), np.max(yCoords)])) ] ) - 1
xMax = -xMin
yMin = xMin
yMax = xMax

Fmin, Fmax = np.min(Fs), np.max(Fs)
Mmin, Mmax = np.min(Ms), np.max(Ms)

#------------------------------------initialization
my_logger = loggers.logger_3wrobot()

my_animator = visuals.animator_3wrobot(objects=([], [], [], [], ['dummy'], [], my_logger),
                                       pars=(x0, u0, t0, t1, ksi0, xMin, xMax, yMin, yMax, [], [], Fmin, Mmin, Fmax, Mmax, 1,
                                             1, 0, 1, r0))

# For animation, data samples are limited to 500 for speed.
# `anim_interval` is the interval between animation frames and should be adjusted depending on hardware for better view
if np.size(ts) > 500:
    sim_step_size = int( np.floor( (np.size(ts)/500) ) )
    anim_interval = 1e-4
    Nframes = 500
else:
    sim_step_size = 1
    anim_interval = np.size(ts)/1e6
    Nframes = np.size(ts)

sim_step_size = speedup * sim_step_size

# Down-sample and feed data into the animator
my_animator.get_sim_data(ts[:-sim_step_size:sim_step_size],
                         xCoords[:-sim_step_size:sim_step_size],
                         yCoords[:-sim_step_size:sim_step_size],
                         alphas[:-sim_step_size:sim_step_size],
                         vs[:-sim_step_size:sim_step_size],
                         omegas[:-sim_step_size:sim_step_size],
                         rs[:-sim_step_size:sim_step_size],
                         icosts[:-sim_step_size:sim_step_size],
                         Fs[:-sim_step_size:sim_step_size],
                         Ms[:-sim_step_size:sim_step_size])

vids_folder = 'vids'

if is_save_anim:
    ffmpeg_writer = animation.writers['ffmpeg']
    metadata = dict(title='RL demo', artist='Matplotlib', comment='Robot parking example')
    writer = ffmpeg_writer(fps=30, metadata=metadata)

#------------------------------------main playback loop
# This is how you debug `FuncAnimation` if needed: just uncomment these two lines and comment out everything that has to do with `FuncAnimation`
# my_animator.init_anim()
# my_animator.animate(1)

anm = animation.FuncAnimation(my_animator.fig_sim,
                              my_animator.animate,
                              init_func=my_animator.init_anim,
                              blit=False, interval=anim_interval, repeat=False, frames=Nframes)

cId = my_animator.fig_sim.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, anm))

anm.running = True

my_animator.fig_sim.tight_layout()
plt.savefig('frame.png')
plt.show()


if is_save_anim:
    anm.save(vids_folder + '/' + datafile.split('.')[0].split('/')[-1]+'.mp4', writer=writer, dpi=200)
