#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains an interface class `animator` along with concrete realizations, each of which is associated with a corresponding system

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import numpy.linalg as la
from .utilities import upd_line
from .utilities import reset_line
from .utilities import upd_scatter
from .utilities import upd_text
from .utilities import to_col_vec

import matplotlib as mpl 
import matplotlib.pyplot as plt
# !pip install mpldatacursor <-- to install this
from mpldatacursor import datacursor

# !pip install svgpath2mpl matplotlib <-- to install this
from svgpath2mpl import parse_path

from collections import namedtuple

class Animator:
    """
    Interface class of visualization machinery for simulation of system-controller loops.
    To design a concrete animator: inherit this class, override:
        | :func:`~visuals.animator.__init__` :
        | define necessary visual elements (required)
        | :func:`~visuals.animator.init_anim` :
        | initialize necessary visual elements (required)        
        | :func:`~visuals.animator.animate` :
        | animate visual elements (required)
    
    Attributes
    ----------
    objects : : tuple
        Objects to be updated within animation cycle
    pars : : tuple
        Fixed parameters of objects and visual elements      
    
    """
    def __init__(self, objects=[], pars=[]):
        pass
    
    def init_anim(self):
        pass
     
    def animate(self, k):
        pass
    
    def get_anm(self, anm):
        """
        ``anm`` should be a ``FuncAnimation`` object.
        This method is needed to hand the animator access to the currently running animation, say, via ``anm.event_source.stop()``
    
        """   
        self.anm = anm
        
    def stop_anm(self):
        """
        Stops animation, provided that ``self.anm`` was defined via ``get_anm``

        """
        self.anm.event_source.stop()           
        # plt.close('all')
        raise Exception('exit')        

class RobotMarker:
    """
    Robot marker for visualization.
    
    """    
    def __init__(self, angle=None, path_string=None):
        self.angle = angle or []
        self.path_string = path_string or """m 66.893258,227.10128 h 5.37899 v 0.91881 h 1.65571 l 1e-5,-3.8513 3.68556,-1e-5 v -1.43933
        l -2.23863,10e-6 v -2.73937 l 5.379,-1e-5 v 2.73938 h -2.23862 v 1.43933 h 3.68556 v 8.60486 l -3.68556,1e-5 v 1.43158
        h 2.23862 v 2.73989 h -5.37899 l -1e-5,-2.73989 h 2.23863 v -1.43159 h -3.68556 v -3.8513 h -1.65573 l 1e-5,0.91881 h -5.379 z"""
        self.path = parse_path( self.path_string )
        self.path.vertices -= self.path.vertices.mean( axis=0 )
        self.marker = mpl.markers.MarkerStyle( marker=self.path )
        self.marker._transform = self.marker.get_transform().rotate_deg(angle)

    def rotate(self, angle=0):
        self.marker._transform = self.marker.get_transform().rotate_deg(angle-self.angle)
        self.angle = angle
    
class Animator3WRobot(Animator):
    """
    Animator class for a 3-wheel robot with dynamic actuators. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        x0, u0, t0, t1, ksi0, xMin, xMax, yMin, yMax, ctrl_mode, uMan, Fmin, Mmin, Fmax, Mmax, Nruns, is_print_sim_step, is_log_data, is_playback, r0 = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.ksi0 = ksi0
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.uMan = uMan
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        xCoord0 = x0[0]
        yCoord0 = x0[1]
        alpha0 = x0[2]
        alpha_deg0 = alpha0/2/np.pi
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))    
            
        # xy plane  
        self.axs_xy_plane = self.fig_sim.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax),
                                                  xlabel='x [m]', ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')
        self.axs_xy_plane.set_aspect('equal', adjustable='box')
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
        self.line_traj, = self.axs_xy_plane.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = 't = {time:2.3f}'.format(time = t0)
        self.text_time_handle = self.axs_xy_plane.text(0.05, 0.95, text_time,
                                                   horizontalalignment='left', verticalalignment='center', transform=self.axs_xy_plane.transAxes)
        self.axs_xy_plane.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Solution
        self.axs_sol = self.fig_sim.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_norm, = self.axs_sol.plot(t0, la.norm([xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.line_alpha, = self.axs_sol.plot(t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
        self.axs_sol.legend(fancybox=True, loc='upper right')
        self.axs_sol.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Cost
        if is_playback:
            r = r0
        else:
            y0 = self.sys.out(x0)
            r = self.ctrl_benchmarking.rcost(y0, u0)
        
        self.axs_cost = self.fig_sim.add_subplot(223, autoscale_on=False, xlim=(t0,t1), ylim=(0, 1e4*r), yscale='symlog', xlabel='t [s]')
        
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = 0)
        self.text_icost_handle = self.fig_sim.text(0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')
        self.line_rcost, = self.axs_cost.plot(t0, r, 'r-', lw=0.5, label='r')
        self.line_icost, = self.axs_cost.plot(t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Control
        self.axs_ctrl = self.fig_sim.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([Fmin, Mmin]), 1.1*np.max([Fmax, Mmax])), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.lines_ctrl = self.axs_ctrl.plot(t0, to_col_vec(u0).T, lw=0.5)
        self.axs_ctrl.legend(iter(self.lines_ctrl), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['line_traj', 'line_norm', 'line_alpha', 'line_rcost', 'line_icost', 'lines_ctrl'])
        self.lines = cLines(line_traj=self.line_traj,
                            line_norm=self.line_norm,
                            line_alpha=self.line_alpha,
                            line_rcost=self.line_rcost,
                            line_icost=self.line_icost,
                            lines_ctrl=self.lines_ctrl)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)
    
    def set_sim_data(self, ts, xCoords, yCoords, alphas, vs, omegas, rs, icosts, Fs, Ms):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.xCoords, self.yCoords, self.alphas, self.vs, self.omegas = ts, xCoords, yCoords, alphas, vs, omegas
        self.rs, self.icosts, self.Fs, self.Ms = rs, icosts, Fs, Ms
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.ksi = np.array([self.xCoords[self.curr_step], self.yCoords[self.curr_step], self.alphas[self.curr_step], self.vs[self.curr_step], self.omegas[self.curr_step]])
        self.r = self.rs[self.curr_step]
        self.icost = self.icosts[self.curr_step]
        self.u = np.array([self.Fs[self.curr_step], self.Ms[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        x0, *_ = self.pars
        
        xCoord0 = x0[0]
        yCoord0 = x0[1]       
        
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c='b')
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            ksi = self.ksi
            u = self.u
            r = self.r
            icost = self.icost        
            
        else:
            self.simulator.sim_step()
            
            t, x, y, ksi = self.simulator.get_sim_step_data()
            
            u = self.ctrl_selector(t, y, self.uMan, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(u)
            self.ctrl_benchmarking.receive_sys_state(self.sys._x) 
            self.ctrl_benchmarking.upd_icost(y, u)
            
            r = self.ctrl_benchmarking.rcost(y, u)
            icost = self.ctrl_benchmarking.icost_val
        
        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        alpha_deg = alpha/np.pi*180
        v = ksi[3]
        omega = ksi[4]

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, xCoord, yCoord, alpha, v, omega, r, icost, u)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, xCoord, yCoord, alpha, v, omega, r, icost, u)
        
        # xy plane  
        text_time = 't = {time:2.3f}'.format(time = t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot
            
        self.robot_marker.rotate(1e-3)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(5, 5, marker=self.robot_marker.marker, s=400, c='b')       
        
        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord, yCoord, marker=self.robot_marker.marker, s=400, c='b')
        
        # # Solution
        upd_line(self.line_norm, t, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, t, alpha)
    
        # Cost
        upd_line(self.line_rcost, t, r)
        upd_line(self.line_icost, t, icost)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost)
        upd_text(self.text_icost_handle, text_icost)
        
        # Control
        for (line, uSingle) in zip(self.lines_ctrl, u):
            upd_line(line, t, uSingle)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))
                
            self.run_curr += 1
            
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            icost = 0     
            
            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_rcost)
            reset_line(self.line_icost)
            reset_line(self.lines_ctrl[0])
            reset_line(self.lines_ctrl[1])
            
            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)
    
            upd_line(self.line_traj, np.nan, np.nan)

class Animator3WRobotNI(Animator):
    """
    Animator class for a 3-wheel robot with static actuators. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        x0, u0, t0, t1, ksi0, xMin, xMax, yMin, yMax, ctrl_mode, uMan, v_min, omega_min, v_max, omega_max, Nruns, is_print_sim_step, is_log_data, is_playback, r0 = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.ksi0 = ksi0
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.uMan = uMan
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        xCoord0 = x0[0]
        yCoord0 = x0[1]
        alpha0 = x0[2]
        alpha_deg0 = alpha0/2/np.pi
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))    
            
        # xy plane  
        self.axs_xy_plane = self.fig_sim.add_subplot(221, autoscale_on=False, xlim=(xMin,xMax), ylim=(yMin,yMax),
                                                  xlabel='x [m]', ylabel='y [m]', title='Pause - space, q - quit, click - data cursor')
        self.axs_xy_plane.set_aspect('equal', adjustable='box')
        self.axs_xy_plane.plot([xMin, xMax], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_xy_plane.plot([0, 0], [yMin, yMax], 'k--', lw=0.75)   # Help line
        self.line_traj, = self.axs_xy_plane.plot(xCoord0, yCoord0, 'b--', lw=0.5)
        self.robot_marker = RobotMarker(angle=alpha_deg0)
        text_time = 't = {time:2.3f}'.format(time = t0)
        self.text_time_handle = self.axs_xy_plane.text(0.05, 0.95, text_time,
                                                   horizontalalignment='left', verticalalignment='center', transform=self.axs_xy_plane.transAxes)
        self.axs_xy_plane.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Solution
        self.axs_sol = self.fig_sim.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=( 2 * np.min([xMin, yMin]), 2 * np.max([xMax, yMax]) ), xlabel='t [s]')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_norm, = self.axs_sol.plot(t0, la.norm([xCoord0, yCoord0]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
        self.line_alpha, = self.axs_sol.plot(t0, alpha0, 'r-', lw=0.5, label=r'$\alpha$ [rad]') 
        self.axs_sol.legend(fancybox=True, loc='upper right')
        self.axs_sol.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Cost
        if is_playback:
            r = r0
        else:
            y0 = self.sys.out(x0)
            r = self.ctrl_benchmarking.rcost(y0, u0)
        
        self.axs_cost = self.fig_sim.add_subplot(223, autoscale_on=False, xlim=(t0,t1), ylim=(0, 1e4*r), yscale='symlog', xlabel='t [s]')
        
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = 0)
        self.text_icost_handle = self.fig_sim.text(0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')
        self.line_rcost, = self.axs_cost.plot(t0, r, 'r-', lw=0.5, label='r')
        self.line_icost, = self.axs_cost.plot(t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Control
        self.axs_ctrl = self.fig_sim.add_subplot(224, autoscale_on=False, xlim=(t0,t1), ylim=(1.1*np.min([v_min, omega_min]), 1.1*np.max([v_max, omega_max])), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.lines_ctrl = self.axs_ctrl.plot(t0, to_col_vec(u0).T, lw=0.5)
        self.axs_ctrl.legend(iter(self.lines_ctrl), ('v [m/s]', r'$\omega$ [rad/s]'), fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['line_traj', 'line_norm', 'line_alpha', 'line_rcost', 'line_icost', 'lines_ctrl'])
        self.lines = cLines(line_traj=self.line_traj,
                            line_norm=self.line_norm,
                            line_alpha=self.line_alpha,
                            line_rcost=self.line_rcost,
                            line_icost=self.line_icost,
                            lines_ctrl=self.lines_ctrl)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)
    
    def set_sim_data(self, ts, xCoords, yCoords, alphas, rs, icosts, vs, omegas):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.xCoords, self.yCoords, self.alphas = ts, xCoords, yCoords, alphas
        self.rs, self.icosts, self.vs, self.omegas = rs, icosts, vs, omegas
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.ksi = np.array([self.xCoords[self.curr_step], self.yCoords[self.curr_step], self.alphas[self.curr_step]])
        self.r = self.rs[self.curr_step]
        self.icost = self.icosts[self.curr_step]
        self.u = np.array([self.vs[self.curr_step], self.omegas[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        x0, *_ = self.pars
        
        xCoord0 = x0[0]
        yCoord0 = x0[1]       
        
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord0, yCoord0, marker=self.robot_marker.marker, s=400, c='b')
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            ksi = self.ksi
            u = self.u
            r = self.r
            icost = self.icost        
            
        else:
            self.simulator.sim_step()
            
            t, x, y, ksi = self.simulator.get_sim_step_data()
            
            u = self.ctrl_selector(t, y, self.uMan, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(u)
            self.ctrl_benchmarking.receive_sys_state(self.sys._x) 
            self.ctrl_benchmarking.upd_icost(y, u)
            
            r = self.ctrl_benchmarking.rcost(y, u)
            icost = self.ctrl_benchmarking.icost_val
        
        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        alpha_deg = alpha/np.pi*180

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, xCoord, yCoord, alpha, r, icost, u)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, xCoord, yCoord, alpha, r, icost, u)
        
        # xy plane  
        text_time = 't = {time:2.3f}'.format(time = t)
        upd_text(self.text_time_handle, text_time)
        upd_line(self.line_traj, xCoord, yCoord)  # Update the robot's track on the plot
            
        self.robot_marker.rotate(1e-3)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(5, 5, marker=self.robot_marker.marker, s=400, c='b')       
        
        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot  
        self.scatter_sol.remove()
        self.scatter_sol = self.axs_xy_plane.scatter(xCoord, yCoord, marker=self.robot_marker.marker, s=400, c='b')
        
        # # Solution
        upd_line(self.line_norm, t, la.norm([xCoord, yCoord]))
        upd_line(self.line_alpha, t, alpha)
    
        # Cost
        upd_line(self.line_rcost, t, r)
        upd_line(self.line_icost, t, icost)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost)
        upd_text(self.text_icost_handle, text_icost)
        
        # Control
        for (line, uSingle) in zip(self.lines_ctrl, u):
            upd_line(line, t, uSingle)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))  
            
            self.run_curr += 1
                    
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            icost = 0     
            
            reset_line(self.line_norm)
            reset_line(self.line_alpha)
            reset_line(self.line_rcost)
            reset_line(self.line_icost)
            reset_line(self.lines_ctrl[0])
            reset_line(self.lines_ctrl[1])
            
            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)
    
            upd_line(self.line_traj, np.nan, np.nan)
            
class Animator2Tank(Animator):
    """
    Animator class for a 2-tank system. 
    
    """
    def __init__(self, objects=[], pars=[]):
        self.objects = objects
        self.pars = pars
        
        # Unpack entities
        self.simulator, self.sys, self.ctrl_nominal, self.ctrl_benchmarking, self.datafiles, self.ctrl_selector, self.logger = self.objects
        
        x0, u0, t0, t1, ksi0, ctrl_mode, uMan, u_min, u_max, Nruns, is_print_sim_step, is_log_data, is_playback, r0, level_target = self.pars
        
        # Store some parameters for later use
        self.t0 = t0
        self.ksi0 = ksi0
        self.t1 = t1
        self.ctrl_mode = ctrl_mode
        self.uMan = uMan
        self.Nruns = Nruns
        self.is_print_sim_step = is_print_sim_step
        self.is_log_data = is_log_data
        self.is_playback = is_playback
        
        self.level_target = level_target
        
        h1_0 = x0[0]
        h2_0 = x0[1]
        p0 = u0
        
        plt.close('all')
     
        self.fig_sim = plt.figure(figsize=(10,10))    
            
        # h1, h2 plot
        self.axs_sol = self.fig_sim.add_subplot(221, autoscale_on=False, xlim=(t0,t1), ylim=( -2, 2 ), xlabel='t [s]', title='Pause - space, q - quit, click - data cursor')
        self.axs_sol.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.axs_sol.plot([t0, t1], [level_target[0], level_target[0]], 'b--', lw=0.75)   # Help line (target)
        self.axs_sol.plot([t0, t1], [level_target[1], level_target[1]], 'r--', lw=0.75)   # Help line (target)
        self.line_h1, = self.axs_sol.plot(t0, h1_0, 'b-', lw=0.5, label=r'$h_1$')
        self.line_h2, = self.axs_sol.plot(t0, h2_0, 'r-', lw=0.5, label=r'$h_2$') 
        self.axs_sol.legend(fancybox=True, loc='upper right')
        self.axs_sol.format_coord = lambda x,y: '%2.2f, %2.2f' % (x,y)
        
        # Cost
        if is_playback:
            r = r0
        else:
            y0 = self.sys.out(x0)
            r = self.ctrl_benchmarking.rcost(y0, u0)
        
        self.axs_cost = self.fig_sim.add_subplot(223, autoscale_on=False, xlim=(t0,t1), ylim=(0, 1e4*r), yscale='symlog', xlabel='t [s]')
        
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost = 0)
        self.text_icost_handle = self.fig_sim.text(0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')
        self.line_rcost, = self.axs_cost.plot(t0, r, 'r-', lw=0.5, label='r')
        self.line_icost, = self.axs_cost.plot(t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Control
        self.axs_ctrl = self.fig_sim.add_subplot(222, autoscale_on=False, xlim=(t0,t1), ylim=(u_min-0.1, u_max+0.1), xlabel='t [s]')
        self.axs_ctrl.plot([t0, t1], [0, 0], 'k--', lw=0.75)   # Help line
        self.line_ctrl, = self.axs_ctrl.plot(t0, p0, lw=0.5, label='p') 
        self.axs_cost.legend(fancybox=True, loc='upper right')
        
        # Pack all lines together
        cLines = namedtuple('lines', ['line_h1', 'line_h2', 'line_rcost', 'line_icost', 'line_ctrl'])
        self.lines = cLines(line_h1=self.line_h1,
                            line_h2=self.line_h2,
                            line_rcost=self.line_rcost,
                            line_icost=self.line_icost,
                            line_ctrl=self.line_ctrl)
    
        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)         
                
    def set_sim_data(self, ts, h1s, h2s, ps, rs, icosts):
        """
        This function is needed for playback purposes when simulation data were generated elsewhere.
        It feeds data into the animator from outside.
        The simulation step counter ``curr_step`` is reset accordingly.

        """   
        self.ts, self.h1s, self.h2s, self.ps = ts, h1s, h2s, ps
        self.rs, self.icosts = rs, icosts
        self.curr_step = 0
        
    def upd_sim_data_row(self):
        self.t = self.ts[self.curr_step]
        self.ksi = np.array([self.h1s[self.curr_step], self.h2s[self.curr_step]])
        self.r = self.rs[self.curr_step]
        self.icost = self.icosts[self.curr_step]
        self.u = np.array([self.ps[self.curr_step]])
        
        self.curr_step = self.curr_step + 1
    
    def init_anim(self):
        x0, *_ = self.pars      
        
        self.run_curr = 1
        self.datafile_curr = self.datafiles[0]
    
    def animate(self, k):
        
        if self.is_playback:
            self.upd_sim_data_row()
            t = self.t
            ksi = self.ksi
            u = self.u
            r = self.r
            icost = self.icost        
            
        else:
            self.simulator.sim_step()
            
            t, x, y, ksi = self.simulator.get_sim_step_data()
            
            u = self.ctrl_selector(t, y, self.uMan, self.ctrl_nominal, self.ctrl_benchmarking, self.ctrl_mode)
        
            self.sys.receive_action(u)
            self.ctrl_benchmarking.receive_sys_state(self.sys._x) 
            self.ctrl_benchmarking.upd_icost(y, u)
            
            r = self.ctrl_benchmarking.rcost(y, u)
            icost = self.ctrl_benchmarking.icost_val
        
        h1 = ksi[0]
        h2 = ksi[1]
        p = u

        if self.is_print_sim_step:
            self.logger.print_sim_step(t, h1, h2, p, r, icost)
            
        if self.is_log_data:
            self.logger.log_data_row(self.datafile_curr, t, h1, h2, p, r, icost)
        
        # # Solution
        upd_line(self.line_h1, t, h1)
        upd_line(self.line_h2, t, h2)
    
        # Cost
        upd_line(self.line_rcost, t, r)
        upd_line(self.line_icost, t, icost)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'.format(icost = icost)
        upd_text(self.text_icost_handle, text_icost)
        
        # Control
        upd_line(self.line_ctrl, t, p)
    
        # Run done
        if t >= self.t1:  
            if self.is_print_sim_step:
                    print('.....................................Run {run:2d} done.....................................'.format(run = self.run_curr))
                
            self.run_curr += 1
            
            if self.run_curr > self.Nruns:
                print('Animation done...')
                self.stop_anm()
                return
            
            if self.is_log_data:
                self.datafile_curr = self.datafiles[self.run_curr-1]
            
            # Reset simulator
            self.simulator.reset()
            
            # Reset controller
            if self.ctrl_mode > 0:
                self.ctrl_benchmarking.reset(self.t0)
            else:
                self.ctrl_nominal.reset(self.t0)
            
            icost = 0     
            
            reset_line(self.line_h1)
            reset_line(self.line_h1)
            reset_line(self.line_ctrl)
            reset_line(self.line_rcost)
            reset_line(self.line_icost)
            
            # for item in self.lines:
            #     if item != self.line_traj:
            #         if isinstance(item, list):
            #             for subitem in item:
            #                 self.reset_line(subitem)
            #                 print('line reset')
            #         else:
            #             self.reset_line(item)
    
            # upd_line(self.line_h1, np.nan)                