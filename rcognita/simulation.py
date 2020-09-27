# general imports
import os
import pathlib
import warnings
import sys
import itertools
import statistics
import pprint
from IPython.display import clear_output


# scipy
import scipy as sp
from scipy import integrate

# numpy
import numpy as np
from numpy.random import rand
from numpy.random import randn
import numpy.linalg as la

# matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

# rcognita
from . import utilities

# other
from mpldatacursor import datacursor
from tabulate import tabulate


class Simulation(utilities.Generic):
    """class to create and run simulation.

    Parameters
    ----------

    system : object of type `System` class

    controller : object of type `Controller` class

    nominal_ctrl : object of type `NominalController` class

    a_tol : float
        * ODE solver sensitivity hyperparameter

    r_tol : float
        * ODE solver sensitivity hyperparameter

    x_min : int
        * minimum x limit of graph

    x_max : int
        * maximum x limit of graph

    y_min : int
        * minimum y limit of graph

    y_max : int
        * minimum y limit of graph

    """

    def __init__(self,
                 system,
                 controller,
                 nominal_ctrl,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10):
        """

        CONTROL FLOW LOGIC: IGNORE

        """

        if hasattr(controller, '__len__') and hasattr(nominal_ctrl, '__len__'):
            self.num_controllers = len(controller)
            self.num_nom_controllers = len(nominal_ctrl)

            if self.num_nom_controllers > 1 and self.num_controllers == 1 and system.num_controllers == 1:
                self.nominal_ctrl = nominal_ctrl[0]
                self.controller = controller[0]

            elif self.num_nom_controllers > 1 and self.num_controllers > 1 and system.num_controllers == 1:
                self.error_message = "You forgot to call `add_bots` method on the System object. Make sure you call add_bots for as many controllers and nominal controllers that are being passed to the Simulation class."

            elif self.num_nom_controllers > 1 and self.num_controllers > 1 and system.num_controllers > 1:

                if self.num_nom_controllers == self.num_controllers == system.num_controllers > 1:
                    self.nominal_ctrlers = nominal_ctrl
                    self.controllers = controller

                else:
                    self.error_message = "Please pass the same number of controllers, nominal controllers, and registered system controllers (via add_bots method) to the Simulation class."

            elif self.num_nom_controllers == 1 and self.num_controllers > 1 and system.num_controllers == 1:
                self.controller = controller[0]
                self.nominal_ctrl = nominal_ctrl[0]

        elif hasattr(controller, '__len__') or hasattr(nominal_ctrl, '__len__'):
            self.error_message = "Please create and pass the same number of controller objects and nominal controller objects to the simulation class."

        else:
            self.controller = controller
            self.nominal_ctrl = nominal_ctrl
            self.num_controllers = 1

            if system.num_controllers > 1:
                self.error_message = "Error: you called system.add_bots() but did not pass the same number of Controller objects when instantiating the Simulation class. Please create and pass the same number of controller objects and nominal controller objects to the simulation class as you call add_bots()."

        if hasattr(self, 'error_message'):
            print(self.error_message)
            return None

        else:
            """

            VISUALIZATION PARAMS

            """

            # x and y limits of scatter plot. Used so far rather for visualization
            # only, but may be integrated into the actor as constraints
            self.x_min = x_min
            self.x_max = x_max
            self.y_min = y_min
            self.y_max = y_max

            """

            CONTROLLER AND SYSTEM PARAMS

            """
            self.system = system

            # control constraints
            self.dim_input = system.dim_input
            self.f_min = system.f_min
            self.f_max = system.f_max
            self.f_man = system.f_man
            self.m_man = system.m_man
            self.m_min = system.m_min
            self.m_max = system.m_max

            self.control_bounds = system.control_bounds

            closed_loop = system.closed_loop

            if self.num_controllers == 1:
                self.sample_time, self.ctrl_mode, self.t1, self.t0 = self._get_controller_info(
                    self.controller)

                self.system_state, self.full_state, self.alpha, self.initial_x, self.initial_y = self._get_system_info(
                    system)

                self.simulator = sp.integrate.RK45(closed_loop,
                                                   self.t0,
                                                   self.full_state,
                                                   self.t1,
                                                   max_step=self.sample_time / 2,
                                                   first_step=1e-6,
                                                   atol=a_tol,
                                                   rtol=r_tol)

            elif self.num_controllers > 1:
                self.sample_times, self.ctrl_modes, self.t1s, self.t0 = self._get_controller_info(
                    self.controllers, multi=True)

                self.system_states, self.full_states, self.alphas, self.initial_xs, self.initial_ys, self.u0s = self._get_system_info(
                    system, multi=True)

                self.simulators = []

                for i in range(self.num_controllers):
                    self.system.set_multi_sim(i)

                    simulator = sp.integrate.RK45(closed_loop,
                                                  self.t0,
                                                  self.full_states[i],
                                                  self.t1s[i],
                                                  max_step=self.sample_times[
                                                      i] / 2,
                                                  first_step=1e-6,
                                                  atol=a_tol,
                                                  rtol=r_tol)

                    self.simulators.append(simulator)

    def _get_controller_info(self, controller, multi=False):
        # if we have a single controller
        if multi is False:
            sample_time = controller.sample_time
            ctrl_mode = controller.ctrl_mode
            t1 = controller.t1

            return sample_time, ctrl_mode, t1, 0

        # if we have multiple controllers
        else:
            controllers = controller
            num_controllers = len(controllers)

            sample_times = []
            ctrl_modes = []
            t1s = []

            for controller in controllers:
                sample_times.append(controller.sample_time)
                ctrl_modes.append(controller.ctrl_mode)
                t1s.append(controller.t1)

            return sample_times, ctrl_modes, t1s, 0

    def _get_system_info(self, system, multi=False):
        if multi is False:
            system_state = system.system_state
            full_state = system.full_state
            alpha = system.initial_alpha
            initial_x = system.initial_x
            initial_y = system.initial_y

            return system_state, full_state, alpha, initial_x, initial_y

        else:
            system_states = system.system_state
            full_states = system.full_state
            alphas = system.alpha
            initial_xs = system.system_state[:, 0]
            initial_ys = system.system_state[:, 1]
            u0s = system.u0
            try:
                q0s = system.q0
            except:
                pass

            return system_states, full_states, alphas, initial_xs, initial_ys, u0s

    def _collect_print_statistics(self, t, x_coord, y_coord, alpha, v, omega, icost, r, u, l2_norm, mid=None):
        if mid is None:
            self.statistics['running_cost'][0].append(r)
            self.statistics['velocity'][0].append(v)
            self.statistics['alpha'][0].append(alpha)
            self.statistics['l2_norm'][0] = l2_norm
        else:
            self.statistics['running_cost'][mid].append(r)
            self.statistics['velocity'][mid].append(v)
            self.statistics['alpha'][mid].append(alpha)
            self.statistics['l2_norm'][mid] = l2_norm

        self.l2_norm

        if self.print_statistics_at_step:
            if mid is not None:
                print(f"Controller {mid+1}: run {self.current_run[mid]}")
            else:
                print(f"Controller 1: run {self.current_run}")

            self._print_sim_step(t, x_coord, y_coord, alpha, v, omega, icost, u)

            if self.print_inline:
                try:
                    __IPYTHON__
                    clear_output(wait=True)
                except:
                    pass


    def _ctrl_selector(self, t, y, uMan, nominal_ctrl, controller, mode):
        """
        Main interface for different agents

        """

        if mode == 0:  # Manual control
            u = uMan
        elif mode == -1:  # Nominal controller
            u = nominal_ctrl.compute_action(t, y)
        elif mode > 0:  # Optimal controller
            u = controller.compute_action(t, y)

        return u

    def _create_figure_plots(self, system, controller, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        y0 = system.get_curr_state(self.system_state)
        self.alpha = self.alpha / 2 / np.pi

        plt.close('all')

        self.sim_fig = plt.figure(figsize=(fig_width, fig_height))

        """

        Simulation subplot

        """
        self.xy_plane_axes = self.sim_fig.add_subplot(221,
                                                      autoscale_on=False,
                                                      xlim=(self.x_min,
                                                            self.x_max),
                                                      ylim=(self.y_min,
                                                            self.y_max),
                                                      xlabel='x [m]',
                                                      ylabel='y [m]',
                                                      title=' Simulation: \n Pause - space, q - quit, click - data cursor')

        self.xy_plane_axes.set_aspect('equal', adjustable='box')

        self.xy_plane_axes.plot([self.x_min, self.x_max], [
            0, 0], 'k--', lw=0.75)   # x-axis

        self.xy_plane_axes.plot([0, 0], [self.y_min, self.y_max],
                                'k--', lw=0.75)   # y-axis

        self.traj_line, = self.xy_plane_axes.plot(
            self.initial_x, self.initial_y, 'b--', lw=0.5)

        self.robot_marker = utilities._pltMarker(angle=self.alpha)

        text_time = 't = {time:2.3f}'.format(time=self.t0)

        self.text_time_handle = self.xy_plane_axes.text(0.05, 0.95,
                                                        text_time,
                                                        horizontalalignment='left',
                                                        verticalalignment='center',
                                                        transform=self.xy_plane_axes.transAxes)

        self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Proximity subplot

        """
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')

        self.sol_axes.title.set_text('Proximity-to-Target')

        self.sol_axes.plot([self.t0, self.t1], [0, 0],
                           'k--', lw=0.75)   # Help line

        self.norm_line, = self.sol_axes.plot(self.t0, la.norm(
            [self.initial_x, self.initial_y]), 'b-', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')

        self.alpha_line, = self.sol_axes.plot(
            self.t0, self.alpha, 'r-', lw=0.5, label=r'$\alpha$ [rad]')

        self.sol_axes.legend(fancybox=True, loc='upper right')

        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Cost subplot

        """
        self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            0, 1e4 * controller.running_cost(y0, system.u0)), yscale='symlog', xlabel='t [s]')

        self.cost_axes.title.set_text('Cost')

        r = controller.running_cost(y0, system.u0)
        text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=0)

        self.text_icost_handle = self.sim_fig.text(
            0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')

        self.r_cost_line, = self.cost_axes.plot(
            self.t0, r, 'r-', lw=0.5, label='r')

        self.i_cost_line, = self.cost_axes.plot(
            self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')

        self.cost_axes.legend(fancybox=True, loc='upper right')

        """

        Control subplot

        """
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([system.f_min, system.m_min]), 1.1 * np.max([system.f_max, system.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line

        self.ctrl_lines = self.ctrlAxs.plot(
            self.t0, utilities._toColVec(system.u0).T, lw=0.5)

        self.ctrlAxs.legend(
            iter(self.ctrl_lines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        # Pack all lines together
        self.lines = [self.traj_line, self.norm_line, self.alpha_line,
                      self.r_cost_line, self.i_cost_line, self.ctrl_lines]

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

        return self.sim_fig

    def _create_figure_plots_multi(self, system, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        self.colors = ['b', 'r', 'g', 'o']
        self.color_pairs = [['b', 'g'], ['r', 'm'], ['g', 'y'], ['o', 'teal']]

        y0_list = []

        for system_state in self.system_states:
            y0 = system.get_curr_state(system_state)
            y0_list.append(y0)

        self.alphas = self.alphas / 2 / np.pi

        plt.close('all')

        self.sim_fig = plt.figure(figsize=(fig_width, fig_height))

        """

        Simulation subplot

        """
        self.xy_plane_axes = self.sim_fig.add_subplot(221,
                                                      autoscale_on=False,
                                                      xlim=(self.x_min,
                                                            self.x_max),
                                                      ylim=(self.y_min,
                                                            self.y_max),
                                                      xlabel='x [m]',
                                                      ylabel='y [m]',
                                                      title=' Simulation: \n Pause - space, q - quit, click - data cursor')

        self.xy_plane_axes.set_aspect('equal', adjustable='box')
        self.xy_plane_axes.plot([self.x_min, self.x_max], [
                                0, 0], 'k--', lw=0.75)   # x-axis
        self.xy_plane_axes.plot(
            [0, 0], [self.y_min, self.y_max], 'k--', lw=0.75)   # y-axis

        self.traj_lines = []
        self.robot_markers = []
        self.text_time_handles = []
        self.run_handles = []
        text_time = 't = {time:2.3f}'.format(time=self.t0)
        time_positions = [[0.05, 0.95], [0.70, 0.95], [0.05, 0.10], [0.70, 0.10]]
        run_positions = [[0.15, 0.90], [0.80, 0.90], [0.15, 0.13], [0.80, 0.13]]

        for i in range(self.num_controllers):
            self.traj_line, = self.xy_plane_axes.plot(self.initial_xs[i], self.initial_ys[i], f'{self.colors[i]}--', lw=0.5, c=self.colors[i])

            self.robot_marker = utilities._pltMarker(angle=self.alphas[i])

            self.run_handle = self.xy_plane_axes.text(run_positions[i][0], run_positions[i][1], f"Run: 0", horizontalalignment='center', transform=self.xy_plane_axes.transAxes)

            self.text_time_handle = self.xy_plane_axes.text(time_positions[i][0], time_positions[i][
                                                            1], text_time, horizontalalignment='left', verticalalignment='center', transform=self.xy_plane_axes.transAxes)

            self.traj_lines.append(self.traj_line)
            self.robot_markers.append(self.robot_marker)
            self.text_time_handles.append(self.text_time_handle)
            self.run_handles.append(self.run_handle)

        self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Proximity subplot

        """
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')

        self.sol_axes.title.set_text('Proximity-to-Target')

        self.sol_axes.plot([self.t0, max(self.t1s)], [0, 0],
                           'k--', lw=0.75)   # Help line

        # logic for multiple controllers
        self.norm_lines = []
        self.alpha_lines = []

        for i in range(self.num_controllers):
            self.norm_line, = self.sol_axes.plot(self.t0, la.norm([self.initial_xs[i], self.initial_ys[i]]), f'{self.color_pairs[i][0]}--', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')

            self.alpha_line, = self.sol_axes.plot(self.t0, self.alphas[i], f'{self.color_pairs[i][1]}--', lw=0.5, label=r'$\alpha$ [rad]')

            self.norm_lines.append(self.norm_line)
            self.alpha_lines.append(self.alpha_line)

        self.sol_axes.legend(fancybox=True, loc='upper right')

        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)

        """

        Cost subplot

        """

        self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            0, 1e4 * self.controllers[0].running_cost(y0_list[0], self.u0s[0])), yscale='symlog', xlabel='t [s]')

        self.cost_axes.title.set_text('Cost')

        self.text_icost_handles = []
        self.r_cost_lines = []
        self.i_cost_lines = []

        text_positions = [[0.05, 0.50], [
            0.05, 0.48], [0.50, 0.50], [0.50, 0.48]]

        for i in range(self.num_controllers):
            r = self.controllers[i].running_cost(y0_list[i], self.u0s[i])
            text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(
                icost=0)

            self.text_icost_handle = self.sim_fig.text(text_positions[i][0], text_positions[
                                                       i][1], text_icost, horizontalalignment='left', verticalalignment='center')

            self.r_cost_line, = self.cost_axes.plot(
                self.t0, r, f'{self.color_pairs[i][0]}-', lw=0.5, label='r')

            self.i_cost_line, = self.cost_axes.plot(
                self.t0, 0, f'{self.color_pairs[i][1]}-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')

            self.text_icost_handles.append(self.text_icost_handle)
            self.r_cost_lines.append(self.r_cost_line)
            self.i_cost_lines.append(self.i_cost_line)

            self.cost_axes.legend(fancybox=True, loc='upper right')

        """

        Control subplot

        """
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, max(self.t1s)), ylim=(
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, max(self.t1s)], [0, 0],
                          'k--', lw=0.75)   # Help line

        # logic for multiple controllers
        self.all_ctrl_lines = []

        clabels = ['F [N]', 'M [Nm]']

        for i in range(self.num_controllers):
            u = np.expand_dims(self.u0s[i], axis=0)
            self.ctrl_lines = self.ctrlAxs.plot(
                self.t0, u, lw=0.5, label=clabels)

            self.all_ctrl_lines.append(self.ctrl_lines)

        handles, labels = self.ctrlAxs.get_legend_handles_labels()

        # clabels = clabels[::-1]
        new_labels = [clabels] * self.num_controllers
        new_labels = list(itertools.chain.from_iterable(new_labels))

        labels = new_labels

        self.ctrlAxs.legend(handles, labels, fancybox=True, loc='upper right')

        self.all_lines = [[] for i in range(self.num_controllers)]

        for i in range(self.num_controllers):
            self.all_lines[i].extend([self.traj_lines[i], self.norm_lines[i], self.alpha_lines[
                                     i], self.r_cost_lines[i], self.i_cost_lines[i], self.all_ctrl_lines[i]])

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for line in self.all_lines:
            for item in line:
                if isinstance(item, list):
                    for subitem in item:
                        datacursor(subitem)
                else:
                    datacursor(item)

        return self.sim_fig

    def _graceful_exit(self, plt_close=True):
        if plt_close is True:
            plt.close('all')

        # graceful exit from Jupyter notebook
        try:
            __IPYTHON__

        # graceful exit from terminal
        except NameError:
            if plt_close is True:
                print("Program exit")
                sys.exit()
            else:
                pass

    def _initialize_figure(self):
        self.scatter_plots = []

        if self.num_controllers > 1:
            self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs, self.initial_ys, s=400, c=self.colors[:self.num_controllers], marker=self.robot_marker.marker)
            self.scatter_plots.append(self.sol_scatter)

            if self.show_annotations:
                self.annotations = []
                for i in range(self.num_controllers):
                    self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')
                    self.annotations.append(self.annotation)

        else:
            self.sol_scatter = self.xy_plane_axes.scatter(
                self.initial_x, self.initial_y, marker=self.robot_marker.marker, s=400, c='b')

        return self.sol_scatter,

    def _log_data_row(self, dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u):
        with open(dataFile, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow([t, xCoord, yCoord, alpha,
                             v, omega, icost, u[0], u[1]])

    def _log_data(self, n_runs, save=False):
        dataFiles = [None] * n_runs

        if save:
            cwd = os.getcwd()
            datafolder = '/data'
            dataFolder_path = cwd + datafolder

            # create data dir
            pathlib.Path(dataFolder_path).mkdir(parents=True, exist_ok=True)

            date = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%Hh%Mm%Ss")
            dataFiles = [None] * n_runs
            for k in range(0, n_runs):
                dataFiles[k] = dataFolder_path + '/RLsim__' + date + \
                    '__' + time + '__run{run:02d}.csv'.format(run=k + 1)
                with open(dataFiles[k], 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]',
                                     'v [m/s]', 'omega [rad/s]', 'run_cost', 'F [N]', 'M [N m]'])

        return dataFiles

    def _on_key_press(self, event, anm, args):
        if event.key == ' ':
            if anm.running is True:
                anm.event_source.stop()
                anm.running = False

            elif anm.running is False:
                anm.event_source.start()
                anm.running = True

        elif event.key == 'q':
            _, controller, nominal_ctrl, simulator, _ = args

            if self.num_controllers > 1:
                for i in range(self.num_controllers):
                    self._reset_sim(
                        controller[i], nominal_ctrl, simulator[i], i)

            else:
                self._reset_sim(controller, nominal_ctrl, simulator)

            self._graceful_exit()

    def print_sim_summary_stats(self):
        self.mean_rc = []  # mean rc
        self.var_rc = []  # var rc
        self.mean_velocity = []  # mean vel
        self.var_velocity = []  # var vel
        self.var_alpha = []  # var alpha

        if self.num_controllers > 1:
            n_runs = self.n_runs[0]
        else:
            n_runs = self.n_runs

        for mid in range(self.num_controllers):
            self.mean_rc.append(round(statistics.mean(
                self.statistics['running_cost'][mid]), 2))
            self.var_rc.append(round(statistics.variance(
                self.statistics['running_cost'][mid]), 2))
            self.mean_velocity.append(
                round(statistics.mean(self.statistics['velocity'][mid]), 2))
            self.var_velocity.append(
                round(statistics.variance(self.statistics['velocity'][mid]), 2))
            self.var_alpha.append(
                round(statistics.variance(self.statistics['alpha'][mid]), 2))

        print(f"Total runs for each controller: {n_runs}")
        
        for mid in range(self.num_controllers):
            print(f"""Statistics for controller {mid+1}:
            - Mean of running cost: {self.mean_rc[mid]}
            - Variance of running cost: {self.var_rc[mid]}
            - Mean of velocity: {self.mean_velocity[mid]}
            - Variance of velocity: {self.var_velocity[mid]}
            - Variance of turning angle: {self.var_alpha[mid]}
            - Final L2-norm: {round(self.statistics['l2_norm'][mid],2)}
                """)

    def _print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180

        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]',
                     'v [m/s]', 'omega [rad/s]', 'run_cost', 'F [N]', 'M [N m]']
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f',
                     '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat,
                         headers='firstrow', tablefmt='grid')

        print(table)

    def _reset_all_lines(self, lines):
        for line in lines:
            if isinstance(line, list) is False:
                self._reset_line(line)
            else:
                self._reset_all_lines(line)

    def _reset_line(self, line):
        line.set_data([], [])

    def _reset_sim(self, controller, nominal_ctrl, simulator, mid=None):
        if self.print_statistics_at_step:
            if mid is not None:
                print(f'........Controller {mid+1}: Run #{self.current_runs[mid]-1} done........')
            else:
                print(f'........Run {self.current_run-1} done........')

        if self.is_log_data:
            self.current_data_file = self.data_files[self.current_run - 1]

        # Reset simulator
        simulator.status = 'running'
        simulator.t = self.t0

        if mid is not None:
            simulator.y = self.full_states[mid]

        else:
            simulator.y = self.full_state

        # Reset controller
        if controller.ctrl_mode > 0:
            controller.reset(self.t0)
        else:
            nominal_ctrl.reset(self.t0)

        if self.is_visualization:
            if mid is None:
                if self.current_run <= self.n_runs:
                    self._reset_all_lines(self.lines)

            else:
                if self.current_runs[mid] <= self.n_runs[mid]:
                    self._reset_all_lines(self.all_lines[mid])

    def _run_animation(self, system, controller, nominal_ctrl, simulator, fig_width, fig_height, multi_controllers=False):
        animate = True
        self.exit_animation = False

        if multi_controllers is True:
            controllers = controller
            simulators = simulator
            nominal_ctrlers = nominal_ctrl

            self.sim_fig = self._create_figure_plots_multi(system,
                fig_width, fig_height)
            fargs = (system, controllers, nominal_ctrlers, simulators, animate)

            self.anm = animation.FuncAnimation(self.sim_fig,
                                               self._wrapper_take_steps_multi,
                                               fargs=fargs,
                                               init_func=self._initialize_figure,
                                               interval=1,
                                               blit=False)

        else:
            self.sim_fig = self._create_figure_plots(
                system, controller, fig_width, fig_height)
            fargs = (system, controller, nominal_ctrl, simulator, animate)

            self.anm = animation.FuncAnimation(self.sim_fig,
                                               self._wrapper_take_steps,
                                               fargs=fargs,
                                               init_func=self._initialize_figure,
                                               interval=1,
                                               blit=False)

        self.anm.running = True
        self.sim_fig.canvas.mpl_connect(
            'key_press_event', lambda event: self._on_key_press(event, self.anm, fargs))
        self.sim_fig.tight_layout()
        plt.show()

    def run_simulation(self, 
                        n_runs=1, 
                        is_visualization=True, 
                        fig_width=8, 
                        fig_height=8, 
                        close_plt_on_finish=True, 
                        show_annotations=False, 
                        print_summary_stats=False,
                        print_statistics_at_step=False,
                        print_inline=False,
                        is_log_data=False):
        """
        n_runs : int
            * number of episodes

        is_visualization : bool
            * visualize simulation?

        fig_width : int
            * if visualizing: width of figure

        fig_height : int
            * if visualizing: height of figure

        close_plt_on_finish : bool
            * if visualizing: close plots automatically on finishing simulation

        show_annotations : bool
            * if visualizing: annotate controller/agent for identification

        print_summary_stats : bool
            * print summary statistics after simulation
        
        is_log_data : bool
            * log data to local drive?

        print_statistics_at_step : bool
            * print results of simulation?
        """

        try:
            if hasattr(self, 'error_message') is False:
                self.is_log_data = is_log_data
                self.is_visualization = is_visualization
                self.print_statistics_at_step = print_statistics_at_step
                self.print_inline = print_inline
                self.print_summary_stats = print_summary_stats
                self.statistics = {'running_cost': {}, 'velocity': {}, 'alpha': {}, 'l2_norm': {}}
                # self.current_data_file = data_files[0]
                self.close_plt_on_finish = close_plt_on_finish
                self.data_files = self._log_data(n_runs, save=self.is_log_data)
                self.show_annotations = show_annotations

                if self.is_visualization is False:
                    self.close_plt_on_finish = False
                    self.show_annotations = False
                    self.print_summary_stats = True

                if self.print_statistics_at_step:
                    warnings.filterwarnings('ignore')

                for i in range(self.num_controllers):
                    self.statistics['running_cost'].setdefault(i, [])
                    self.statistics['velocity'].setdefault(i, [])
                    self.statistics['alpha'].setdefault(i, [])
                    self.statistics['l2_norm'].setdefault(i, 0)

                if self.num_controllers > 1:
                    self.current_runs = np.ones(self.num_controllers, dtype=np.int64)
                    self.n_runs = np.array([n_runs] * self.num_controllers)
                    self.keep_stepping = np.ones((self.num_controllers), dtype=bool)
                    self.t_elapsed = np.zeros(self.num_controllers)

                else:
                    self.current_run = 1
                    self.n_runs = n_runs


                # IF NOT VISUALIZING TRAINING
                if self.is_visualization is False:
                    if self.num_controllers > 1:
                        self._wrapper_take_steps_multi_no_viz(self.system, self.controllers, self.nominal_ctrlers, self.simulators)

                    else:
                        self._wrapper_take_steps_no_viz(self.system, self.controller, self.nominal_ctrl, self.simulator)

                else:
                    if self.num_controllers > 1:
                        self._run_animation(self.system,
                                            self.controllers,
                                            self.nominal_ctrlers,
                                            self.simulators,
                                            fig_width,
                                            fig_height,
                                            multi_controllers=True)
                    else:
                        self._run_animation(
                            self.system, self.controller, self.nominal_ctrl, self.simulator, fig_width, fig_height)

            else:
                pass
        except KeyboardInterrupt:
            print("Cancelled.")

    def _take_step(self, system, controller, nominal_ctrl, simulator, animate=False):
        simulator.step()

        t = simulator.t
        full_state = simulator.y

        system_state = system.system_state
        y = system.get_curr_state(system_state)

        u = self._ctrl_selector(
            t, y, system.u_man, nominal_ctrl, controller, controller.ctrl_mode)

        system.set_latest_action(u)

        controller.record_sys_state(system_state)
        controller.update_icost(y, u)

        x_coord = full_state[0]
        y_coord = full_state[1]
        alpha = full_state[2]
        v = full_state[3]
        omega = full_state[4]

        icost = controller.i_cost_val
        alpha_deg = alpha / np.pi * 180
        r = controller.running_cost(y, u)
        text_time = 't = {time:2.3f}'.format(time=t)

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        self._collect_print_statistics(t, x_coord, y_coord, alpha, v, omega, icost, r, u, self.l2_norm)

        if self.is_log_data:
            self._log_data_row(self.current_data_file, t, x_coord,
                               y_coord, alpha, v, omega, icost.val, u)

        if animate:
            self._update_all_lines(
                text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, self.l2_norm)

        return t

    def _take_step_multi(self, mid, system, controller, nominal_ctrl, simulator, animate=False):
        system.set_multi_sim(mid)
        simulator.step()

        t = simulator.t
        full_state = simulator.y

        system_state = self.system_states[mid]

        y = system.get_curr_state(system_state)

        u = self._ctrl_selector(
            t, y, system.u_man, nominal_ctrl, controller, controller.ctrl_mode)

        system.set_latest_action(u, mid)

        controller.record_sys_state(system_state)
        controller.update_icost(y, u)

        x_coord = full_state[0]
        y_coord = full_state[1]
        alpha = full_state[2]
        v = full_state[3]
        omega = full_state[4]

        icost = controller.i_cost_val

        alpha_deg = alpha / np.pi * 180
        r = controller.running_cost(y, u)
        text_time = 't = {time:2.3f}'.format(time=t)

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        self._collect_print_statistics(t, x_coord, y_coord, alpha, v, omega, icost, r, u, self.l2_norm, mid)
        
        if self.is_log_data:
            self._log_data_row(self.current_data_file, t, x_coord,
                               y_coord, alpha, v, omega, icost.val, u)

        if animate:
            self._update_all_lines_multi(text_time, full_state, alpha_deg,
                                         x_coord, y_coord, t, alpha, r, icost, u, self.l2_norm, mid)

        return t, x_coord, y_coord

    def _update_line(self, line, new_x, new_y):
        line.set_xdata(np.append(line.get_xdata(), new_x))
        line.set_ydata(np.append(line.get_ydata(), new_y))

    def _update_text(self, text_handle, new_text):
        text_handle.set_text(new_text)

    def _update_all_lines(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, l2_norm):
        """
        Update lines on all scatter plots
        """
        self._update_text(self.text_time_handle, text_time)

        # Update the robot's track on the plot
        self._update_line(self.traj_line, *full_state[:2])

        self.robot_marker.rotate(alpha_deg)    # Rotate the robot on the plot
        self.sol_scatter.remove()
        self.sol_scatter = self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_marker.marker, s=400, c='b')

        # Solution
        self._update_line(self.norm_line, t, l2_norm)
        self._update_line(self.alpha_line, t, alpha)

        # Cost
        self._update_line(self.r_cost_line, t, r)
        self._update_line(self.i_cost_line, t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handle, text_icost)

        # Control
        for (line, uSingle) in zip(self.ctrl_lines, u):
            self._update_line(line, t, uSingle)

    def _update_all_lines_multi(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, l2_norm, mid):
        """
        Update lines on all scatter plots
        """
        self._update_text(self.text_time_handles[mid], text_time)
        self._update_text(self.run_handles[mid], f"A{mid+1}, run: {self.current_runs[mid]}")

        # Update the robot's track on the plot
        self._update_line(self.traj_lines[mid], x_coord, y_coord)

        # Rotate the robot on the plot
        self.robot_markers[mid].rotate(alpha_deg)

        self.scatter_plots.append(self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_markers[mid].marker, s=400, c=self.colors[mid]))

        # Solution
        self._update_line(self.norm_lines[mid], t, l2_norm)
        self._update_line(self.alpha_lines[mid], t, alpha)

        # Cost
        self._update_line(self.r_cost_lines[mid], t, r)
        self._update_line(self.i_cost_lines[mid], t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handles[mid], text_icost)

        # Control
        for (line, uSingle) in zip(self.all_ctrl_lines[mid], u):
            self._update_line(line, t, uSingle)

    def _wrapper_take_steps(self, k, *args):
        _, controller, nominal_ctrl, simulator, _ = args
        t = simulator.t

        if self.current_run <= self.n_runs:
            if t < self.t1:
                t = self._take_step(*args)

            else:
                self.current_run += 1
                self.t_elapsed = t
                self._reset_sim(controller, nominal_ctrl, simulator)

        elif self.current_run > self.n_runs and self.exit_animation is False:
            if self.print_summary_stats is True:
                self.print_sim_summary_stats()

            self.anm.running = False
            self.exit_animation = True

        else:
            if self.close_plt_on_finish is True:
                self._graceful_exit()

            elif self.close_plt_on_finish is False:
                self._graceful_exit(plt_close=False)

    def _wrapper_take_steps_multi(self, k, *args):
        system, controllers, nominal_ctrlers, simulators, animate = args

        for i in range(self.num_controllers):
            if self.current_runs[i] <= self.n_runs[i]:
                self.keep_stepping[i] = True
            else:
                self.keep_stepping[i] = False

        if self.keep_stepping.any() == True:
            for scat in self.scatter_plots:
                scat.remove()

            self.scatter_plots = []

            if self.show_annotations:
                for ann in self.annotations:
                    ann.remove()

                self.annotations = []

            for i in range(self.num_controllers):
                if self.keep_stepping[i] == True:
                    t = simulators[i].t

                    if t < self.t1s[i]:
                        t, x_coord, y_coord = self._take_step_multi(
                            i, system, controllers[i], nominal_ctrlers[i], simulators[i], animate)

                        if self.show_annotations:
                            self.annotations.append(self.xy_plane_axes.annotate(f'{i+1}', xy=(x_coord + 0.5, y_coord + 0.5), color='k'))

                    else:
                        self.current_runs[i] += 1
                        self.t_elapsed[i] = t
                        self._reset_sim(controllers[i], nominal_ctrlers[
                                        i], simulators[i], i)
                else:
                    self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs[i], self.initial_ys[
                                                                  i], s=400, c=self.colors[i], marker=self.robot_markers[i].marker)

                    if self.show_annotations:
                        self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')
                    continue

        elif self.keep_stepping.all() == False and self.exit_animation is False:
            for i in range(self.num_controllers):
                self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs[i], self.initial_ys[
                                                              i], s=400, c=self.colors[i], marker=self.robot_markers[i].marker)

                if self.show_annotations:
                    self.annotation = self.xy_plane_axes.annotate(f'{i+1}', xy=(self.initial_xs[i] + 0.5, self.initial_ys[i] + 0.5), color='k')

            if self.print_summary_stats:
                self.print_sim_summary_stats()

            self.anm.running = False
            self.exit_animation = True

        else:
            self.t_elapsed = self.t1s
            if self.close_plt_on_finish is True:
                self._graceful_exit()

            elif self.close_plt_on_finish is False:
                self._graceful_exit(plt_close=False)

    def _wrapper_take_steps_no_viz(self, *args):
        system, controller, nominal_ctrl, simulator = args

        while self.current_run <= self.n_runs:
            t = simulator.t
            print(f"... Running - run {self.current_run}...")

            while t < self.t1:
                t = self._take_step(*args)

            else:
                self.current_run += 1
                self.t_elapsed = t
                self._reset_sim(controller, nominal_ctrl, simulator)

        else:
            if self.print_summary_stats is True:
                self.print_sim_summary_stats()

            self._graceful_exit()

    def _wrapper_take_steps_multi_no_viz(self, *args):
        system, controllers, nominal_ctrlers, simulators = args

        for i in range(self.num_controllers):
            t = simulators[i].t
            print(f"... Running for controller {i+1}")

            while self.current_runs[i] <= self.n_runs[i]:
                print(f"... Controller {i}, run {self.current_runs[i]}...")
                
                while t < self.t1s[i]:
                    t, x_coord, y_coord = self._take_step_multi(i, system, controllers[i], nominal_ctrlers[i], simulators[i])

                else:
                    self.current_runs[i] += 1
                    self.t_elapsed[i] = t
                    self._reset_sim(controllers[i], nominal_ctrlers[i], simulators[i], i)


        if self.print_summary_stats:
            self.print_sim_summary_stats()

        self._graceful_exit(plt_close=False)
