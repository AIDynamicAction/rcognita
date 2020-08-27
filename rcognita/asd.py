class Simulation(utilities.Generic):
    """class to create and run simulation.

    a_tol, r_tol -- sensitivity of the solverxs
    """

    def __init__(self,
                 system,
                 controller,
                 nominal_ctrl,
                 t0=0,
                 t1=60,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10,
                 is_log_data=0,
                 is_visualization=True,
                 is_print_sim_step=False):
        
        
        # start time of episode
        self.t0 = t0

        # stop time of episode
        self.t1 = t1
        
        # control constraints
        self.f_min = system.f_min
        self.f_max = system.f_max
        self.f_man = system.f_man
        self.n_man = system.n_man
        self.m_min = system.m_min
        self.m_max = system.m_max

        self.control_bounds = system.control_bounds

        if isinstance(self.controller, list) and len(self.controller) > 1:
            self.num_controllers = len(self.controller)

            self.systems = system
            self.controllers = controller
            self.nominal_ctrl = nominal_ctrl

            self.sample_times, self.ctrl_modes = self._get_controller_info(self.controllers)

            self.system_states, self.full_states, self.alphas = self._get_system_info(self.systems)

            self.simulators = []

            for pseudo_system in self.systems:
            	simulator = self._create_simulator(pseudo_system.closed_loop, self.full_states, self.t0, self.t1, self.sample_time, a_tol, r_tol)
            	self.simulators.append(simulator)

        else: 
            self.num_controllers = 1

            self.system = system
            self.controller = controller
            self.nominal_ctrl = nominal_ctrl

            self.sample_time, self.ctrl_mode = self._get_controller_info(controller)

            self.system_state, self.full_state, self.alpha = self._get_system_info(system)

    		closed_loop = system.closed_loop
            self.simulator = self._create_simulator(closed_loop, self.full_state, self.t0, self.t1, self.sample_time, a_tol, r_tol)

        """
    
        VISUALIZATION PARAMS
    
        """
        self.colors = ['b','r','g']

        # x and y limits of scatter plot. Used so far rather for visualization
        # only, but may be integrated into the actor as constraints
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # other
        self.is_log_data = is_log_data
        self.is_visualization = is_visualization
        self.is_print_sim_step = is_print_sim_step

        if self.is_print_sim_step:
            warnings.filterwarnings('ignore')

    def _get_controller_info(self, controller, multi=False):
        # if we have a single controller
        if multi is False:
            initial_x = controller.initial_x
            initial_y = controller.initial_y

            return sample_time, ctrl_mode

        # if we have multiple controllers
        else:
            controllers = controller
            num_controllers = len(controllers)
            
            sample_times = []
            ctrl_modes = []

            for controller in controllers:
                sample_times.append(controller.sample_time)
                ctrl_modes.append(controller.ctrl_mode)

            return sample_times, ctrl_modes

    def _get_system_info(self, system, multi=False):
        if multi is False:
            system_state = system.system_state
            full_state = system.full_state
            alpha = system.initial_alpha

            return system_state, full_state, alpha

        else:
            num_controllers = len(initial_x)
            initial_xs = initial_x
            initial_ys = initial_x

            system_states = np.zeros((num_controllers, dim_state))

            for i in range(num_controllers):
                system_states[i, 0] = initial_xs[i]
                system_states[i, 1] = initial_ys[i]
                system_states[i, 2] = alpha

                if is_dyn_ctrl:
                    full_states = np.concatenate((system_states, q0, u0), axis=1)
                else:
                    full_states = np.concatenate((system_states, q0), axis=1)

            alphas = system_states[:, 2]

            return system_states, full_states, alphas

    def _create_simulator(self, closed_loop, full_state, t0, t1, sample_time, a_tol, r_tol):
        simulator = sp.integrate.RK45(closed_loop,
                                           t0,
                                           full_state,
                                           t1,
                                           max_step=sample_time / 2,
                                           first_step=1e-6,
                                           atol=a_tol,
                                           rtol=r_tol)

        return simulator


    def _ctrl_selector(self, t, y, uMan, nominal_ctrl, agent, mode):
        """
        Main interface for different agents

        """

        if mode == 0:  # Manual control
            u = uMan
        elif mode == -1:  # Nominal controller
            u = nominal_ctrl.compute_action(t, y)
        elif mode > 0:  # Optimal controller
            u = agent.compute_action(t, y)

        return u

    def _create_figure_plots(self, agent, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        y0 = System.get_curr_state(self.system_state)
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
            0, 1e4 * agent.running_cost(y0, self.u0)), yscale='symlog', xlabel='t [s]')

        self.cost_axes.title.set_text('Cost')

        r = agent.running_cost(y0, self.u0)
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
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line

        self.ctrl_lines = self.ctrlAxs.plot(
            self.t0, utilities._toColVec(self.u0).T, lw=0.5)

        self.ctrlAxs.legend(
            iter(self.ctrl_lines), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        # Pack all lines together
        cLines = namedtuple('lines', [
                            'traj_line', 'norm_line', 'alpha_line', 'r_cost_line', 'i_cost_line', 'ctrl_lines'])
        self.lines = cLines(traj_line=self.traj_line,
                            norm_line=self.norm_line,
                            alpha_line=self.alpha_line,
                            r_cost_line=self.r_cost_line,
                            i_cost_line=self.i_cost_line,
                            ctrl_lines=self.ctrl_lines)

        self.current_data_file = self.data_files[0]

        # Enable data cursor
        for item in self.lines:
            if isinstance(item, list):
                for subitem in item:
                    datacursor(subitem)
            else:
                datacursor(item)

        return self.sim_fig

    def _create_figure_plots_multi(self, controller, fig_width, fig_height):
        """ returns a pyplot figure with 4 plots """

        y0_list = []
        
        for system_state in self.system_states:
            y0 = System.get_curr_state(system_state)
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

        self.xy_plane_axes.plot([0, 0], [self.y_min, self.y_max],
                                'k--', lw=0.75)   # y-axis

        if self.num_controllers > 1:
            self.traj_lines = []
            self.robot_markers = []
            self.text_time_handles = []
            text_time = 't = {time:2.3f}'.format(time=self.t0)

            for i in range(self.num_controllers):
                self.traj_line, = self.xy_plane_axes.plot(
                    self.initial_xs[i], self.initial_ys[i], f'{self.colors[i]}--', lw=0.5)

                self.robot_marker = utilities._pltMarker(angle=self.alphas[i])

                self.text_time_handle = self.xy_plane_axes.text(0.05, 0.95,
                                                                text_time,
                                                                horizontalalignment='left',
                                                                verticalalignment='center',
                                                                transform=self.xy_plane_axes.transAxes)
                
                
                self.traj_lines.append(self.traj_line)
                self.robot_markers.append(self.robot_marker)
                self.text_time_handles.append(self.text_time_handle)
            
            self.xy_plane_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)


        """
        
        Proximity subplot
        
        """
        self.sol_axes = self.sim_fig.add_subplot(222, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            2 * np.min([self.x_min, self.y_min]), 2 * np.max([self.x_max, self.y_max])), xlabel='t [s]')

        self.sol_axes.title.set_text('Proximity-to-Target')

        self.sol_axes.plot([self.t0, self.t1], [0, 0],
                           'k--', lw=0.75)   # Help line

        # logic for multiple controllers
        if self.num_controllers > 1:
            self.norm_lines = []
            self.alpha_lines = []
            
            for i in range(self.num_controllers):
                self.norm_line, = self.sol_axes.plot(self.t0, la.norm([self.initial_xs[i], self.initial_ys[i]]), f'{self.colors[i]}--', lw=0.5, label=r'$\Vert(x,y)\Vert$ [m]')
                
                self.alpha_line, = self.sol_axes.plot(self.t0, self.alphas[i], f'{self.colors[i]}--', lw=0.5, label=r'$\alpha$ [rad]')

                self.norm_lines.append(self.norm_line)
                self.alpha_lines.append(self.alpha_line)

        self.sol_axes.legend(fancybox=True, loc='upper right')

        self.sol_axes.format_coord = lambda x, y: '%2.2f, %2.2f' % (x, y)


        """
        
        Cost subplot
        
        """

        if self.num_controllers > 1:
            self.cost_axes = self.sim_fig.add_subplot(223, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(0, 1e4 * self.controllers[0].running_cost(y0_list[0], self.u0[0])), yscale='symlog', xlabel='t [s]')

            self.cost_axes.title.set_text('Cost')

            self.text_icost_handles = []
            self.r_cost_lines = []
            self.i_cost_lines = []

            for i in range(self.num_controllers):
                r = controller[i].running_cost(y0_list[i], self.u0[i])
                text_icost = r'$\int r \,\mathrm{{d}}t$ = {icost:2.3f}'.format(icost=0)

                self.text_icost_handle = self.sim_fig.text(
                    0.05, 0.5, text_icost, horizontalalignment='left', verticalalignment='center')

                self.r_cost_line, = self.cost_axes.plot(
                    self.t0, r, 'r-', lw=0.5, label='r')

                self.i_cost_line, = self.cost_axes.plot(
                    self.t0, 0, 'g-', lw=0.5, label=r'$\int r \,\mathrm{d}t$')

                self.text_icost_handles.append(self.text_icost_handle)
                self.r_cost_lines.append(self.r_cost_line)
                self.i_cost_lines.append(self.i_cost_line)

        self.cost_axes.legend(fancybox=True, loc='upper right')

        """
        
        Control subplot
        
        """
        self.ctrlAxs = self.sim_fig.add_subplot(224, autoscale_on=False, xlim=(self.t0, self.t1), ylim=(
            1.1 * np.min([self.f_min, self.m_min]), 1.1 * np.max([self.f_max, self.m_max])), xlabel='t [s]')

        self.ctrlAxs.title.set_text('Control')

        self.ctrlAxs.plot([self.t0, self.t1], [0, 0],
                          'k--', lw=0.75)   # Help line

        # Pack all lines together
        cLines = namedtuple('lines', ['traj_line', 'norm_line', 'alpha_line', 'r_cost_line', 'i_cost_line', 'ctrl_lines'])

        # logic for multiple controllers
        if self.num_controllers > 1:
            self.all_ctrl_lines = []
            self.all_lines = []
            
            for i in range(self.num_controllers):
                self.ctrl_lines = self.ctrlAxs.plot(self.t0, utilities._toColVec(self.u0[i]).T, lw=0.5)
                
                self.all_ctrl_lines.append(self.ctrl_lines)

            for i in range(self.num_controllers):
                self.all_lines.append(cLines(traj_line=self.traj_lines[i], norm_line=self.norm_lines[i], alpha_line=self.alpha_lines[i], r_cost_line=self.r_cost_lines[i],i_cost_line=self.i_cost_lines[i], ctrl_lines=self.all_ctrl_lines[i]))

            self.ctrlAxs.legend(iter(self.all_ctrl_lines[0]), ('F [N]', 'M [Nm]'), fancybox=True, loc='upper right')

        
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

    def _initialize_figure(self):
        if self.num_controllers > 1:
            for i in range(self.num_controllers):
                self.sol_scatter = self.xy_plane_axes.scatter(self.initial_xs[i], self.initial_ys[i], marker=self.robot_markers[i].marker, s=400, c=f"{self.colors[i]}")

        else:
            self.sol_scatter = self.xy_plane_axes.scatter(self.initial_x, self.initial_y, marker=self.robot_marker.marker, s=400, c='b')

        return self.sol_scatter

    def _update_line(self, line, new_x, new_y):
        line.set_xdata(np.append(line.get_xdata(), new_x))
        line.set_ydata(np.append(line.get_ydata(), new_y))

    def _reset_line(self, line):
        line.set_data([], [])

    def _update_text(self, text_handle, new_text):
        text_handle.set_text(new_text)

    def _update_all_lines(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u):
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

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        # Solution
        self._update_line(self.norm_line, t, self.l2_norm)
        self._update_line(self.alpha_line, t, alpha)

        # Cost
        self._update_line(self.r_cost_line, t, r)
        self._update_line(self.i_cost_line, t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handle, text_icost)

        # Control
        for (line, uSingle) in zip(self.ctrl_lines, u):
            self._update_line(line, t, uSingle)

    def _update_all_lines_multi(self, text_time, full_state, alpha_deg, x_coord, y_coord, t, alpha, r, icost, u, multi_controller_id):
        """
        Update lines on all scatter plots
        """
        cid = multi_controller_id
        self._update_text(self.text_time_handles[cid], text_time)

        # Update the robot's track on the plot
        self._update_line(self.traj_lines[cid], *full_state[:2])

        self.robot_markers[cid].rotate(alpha_deg)    # Rotate the robot on the plot
        self.sol_scatter.remove()
        self.sol_scatter = self.xy_plane_axes.scatter(
            x_coord, y_coord, marker=self.robot_markers[cid].marker, s=400, c='b')

        # Euclidean (aka Frobenius) norm
        self.l2_norm = la.norm([x_coord, y_coord])

        # Solution
        self._update_line(self.norm_lines[cid], t, self.l2_norm)
        self._update_line(self.alpha_lines[cid], t, alpha)

        # Cost
        self._update_line(self.r_cost_lines[cid], t, r)
        self._update_line(self.i_cost_lines[cid], t, icost)
        text_icost = f'$\int r \,\mathrm{{d}}t$ = {icost:2.1f}'
        self._update_text(self.text_icost_handles[cid], text_icost)

        # Control
        for (line, uSingle) in zip(self.all_ctrl_lines[cid], u):
            self._update_line(line, t, uSingle)

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
                                     'v [m/s]', 'omega [rad/s]', 'int r sample_time', 'F [N]', 'M [N m]'])

        return dataFiles

    def _print_sim_step(self, t, xCoord, yCoord, alpha, v, omega, icost, u):
        # alphaDeg = alpha/np.pi*180

        headerRow = ['t [s]', 'x [m]', 'y [m]', 'alpha [rad]',
                     'v [m/s]', 'omega [rad/s]', 'int r sample_time', 'F [N]', 'M [N m]']
        dataRow = [t, xCoord, yCoord, alpha, v, omega, icost, u[0], u[1]]
        rowFormat = ('8.1f', '8.3f', '8.3f', '8.3f',
                     '8.3f', '8.3f', '8.1f', '8.3f', '8.3f')
        table = tabulate([headerRow, dataRow], floatfmt=rowFormat,
                         headers='firstrow', tablefmt='grid')

        print(table)

    def _take_step(self, sys, controller, nominal_ctrl, simulator, animate=False, multi_controller_id=None):
        simulator.step()

        t = simulator.t
        full_state = simulator.y
        
        system_state = full_state[0:self.dim_state]
        y = sys.get_curr_state(system_state)

        u = self._ctrl_selector(
            t, y, self.u_man, nominal_ctrl, controller, controller.ctrl_mode)

        sys.get_action(u)
        controller.record_sys_state(sys.system_state)
        controller.update_icost(y, u)

        x_coord = full_state[0]
        y_coord = full_state[1]
        alpha = full_state[2]
        v = full_state[3]
        omega = full_state[4]
        icost = controller.i_cost_val

        if multi_controller_id is None:
            if self.is_print_sim_step:
                self._print_sim_step(t, x_coord, y_coord,
                                     alpha, v, omega, icost, u)

            if self.is_log_data:
                self._log_data_row(self.current_data_file, t, x_coord,
                                   y_coord, alpha, v, omega, icost.val, u)
        if animate == True:
            alpha_deg = alpha / np.pi * 180
            r = controller.running_cost(y, u)
            text_time = 't = {time:2.3f}'.format(time=t)
            
            if multi_controller_id is None:
                self._update_all_lines(text_time, full_state, alpha_deg,
                                       x_coord, y_coord, t, alpha, r, icost, u)
            else:
                self._update_all_lines_multi(text_time, full_state, alpha_deg,
                                       x_coord, y_coord, t, alpha, r, icost, u, multi_controller_id)

    def _reset_sim(self, controller, nominal_ctrl, simulator, multi_controller_id = None):
        if self.is_print_sim_step:
            print('.....................................Run {run:2d} done.....................................'.format(
                run=self.current_run))

        if self.is_log_data:
            self.current_data_file = self.data_files[self.current_run - 1]

        # Reset simulator
        simulator.status = 'running'
        simulator.t = self.t0

        if multi_controller_id is not None:
            simulator.y = self.full_states[multi_controller_id]

        else:
            simulator.y = self.full_state

        # Reset controller
        if controller.ctrl_mode > 0:
            controller.reset(self.t0)
        else:
            nominal_ctrl.reset(self.t0)

    def graceful_exit(self, plt_close=True):
        if plt_close is True:
            plt.close('all')

        # graceful exit from Jupyter notebook
        try:
            __IPYTHON__
            return None

        # graceful exit from terminal
        except NameError:
            print("Program exit")
            sys.exit()

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
                    self._reset_sim(controller[i], nominal_ctrl, simulator[i], i)

            else:
                self._reset_sim(controller, nominal_ctrl, simulator)
            
            self.graceful_exit()

    def _wrapper_take_steps(self, k, *args):
        system, controller, nominal_ctrl, simulator, animate = args
        
        if self.num_controllers > 1:
            simulators = simulator
            controllers = controller
            systems = system

            for i in range(self.num_controllers):
                t = simulators[i].t

                if self.current_run[i] <= self.n_runs:
                    if t < self.t1:
                        print(systems)
                        self._take_step(systems[i], controllers[i], nominal_ctrl, simulators[i], animate, multi_controller_id=i)

                    else:
                        self.current_run[i] += 1
                        self._reset_sim(controllers[i], nominal_ctrl, simulators[i])
                
                else:
                    if self.close_plt_on_finish is True:
                        self.graceful_exit()

                    elif self.close_plt_on_finish is False:
                        self.graceful_exit(plt_close=False)
        else:
            _, controller, nominal_ctrl, simulator, _ = args
            t = simulator.t

            if self.current_run <= self.n_runs:
                if t < self.t1:
                    self.t_elapsed = t
                    self._take_step(*args)

                else:
                    self.current_run += 1
                    self._reset_sim(controller, nominal_ctrl, simulator)
            else:
                if self.close_plt_on_finish is True:
                    self.graceful_exit()

                elif self.close_plt_on_finish is False:
                    self.graceful_exit(plt_close=False)

    def run_animation(self, system, controller, nominal_ctrl, simulator, fig_width, fig_height, multi_controllers = False):
            animate = True

            if multi_controllers is True:
                controllers = controller
                simulators = simulator
                systems = system
                self.sim_fig = self._create_figure_plots_multi(controllers, fig_width, fig_height)
                fargs = (systems, controllers, nominal_ctrl, simulators, animate)
            
            else:
                self.sim_fig = self._create_figure_plots(controllers, fig_width, fig_height)
                fargs = (system, controller, nominal_ctrl, simulator, animate)


            anm = animation.FuncAnimation(self.sim_fig,
                                          self._wrapper_take_steps,
                                          fargs=fargs,
                                          init_func=self._initialize_figure,
                                          interval=1)

            anm.running = True
            self.sim_fig.canvas.mpl_connect(
                'key_press_event', lambda event: self._on_key_press(event, anm, fargs))
            self.sim_fig.tight_layout()
            plt.show()

    def run_simulation(self, n_runs=1, fig_width=8, fig_height=8, close_plt_on_finish=True):
        if self.num_controllers > 1:
            self.current_run = [1, 1]
            self.t_elapsed = [0, 0]

        else:
            self.current_run = 1
            self.t_elapsed = 0

        self.close_plt_on_finish = close_plt_on_finish
        self.n_runs = n_runs
        self.data_files = self._log_data(n_runs, save=self.is_log_data)

        if self.is_visualization is False:
            self.current_data_file = data_files[0]

            t = self.simulator.t

            while self.current_run <= self.n_runs:
                while t < self.t1:
                    self._take_step(self.system, self.controller,
                                    self.nominal_ctrl, self.simulator)
                    t += 1

                else:
                    self._reset_sim(self.controller,
                                    self.nominal_ctrl, self.simulator)
                    icost = 0


                    for line in self.all_lines:
                        for item in line:
                            if item != self.traj_line:
                                if isinstance(item, list):
                                    for subitem in item:
                                        self._reset_line(subitem)
                                else:
                                    self._reset_line(item)

                    self._update_line(self.traj_line, np.nan, np.nan)
                self.current_run += 1
            else:
                self.graceful_exit()

        else:
            if self.num_controllers > 1:
                self.run_animation(self.systems, 
                    self.controllers, 
                    self.nominal_ctrl, 
                    self.simulators, 
                    fig_width, 
                    fig_height, 
                    multi_controllers = True)
            else:
                self.run_animation(self.system, self.controller, self.nominal_ctrl, self.simulator, fig_width, fig_height)
