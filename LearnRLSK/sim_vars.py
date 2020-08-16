class Simulation:
    """class to create simulation and run simulation."""

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 dimDisturb=2,
                 m=10,
                 I=1,
                 t0=0,
                 t1=100,
                 n_runs=1,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10,
                 dt=0.05,
                 mod_est_phase=2,
                 model_order=5,
                 prob_noise_pow=8,
                 mod_est_checks=0,
                 f_man=-3,
                 n_man=-1,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 nactor=6,
                 buffer_size=200,
                 r_cost_struct=1,
                 n_critic=50,
                 gamma=1,
                 critic_struct=3,
                 is_log_data=0,
                 is_visualization=1,
                 is_print_sim_step=1,
                 is_disturb=0,
                 is_dyn_ctrl=0,
                 ctrl_mode=5):
        """system - needs description"""
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dimDisturb = dimDisturb
        self.m = m
        self.I = I

        """disturbance - needs description"""
        self.sigma_q = 1e-3 * np.ones(dimDisturb)
        self.mu_q = np.zeros(dimDisturb)
        self.tau_q = np.ones(dimDisturb)

        """simulation"""

        # start time of episode
        self.t0 = t0

        # stop time of episode
        self.t1 = t1

        # number of episodes
        self.n_runs = n_runs

        # initial values of state
        self.x0 = np.zeros(dim_state)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2

        # initial value of control
        self.u0 = np.zeros(dim_input)

        # initial value of disturbance
        self.q0 = np.zeros(dimDisturb)

        # sensitivity of the solver. The lower the values, the more accurate the simulation results are
        self.a_tol = a_tol
        self.r_tol = r_tol

        # x and y limits of scatter plot. Used so far rather for visualization
        # only, but may be integrated into the actor as constraints
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        """ controller sampling time.
        The system itself is continuous as a physical process while the controller is digital.

        Things to note:
            * the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
            * smaller sampling times lead to higher computation times
            * especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with dt and try achieve a trade-off between stability and computational performance
        """
        self.dt = dt

        """ model estimator """

        # In seconds, an initial phase to fill the estimator's buffer before
        # applying optimal control
        self.mod_est_phase = mod_est_phase

        # In seconds, the time between model estimate updates. This constant
        # determines how often the estimated parameters are updated. The more
        # often the model is updated, the higher the computational burden is.
        # On the other hand, more frequent updates help keep the model actual.
        self.modEstPeriod = 1 * dt

        # The order of the state-space estimation model. We are interested in
        # adequate predictions of y under given u's. The higher the model
        # order, the better estimation results may be achieved, but be aware of
        # overfitting
        self.model_order = model_order
        self.prob_noise_pow = prob_noise_pow

        # Estimated model parameters can be stored in stacks and the best among
        # the modEstchecks last ones is picked
        self.mod_est_checks = mod_est_checks

        # The size of the buffer to store data for model estimation. The bigger
        # the buffer, the more accurate the estimation may be achieved. For
        # successful model estimation, the system must be sufficiently excited.
        # Using bigger buffers is a way to achieve this.
        self.buffer_size = buffer_size

        """ Controller 

            # u[0]: Pushing force F [N]
            # u[1]: Steering torque M [N m]
        """

        # Number of prediction steps. Nactor=1 means the controller is purely
        # data-driven and doesn't use prediction.
        self.nactor = nactor

        # In seconds. Should be a multiple of dt
        self.predStepSize = 5 * dt

        """ RL elements

            Running cost. 

            Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.

            Notation: chi = [y, u]
            1 - quadratic chi.T R1 chi
            2 - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
            R1, R2 must be positive-definite
        """
        self.r_cost_struct = r_cost_struct
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        # R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
        # R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
        # R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1],
        # [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        # R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
        # R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1],
        # [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

        """Critic stack size.

            Should not greater than bufferSize. The critic optimizes the temporal error which is a measure of critic's ability to capture the optimal infinite-horizon cost (a.k.a. the value function). The temporal errors are stacked up using the said buffer. The principle here is pretty much the same as with the model estimation: accuracy against performance

        """
        if n_critic > buffer_size:
            n_critic = buffer_size

        self.n_critic = n_critic

        # Time between critic updates
        self.critic_period = 5 * dt

        """critic structure

            1 - quadratic-linear
            2 - quadratic
            3 - quadratic, no mixed terms
            4 - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...]
            # u[0]^2 + ...
        """
        self.critic_struct = critic_struct

        """control mode
        
            Modes with online model estimation are experimental
            
            0     - manual constant control (only for basic testing)
            -1    - nominal parking controller (for benchmarking optimal controllers)
            1     - model-predictive control (MPC). Prediction via discretized true model
            2     - adaptive MPC. Prediction via estimated model
            3     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via discretized true model
            4     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via estimated model
            5     - RL: stacked Q-learning. Prediction via discretized true model
            6     - RL: stacked Q-learning. Prediction via estimated model
        """
        self.ctrl_mode = ctrl_mode

        # manual control
        self.f_man = f_man
        self.n_man = n_man
        self.uMan = np.array([f_man, n_man])

        # control constraints
        self.f_min = f_min
        self.f_max = f_max
        self.m_min = m_min
        self.m_max = m_max

        self.ctrlBnds = np.array(
            [[self.f_min, self.f_max], [self.m_min, self.m_max]])

        """Other"""
        # Discounting factor
        self.gamma = gamma

        #%% User settings: main switches
        self.is_log_data = is_log_data
        self.is_visualization = is_visualization
        self.is_print_sim_step = is_print_sim_step
        self.is_disturb = is_disturb

        # Static or dynamic controller
        self.is_dyn_ctrl = is_dyn_ctrl

        if self.is_dyn_ctrl:
            self.ksi0 = np.concatenate([self.x0, self.q0, self.u0])
        else:
            self.ksi0 = np.concatenate([self.x0, self.q0])

        # extras
        self.dataFiles = logdata(self.n_runs, save=self.is_log_data)

        if self.is_print_sim_step:
            warnings.filterwarnings('ignore')