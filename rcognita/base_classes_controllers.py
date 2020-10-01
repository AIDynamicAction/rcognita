import warnings
import itertools
from rcognita import EndiSystem

# numpy
import numpy as np

class EndiControllerBase:
    """
    Optimal controller (a.k.a. agent) class. Actor-Critic model.

    ----------
    Parameters
    ----------

    system : object of type `System` class
        object of type System (class)

    t0 : int
        * Initial value of the controller's internal clock

    t1 : int
        * End value of controller's internal clock

    r_cost_struct : int
        * Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.
        * 1 - quadratic chi.T @ R1 @ chi
        * 2 - 4th order chi**2.T @ R2 @ chi**2 + chi.T @ R2 @ chi

    sample_time : int or float
        Controller's sampling time (in seconds). The system itself is continuous as a physical process while the controller is digital.
        * the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
        * smaller sampling times lead to higher computation times
        * especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with sample_time and try achieve a trade-off between stability and computational performance

    gamma : float
        * Discounting factor
        * number in (0, 1]
        * Characterizes fading of running costs along horizon


    ----------
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155

    """

    def __init__(self,
                 system,
                 t0=0,
                 t1=15,
                 buffer_size=10,
                 r_cost_struct=1,
                 sample_time=0.2,
                 step_size=0.3,
                 gamma=0.95):
        """

        SYSTEM-RELATED ATTRIBUTES

        """
        self.dim_state = system.dim_state
        self.dim_input = system.dim_input
        self.dim_output = system.dim_output
        self.m = system.m
        self.I = system.I
        self.is_disturb = system.is_disturb
        self.system_state = system.system_state
        self.ctrl_bnds = system.control_bounds
        self.sys_dynamics = system._get_system_dynamics
        self.sys_output = system.get_curr_state
        self.f_min = system.f_min
        self.f_max = system.f_max
        self.m_min = system.m_min
        self.m_max = system.m_max

        """

        CONTROLLER-RELATED ATTRIBUTES

        """
        self.t0 = t0
        self.t1 = t1
        self.est_clock = t0
        self.ctrl_clock = self.t0

        self.r_cost_struct = r_cost_struct

        # running cost parameters
        # state space
        self.Q = np.diag([10, 10, 1, 0, 0])

        # action space
        self.R = np.diag([0, 0])

        # for 4th order
        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

        self.i_cost_val = 0

        self.sample_time = sample_time
        self.step_size = step_size

        self.min_bounds = self.ctrl_bnds[:, 0]
        self.max_bounds = self.ctrl_bnds[:, 1]
        self.u_curr = self.min_bounds / 10
        self.buffer_size = buffer_size

        # buffer of previous controls
        self.u_buffer = np.zeros([buffer_size, self.dim_input])

        # buffer of previous outputs
        self.y_buffer = np.zeros([buffer_size, self.dim_output])

        # discount factor
        self.gamma = gamma

    def record_sys_state(self, system_state):
        self.system_state = system_state

    def running_cost(self, y, u):
        """
        Running cost (a.k.a. utility, reward, instantaneous cost etc.)
        """

        r = 0

        if self.r_cost_struct == 1:
            r = (y @ self.Q @ y) + (u @ self.R @ u)

        elif self.r_cost_struct == 2:
            chi = np.concatenate((y, u))
            r = chi**2 @ self.R2 @ chi**2 + chi @ self.R1 @ chi

        return r

    def update_icost(self, y, u):
        """
        Sample-to-sample integrated running cost. This can be handy to evaluate the performance of the agent.

        If the agent succeeded to stabilize the system, `icost` would converge to a finite value which is the performance mark.

        The smaller, the better (depends on the problem specification of course - you might want to maximize cost instead)

        """
        self.i_cost_val += self.running_cost(y, u) * self.sample_time

        return self.i_cost_val


    def reset(self, t0):
        """
        Resets agent for use in multi-episode simulation.
        All the learned parameters are retained
        """
        self.ctrl_clock = t0
        self.u_curr = self.min_bounds / 10