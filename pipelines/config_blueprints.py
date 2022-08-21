import numpy as np
import argparse
from abc import abstractmethod
import sys
import pickle5 as pickle
import sys


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            # parse arguments in the file and store them in the target namespace
            parser.parse_args(f.read().split(), namespace)


class RcognitaArgParser(argparse.ArgumentParser):
    def __init__(self, description):

        super().__init__(description=description)
        self.add_argument(
            "--control_mode",
            metavar="control_mode",
            type=str,
            choices=["manual", "nominal", "MPC", "RQL", "SQL", "RLSTAB"],
            default="MPC",
            help="Control mode. Currently available: "
            + "----manual: manual constant control specified by action_manual; "
            + "----nominal: nominal controller, usually used to benchmark optimal controllers;"
            + "----MPC:model-predictive control; "
            + "----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; "
            + "----SQL: stacked Q-learning; "
            + "----RLSTAB: joint actor-critic (stabilizing), system-specific, needs proper setup.",
        )
        self.add_argument(
            "--is_log",
            action="store_true",
            help="Flag to log data into a data file. Data are stored in simdata folder.",
        )
        self.add_argument(
            "--no_visual",
            action="store_true",
            help="Flag to produce graphical output.",
        )
        self.add_argument(
            "--no_print",
            action="store_true",
            help="Flag to print simulation data into terminal.",
        )
        self.add_argument(
            "--is_est_model",
            action="store_true",
            help="Flag to estimate environment model.",
        )
        self.add_argument(
            "--save_trajectory",
            action="store_true",
            help="Flag to store trajectory inside the pipeline during execution.",
        )
        self.add_argument(
            "--dt",
            type=float,
            metavar="dt",
            default=0.01,
            help="Controller sampling time.",
        )
        self.add_argument(
            "--t1",
            type=float,
            metavar="t1",
            default=10.0,
            help="Final time of episode.",
        )
        self.add_argument("--config", type=open, action=LoadFromFile)


class MetaConf(type):
    def __init__(cls, name, bases, clsdict):
        if "argument_parser" in clsdict:

            def new_argument_parser(self):
                args = clsdict["argument_parser"](self)
                self.__dict__.update(vars(args))

            setattr(cls, "argument_parser", new_argument_parser)


class AbstractConfig(object, metaclass=MetaConf):
    @abstractmethod
    def __init__(self):
        self.config_name = None

    @abstractmethod
    def argument_parser(self):
        pass

    @abstractmethod
    def pre_processing(self):
        pass

    def get_env(self):
        self.argument_parser()
        self.pre_processing()
        return self.__dict__

    def config_to_pickle(self):
        with open(
            f"../tests/refs/env_{self.config_name}.pickle", "wb"
        ) as env_description_out:
            pickle.dump(self.__dict__, env_description_out)


class Config3WRobot(AbstractConfig):
    def __init__(self):
        self.config_name = "3wrobot"

    def argument_parser(self):
        description = (
            "Agent-environment preset: 3-wheel robot with dynamical actuators."
        )

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["5", "5", "-3*pi/4", "0", "0"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[-5, -3],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--Nactor",
            type=int,
            default=5,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.",
        )
        parser.add_argument(
            "--buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--stage_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0, 0, 0],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0, 0, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--Ncritic",
            type=int,
            default=4,
            help="Critic stack size (number of temporal difference terms in critic objective).",
        )
        parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 5
        self.dim_input = 2
        self.dim_output = self.dim_state
        self.dim_disturb = 0

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)

        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.dt * self.pred_step_size_multiplier
        self.model_est_period = self.dt * self.model_est_period_multiplier
        self.critic_period = self.dt * self.critic_period_multiplier
        if self.control_mode == "RLSTAB":
            self.Nactor = 1

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))
        self.is_disturb = 0

        self.is_dyn_ctrl = 0

        self.t0 = 0

        self.action_init = np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # xy-plane
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.Fmin = -300
        self.Fmax = 300
        self.Mmin = -100
        self.Mmax = 100
        self.control_bounds = np.array([[self.Fmin, self.Fmax], [self.Mmin, self.Mmax]])

        # System parameters
        self.m = 10  # [kg]
        self.I = 1  # [kg m^2]
        self.observation_target = []


class Config3WRobotNI(AbstractConfig):
    def __init__(self):
        self.config_name = "3wrobot_NI"

    def argument_parser(self):
        description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["5", "5", "-3*pi/4"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )
        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[-5, -3],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--Nactor",
            type=int,
            default=3,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=1.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.",
        )
        parser.add_argument(
            "--buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--stage_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[1, 10, 1, 0, 0],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--Ncritic",
            type=int,
            default=4,
            help="Critic stack size (number of temporal difference terms in critic objective).",
        )
        parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix", "NN"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations)."
            + "----NN: PyTorch neural network.",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):

        self.trajectory = []
        self.dim_state = 3
        self.dim_input = 2
        self.dim_output = self.dim_state
        self.dim_disturb = 2
        if self.control_mode == "RLSTAB":
            self.Nactor = 1

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1
        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)
        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.dt * self.pred_step_size_multiplier
        self.model_est_period = self.dt * self.model_est_period_multiplier
        self.critic_period = self.dt * self.critic_period_multiplier

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))

        assert self.t1 > self.dt > 0.0
        assert self.state_init.size == self.dim_state

        # ----------------------------------------(So far) fixed settings
        self.is_disturb = 0
        self.is_dyn_ctrl = 0

        self.t0 = 0

        self.action_init = 0 * np.ones(self.dim_input)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # xy-plane
        self.xMin = -10
        self.xMax = 10
        self.yMin = -10
        self.yMax = 10

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.v_min = -25
        self.v_max = 25
        self.omega_min = -5
        self.omega_max = 5
        self.control_bounds = np.array(
            [[self.v_min, self.v_max], [self.omega_min, self.omega_max]]
        )

        self.xCoord0 = self.state_init[0]
        self.yCoord0 = self.state_init[1]
        self.alpha0 = self.state_init[2]
        self.alpha_deg_0 = self.alpha0 / 2 / np.pi
        self.observation_target = []


class ConfigROS3WRobotNI(Config3WRobotNI):
    def get_env(self):
        self.argument_parser()
        self.pre_processing()
        self.v_min = -0.22
        self.v_max = 0.22
        self.omega_min = -2.84
        self.omega_max = 2.84
        self.control_bounds = np.array(
            [[self.v_min, self.v_max], [self.omega_min, self.omega_max]]
        )
        self.state_init = np.array([2, 2, 3.1415])
        return self.__dict__


class Config2Tank(AbstractConfig):
    def __init__(self):
        self.config_name = "2tank"

    def argument_parser(self):
        description = "Agent-environment preset: nonlinear double-tank system."

        parser = RcognitaArgParser(description=description)

        parser.add_argument(
            "--Nruns",
            type=int,
            default=1,
            help="Number of episodes. Learned parameters are not reset after an episode.",
        )
        parser.add_argument(
            "--state_init",
            type=str,
            nargs="+",
            metavar="state_init",
            default=["2", "-2"],
            help="Initial state (as sequence of numbers); "
            + "dimension is environment-specific!",
        )

        parser.add_argument(
            "--model_est_stage",
            type=float,
            default=1.0,
            help="Seconds to learn model until benchmarking controller kicks in.",
        )
        parser.add_argument(
            "--model_est_period_multiplier",
            type=float,
            default=1,
            help="Model is updated every model_est_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--model_order",
            type=int,
            default=5,
            help="Order of state-space estimation model.",
        )
        parser.add_argument(
            "--prob_noise_pow",
            type=float,
            default=False,
            help="Power of probing (exploration) noise.",
        )
        parser.add_argument(
            "--action_manual",
            type=float,
            default=[0.5],
            nargs="+",
            help="Manual control action to be fed constant, system-specific!",
        )
        parser.add_argument(
            "--Nactor",
            type=int,
            default=10,
            help="Horizon length (in steps) for predictive controllers.",
        )
        parser.add_argument(
            "--pred_step_size_multiplier",
            type=float,
            default=2.0,
            help="Size of each prediction step in seconds is a pred_step_size_multiplier multiple of controller sampling time dt.",
        )
        parser.add_argument(
            "--buffer_size",
            type=int,
            default=10,
            help="Size of the buffer (experience replay) for model estimation, agent learning etc.",
        )
        parser.add_argument(
            "--stage_obj_struct",
            type=str,
            default="quadratic",
            choices=["quadratic", "biquadratic"],
            help="Structure of stage objective function.",
        )
        parser.add_argument(
            "--R1_diag",
            type=float,
            nargs="+",
            default=[10, 10, 1],
            help="Parameter of stage objective function. Must have proper dimension. "
            + "Say, if chi = [observation, action], then a quadratic stage objective reads chi.T diag(R1) chi, where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--R2_diag",
            type=float,
            nargs="+",
            default=[10, 10, 1],
            help="Parameter of stage objective function . Must have proper dimension. "
            + "Say, if chi = [observation, action], then a bi-quadratic stage objective reads chi**2.T diag(R2) chi**2 + chi.T diag(R1) chi, "
            + "where diag() is transformation of a vector to a diagonal matrix.",
        )
        parser.add_argument(
            "--Ncritic",
            type=int,
            default=4,
            help="Critic stack size (number of temporal difference terms in critic objective).",
        )
        parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
        parser.add_argument(
            "--critic_period_multiplier",
            type=float,
            default=1.0,
            help="Critic is updated every critic_period_multiplier times dt seconds.",
        )
        parser.add_argument(
            "--critic_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix", "quad-mix"],
            help="Feature structure (critic). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms; "
            + "----quad-mix: quadratic, mixed observation-action terms (for, say, Q or advantage function approximations).",
        )
        parser.add_argument(
            "--actor_struct",
            type=str,
            default="quad-nomix",
            choices=["quad-lin", "quadratic", "quad-nomix"],
            help="Feature structure (actor). Currently available: "
            + "----quad-lin: quadratic-linear; "
            + "----quadratic: quadratic; "
            + "----quad-nomix: quadratic, no mixed terms.",
        )

        args = parser.parse_args()
        return args

    def pre_processing(self):
        self.trajectory = []
        self.dim_state = 2
        self.dim_input = 1
        self.dim_output = self.dim_state
        self.dim_disturb = 1

        self.dim_R1 = self.dim_output + self.dim_input
        self.dim_R2 = self.dim_R1

        # ----------------------------------------Post-processing of arguments
        # Convert `pi` to a number pi
        for k in range(len(self.state_init)):
            self.state_init[k] = eval(self.state_init[k].replace("pi", str(np.pi)))

        self.state_init = np.array(self.state_init)
        self.action_manual = np.array(self.action_manual)

        self.pred_step_size = self.dt * self.pred_step_size_multiplier
        self.model_est_period = self.dt * self.model_est_period_multiplier
        self.critic_period = self.dt * self.critic_period_multiplier

        self.R1 = np.diag(np.array(self.R1_diag))
        self.R2 = np.diag(np.array(self.R2_diag))

        assert self.t1 > self.dt > 0.0
        assert self.state_init.size == self.dim_state

        # ----------------------------------------(So far) fixed settings
        self.is_disturb = 0
        self.is_dyn_ctrl = 0

        self.t0 = 0

        self.action_init = 0.5 * np.ones(self.dim_input)

        self.disturb_init = 0 * np.ones(self.dim_disturb)

        # Solver
        self.atol = 1e-5
        self.rtol = 1e-3

        # Model estimator stores models in a stack and recall the best of model_est_checks
        self.model_est_checks = 0

        # Control constraints
        self.action_min = 0
        self.action_max = 1
        self.control_bounds = np.array([[self.action_min], [self.action_max]]).T

        # System parameters
        self.tau1 = 18.4
        self.tau2 = 24.4
        self.K1 = 1.3
        self.K2 = 1
        self.K3 = 0.2

        # Target filling of the tanks
        self.observation_target = np.array([0.5, 0.5])
