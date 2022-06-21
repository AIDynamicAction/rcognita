import numpy as np
import argparse


def get_local_vars_namespace(args, env_snapshot):
    excl = ["args", "pars"]
    for key in env_snapshot:
        if all(x not in key for x in excl):
            args.__dict__[key] = env_snapshot[key]
    return args


def config_3wrobot():
    description = "Agent-environment preset: 3-wheel robot with dynamical actuators."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--ctrl_mode",
        metavar="ctrl_mode",
        type=str,
        choices=["manual", "nominal", "MPC", "RQL", "SQL", "JACS"],
        default="MPC",
        help="Control mode. Currently available: "
        + "----manual: manual constant control specified by action_manual; "
        + "----nominal: nominal controller, usually used to benchmark optimal controllers;"
        + "----MPC:model-predictive control; "
        + "----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; "
        + "----SQL: stacked Q-learning; "
        + "----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.",
    )
    parser.add_argument(
        "--dt", type=float, metavar="dt", default=0.01, help="Controller sampling time."
    )
    parser.add_argument(
        "--t1", type=float, metavar="t1", default=10.0, help="Final time of episode."
    )
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
        "--is_log_data",
        type=bool,
        default=False,
        help="Flag to log data into a data file. Data are stored in simdata folder.",
    )
    parser.add_argument(
        "--is_visualization",
        type=bool,
        default=True,
        help="Flag to produce graphical output.",
    )
    parser.add_argument(
        "--is_print_sim_step",
        type=bool,
        default=True,
        help="Flag to print simulation data into terminal.",
    )
    parser.add_argument(
        "--is_est_model",
        type=bool,
        default=False,
        help="Flag to estimate environment model.",
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
        help="Critic stack size (number of temporal difference terms in critic cost).",
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
    args.dim_state = 5
    args.dim_input = 2
    args.dim_output = args.dim_state
    args.dim_disturb = 0

    args.dim_R1 = args.dim_output + args.dim_input
    args.dim_R2 = args.dim_R1

    # ----------------------------------------Post-processing of arguments
    # Convert `pi` to a number pi
    for k in range(len(args.state_init)):
        args.state_init[k] = eval(args.state_init[k].replace("pi", str(np.pi)))

    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)

    args.pred_step_size = args.dt * args.pred_step_size_multiplier
    args.model_est_period = args.dt * args.model_est_period_multiplier
    args.critic_period = args.dt * args.critic_period_multiplier

    args.R1 = np.diag(np.array(args.R1_diag))
    args.R2 = np.diag(np.array(args.R2_diag))
    args.is_disturb = 0

    args.is_dyn_ctrl = 0

    args.t0 = 0

    args.action_init = 0 * np.ones(args.dim_input)

    # Solver
    args.atol = 1e-5
    args.rtol = 1e-3

    # xy-plane
    args.xMin = -10
    args.xMax = 10
    args.yMin = -10
    args.yMax = 10

    # Model estimator stores models in a stack and recall the best of model_est_checks
    args.model_est_checks = 0

    # Control constraints
    args.Fmin = -300
    args.Fmax = 300
    args.Mmin = -100
    args.Mmax = 100
    args.ctrl_bnds = np.array([[args.Fmin, args.Fmax], [args.Mmin, args.Mmax]])

    # System parameters
    args.m = 10  # [kg]
    args.I = 1  # [kg m^2]

    return args


def config_3wrobot_ni():

    description = "Agent-environment preset: a 3-wheel robot (kinematic model a. k. a. non-holonomic integrator)."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--ctrl_mode",
        metavar="ctrl_mode",
        type=str,
        choices=["manual", "nominal", "MPC", "RQL", "SQL", "JACS"],
        default="MPC",
        help="Control mode. Currently available: "
        + "----manual: manual constant control specified by action_manual; "
        + "----nominal: nominal controller, usually used to benchmark optimal controllers;"
        + "----MPC:model-predictive control; "
        + "----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; "
        + "----SQL: stacked Q-learning; "
        + "----JACS: joint actor-critic (stabilizing), system-specific, needs proper setup.",
    )
    parser.add_argument(
        "--dt", type=float, metavar="dt", default=0.01, help="Controller sampling time."
    )
    parser.add_argument(
        "--t1", type=float, metavar="t1", default=10.0, help="Final time of episode."
    )
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
        "--is_log_data",
        type=bool,
        default=False,
        help="Flag to log data into a data file. Data are stored in simdata folder.",
    )
    parser.add_argument(
        "--is_visualization",
        type=bool,
        default=True,
        help="Flag to produce graphical output.",
    )
    parser.add_argument(
        "--is_print_sim_step",
        type=bool,
        default=True,
        help="Flag to print simulation data into terminal.",
    )
    parser.add_argument(
        "--is_est_model",
        type=bool,
        default=False,
        help="Flag to estimate environment model.",
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
        help="Critic stack size (number of temporal difference terms in critic cost).",
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
    args.dim_state = 3
    args.dim_input = 2
    args.dim_output = args.dim_state
    args.dim_disturb = 2

    args.dim_R1 = args.dim_output + args.dim_input
    args.dim_R2 = args.dim_R1
    # ----------------------------------------Post-processing of arguments
    # Convert `pi` to a number pi
    for k in range(len(args.state_init)):
        args.state_init[k] = eval(args.state_init[k].replace("pi", str(np.pi)))

    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)

    args.pred_step_size = args.dt * args.pred_step_size_multiplier
    args.model_est_period = args.dt * args.model_est_period_multiplier
    args.critic_period = args.dt * args.critic_period_multiplier

    args.R1 = np.diag(np.array(args.R1_diag))
    args.R2 = np.diag(np.array(args.R2_diag))

    assert args.t1 > args.dt > 0.0
    assert args.state_init.size == args.dim_state

    # ----------------------------------------(So far) fixed settings
    args.is_disturb = 0
    args.is_dyn_ctrl = 0

    args.t0 = 0

    args.action_init = 0 * np.ones(args.dim_input)

    # Solver
    args.atol = 1e-5
    args.rtol = 1e-3

    # xy-plane
    args.xMin = -10
    args.xMax = 10
    args.yMin = -10
    args.yMax = 10

    # Model estimator stores models in a stack and recall the best of model_est_checks
    args.model_est_checks = 0

    # Control constraints
    args.v_min = -25
    args.v_max = 25
    args.omega_min = -5
    args.omega_max = 5
    args.ctrl_bnds = np.array(
        [[args.v_min, args.v_max], [args.omega_min, args.omega_max]]
    )

    args.xCoord0 = args.state_init[0]
    args.yCoord0 = args.state_init[1]
    args.alpha0 = args.state_init[2]
    args.alpha_deg_0 = args.alpha0 / 2 / np.pi

    return args


def config_2tank():
    dim_state = 2
    dim_input = 1
    dim_output = dim_state
    dim_disturb = 1

    dim_R1 = dim_output + dim_input
    dim_R2 = dim_R1

    description = "Agent-environment preset: nonlinear double-tank system."

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--ctrl_mode",
        metavar="ctrl_mode",
        type=str,
        choices=["manual", "MPC", "RQL", "SQL"],
        default="MPC",
        help="Control mode. Currently available: "
        + "----manual: manual constant control specified by action_manual; "
        + "----MPC:model-predictive control; "
        + "----RQL: Q-learning actor-critic with Nactor-1 roll-outs of stage objective; "
        + "----SQL: stacked Q-learning.",
    )
    parser.add_argument(
        "--dt", type=float, metavar="dt", default=0.1, help="Controller sampling time."
    )
    parser.add_argument(
        "--t1", type=float, metavar="t1", default=100.0, help="Final time of episode."
    )
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
        "--is_log_data",
        type=bool,
        default=False,
        help="Flag to log data into a data file. Data are stored in simdata folder.",
    )
    parser.add_argument(
        "--is_visualization",
        type=bool,
        default=True,
        help="Flag to produce graphical output.",
    )
    parser.add_argument(
        "--is_print_sim_step",
        type=bool,
        default=True,
        help="Flag to print simulation data into terminal.",
    )
    parser.add_argument(
        "--is_est_model",
        type=bool,
        default=False,
        help="Flag to estimate environment model.",
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
        help="Critic stack size (number of temporal difference terms in critic cost).",
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

    # ----------------------------------------Post-processing of arguments
    # Convert `pi` to a number pi
    for k in range(len(args.state_init)):
        args.state_init[k] = eval(args.state_init[k].replace("pi", str(np.pi)))

    args.state_init = np.array(args.state_init)
    args.action_manual = np.array(args.action_manual)

    pred_step_size = args.dt * args.pred_step_size_multiplier
    model_est_period = args.dt * args.model_est_period_multiplier
    critic_period = args.dt * args.critic_period_multiplier

    R1 = np.diag(np.array(args.R1_diag))
    R2 = np.diag(np.array(args.R2_diag))

    assert args.t1 > args.dt > 0.0
    assert args.state_init.size == dim_state

    # ----------------------------------------(So far) fixed settings
    is_disturb = 0
    is_dyn_ctrl = 0

    t0 = 0

    action_init = 0.5 * np.ones(dim_input)

    disturb_init = 0 * np.ones(dim_disturb)

    # Solver
    atol = 1e-5
    rtol = 1e-3

    # Model estimator stores models in a stack and recall the best of model_est_checks
    model_est_checks = 0

    # Control constraints
    action_min = 0
    action_max = 1
    ctrl_bnds = np.array([[action_min], [action_max]]).T

    # System parameters
    tau1 = 18.4
    tau2 = 24.4
    K1 = 1.3
    K2 = 1
    K3 = 0.2

    # Target filling of the tanks
    observation_target = np.array([0.5, 0.5])

    # print(locals())
    env_snapshot = locals()
    args = get_local_vars_namespace(args, env_snapshot)
    print(args)
    return args
