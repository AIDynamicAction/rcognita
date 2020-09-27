from rcognita import EndiSystem, NominalController, ActorCritic, Simulation
import argparse
import sys

def main(args=None):
    """main"""
    parser = argparse.ArgumentParser()

    parser.add_argument('-t0', type=int, default=0,
                        help="Start time of an episode")

    parser.add_argument('-t1', type=int, default=20,
                        help="Stop time of an episode")

    parser.add_argument('-n_runs', type=int, default=1,
                        help="Number of episodes. After an episode, the system is reset to the initial state, whereas all the learned parameters continue to get updated. This emulates multi-trial RL")

    parser.add_argument('-a_tol', type=float,
                        default=1e-5, help="Sensitivity of solver. The lower the values, the more accurate the simulation results are.")

    parser.add_argument('-r_tol', type=float,
                        default=1e-3, help="Sensitivity of solver. The lower the values, the more accurate the simulation results are.")

    parser.add_argument('-x_min', type=int, default=-
                        10, help="Min x bound for scatter plot")

    parser.add_argument('-x_max', type=int, default=10,
                        help="Max x bound for scatter plot")

    parser.add_argument('-y_min', type=int, default=-
                        10, help="Min y bound for scatter plot")

    parser.add_argument('-y_max', type=int, default=10,
                        help="Max y bound for scatter plot")

    # controller
    parser.add_argument('-gamma', type=int, default=1,
                        help="Discounting factor gamma.")

    parser.add_argument('-estimator_buffer_power', type=int, default=6,
                        help="Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control")

    parser.add_argument('-estimator_buffer_fill', type=int,
                        default=3, help="In seconds, an initial phase to fill the estimator's buffer before applying optimal control.")

    parser.add_argument('-estimator_update_time', type=float,
                        default=0.3, help="In seconds, the time between model estimate updates. This constant determines how often the estimated parameters")

    parser.add_argument('-model_order', type=int,
                        default=3, help="The order of the state-space estimation model.")

    parser.add_argument('-stacked_model_params', type=int,
                        default=0, help="Estimated model parameters can be stored in stacks and the best among the last ones is picked")

    parser.add_argument('-buffer_size', type=int,
                        default=20, help="critic stack size.")

    parser.add_argument('-actor_control_horizon', type=int, default=10,
                        help="Number of prediction steps. actor_control_horizon=1 means the controller is purely data-driven and doesn't use prediction.")

    # parser.add_argument('-r_cost_struct', type=int, default=1, help="Choice of the running cost structure.")

    parser.add_argument('-sample_time', type=float,
                        default=0.3, help="Sample time")

    parser.add_argument('-pred_step_size', type=float,
                        default=0.6, help="prediction step size")

    parser.add_argument('-critic_update_time', type=float,
                        default=0.1, help="Time between critic updates")

    parser.add_argument('-critic_mode', type=int,
                        default=3, help="choice of the structure of critic's feature vector. Options: 1, 2, 3, or 4. See code.")

    parser.add_argument('-ctrl_mode', type=int,
                        default=3, help="Control modes. 0, -1, 1, 2, 3, 4, 5 or 6")

    parser.add_argument('-is_dyn_ctrl', type=int,
                        default=0, help="Is dynamical controller.")

    parser.add_argument('-dim_state', type=int,
                        default=5, help="Dimension of the state (x)")

    parser.add_argument('-dim_input', type=int,
                        default=2, help="Dimension of control input (u)")

    parser.add_argument('-dim_output', type=int,
                        default=5, help="Dimension of output (y)")

    parser.add_argument('-dim_disturb', type=int,
                        default=2, help="description")

    parser.add_argument('-m', type=int, default=10, help="description")

    parser.add_argument('-I', dest="I", type=int,
                        default=1, help="description")

    parser.add_argument('-is_log_data', type=int,
                        default=0, help="Log data?")

    parser.add_argument('-is_visualization', type=bool,
                        default=True, help="Visualize data?")

    parser.add_argument('-print_statistics_at_step', type=bool,
                        default=True, help="Print simulation steps?")

    parser.add_argument('-is_disturb', type=int,
                        default=0, help="description")

    parser.add_argument('-ctrl_gain', type=int,
                        default=10, help="controller gain")

    args = parser.parse_args(args)

    dim_state = args.dim_state
    dim_input = args.dim_input
    dim_output = args.dim_output
    dim_disturb = args.dim_disturb
    m = args.m
    I = args.I
    t0 = args.t0
    t1 = args.t1
    n_runs = args.n_runs
    a_tol = args.a_tol
    r_tol = args.r_tol
    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max
    estimator_buffer_fill = args.estimator_buffer_fill
    model_order = args.model_order
    estimator_buffer_power = args.estimator_buffer_power
    estimator_update_time = args.estimator_update_time
    stacked_model_params = args.stacked_model_params
    actor_control_horizon = args.actor_control_horizon
    r_cost_struct = 1
    buffer_size = args.buffer_size
    critic_update_time = args.critic_update_time
    gamma = args.gamma
    critic_mode = args.critic_mode
    is_log_data = args.is_log_data
    is_visualization = args.is_visualization
    print_statistics_at_step = args.print_statistics_at_step
    is_disturb = args.is_disturb
    is_dyn_ctrl = args.is_dyn_ctrl
    sample_time = args.sample_time
    pred_step_size = args.pred_step_size
    ctrl_mode = args.ctrl_mode
    ctrl_gain = args.ctrl_gain
    initial_x = 5
    initial_y = 5

    sys = EndiSystem(dim_state,
                dim_input,
                dim_output,
                dim_disturb,
                initial_x,
                initial_y,
                m,
                I,
                is_dyn_ctrl,
                is_disturb)

    controller = ActorCritic(sys,
                            t0,
                            t1,
                            actor_control_horizon,
                            buffer_size,
                            ctrl_mode,
                            critic_mode,
                            critic_update_time,
                            r_cost_struct,
                            sample_time,
                            pred_step_size,
                            estimator_update_time,
                            estimator_buffer_fill,
                            estimator_buffer_power,
                            stacked_model_params,
                            model_order,
                            gamma)

    nominal_ctrl = NominalController(t0,
                                    m,
                                    I,
                                    ctrl_gain,
                                    sample_time)


    sim = Simulation(sys, controller, nominal_ctrl)
    sim.run_simulation(n_runs=n_runs, 
        is_visualization = is_visualization, 
        print_summary_stats=True, 
        print_statistics_at_step=print_statistics_at_step)

if __name__ == "__main__":
    main()
