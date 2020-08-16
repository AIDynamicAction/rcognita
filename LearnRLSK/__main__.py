from LearnRLSK import System, NominalController, Controller,Simulation
import argparse
import sys


def main(args=None):
    """main"""
    if args is not None:
        parser = argparse.ArgumentParser()

        parser.add_argument('-t0', type=int, default=0,
                            help="Start time of an episode")

        parser.add_argument('-t1', type=int, default=100,
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

        parser.add_argument('-dt', type=float,
                            default=0.05, help="Controller sampling time")

        # controller
        parser.add_argument('-gamma', type=int, default=1, help="Discounting factor gamma.")

        parser.add_argument('-mod_est_phase', type=int,
                            default=2, help="In seconds, an initial phase to fill the estimator's buffer before applying optimal control.")

        parser.add_argument('-model_order', type=int,
                            default=5, help="The order of the state-space estimation model.")

        parser.add_argument('-mod_est_checks', type=int,
                            default=0, help=" estimated model parameters can be stored in stacks and the best among the modEstchecks last ones is picked")

        parser.add_argument('-buffer_size', type=int,
                            default=200, help="The size of the buffer to store data for model estimation. Is measured in numbers of periods of length dt.")

        parser.add_argument('-nactor', type=int, default=6, help="Number of prediction steps. Nactor=1 means the controller is purely data-driven and doesn't use prediction.")

        parser.add_argument('-r_cost_struct', type=int,
                            default=1, help="Choice of the running cost structure.")

        parser.add_argument('-n_critic', type=int,
                            default=50, help="critic stack size.")

        parser.add_argument('-critic_struct', type=int,
                            default=3, help="choice of the structure of critic's feature vector. Options: 1, 2, 3, or 4. See code.")

        parser.add_argument('-ctrl_mode', type=int,
                            default=5, help="Control modes. 0, -1, 1, 2, 3, 4, 5 or 6")

        parser.add_argument('-is_dyn_ctrl', type=int,
                            default=0, help="Is dynamical controller.")

        # system
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

        # other
        parser.add_argument('-is_log_data', type=int,
                            default=0, help="Log data?")

        parser.add_argument('-is_visualization', type=int,
                            default=1, help="Visualize data?")

        parser.add_argument('-is_print_sim_step', type=int,
                            default=1, help="Print simulation steps?")

        parser.add_argument('-is_disturb', type=int,
                            default=0, help="description")

        # tbd
        parser.add_argument('-f_man', type=int, default=-3, help="description")
        parser.add_argument('-n_man', type=int, default=-1, help="description")
        parser.add_argument('-f_min', type=int, default=5, help="description")
        parser.add_argument('-f_max', type=int, default=5, help="description")
        parser.add_argument('-m_min', type=int, default=-1, help="description")
        parser.add_argument('-m_max', type=int, default=1, help="description")
        parser.add_argument('-prob_noise_pow', type=int,
                            default=8, help="description")

        args = parser.parse_args()

        dim_state = args.dim_state,
        dim_input = args.dim_input,
        dim_output = args.dim_output,
        dim_disturb = args.dim_disturb,
        m = args.m,
        I = args.I,
        t0 = args.t0,
        t1 = args.t1,
        n_runs = args.n_runs,
        a_tol = args.a_tol,
        r_tol = args.r_tol,
        x_min = args.x_min,
        x_max = args.x_max,
        y_min = args.y_min,
        y_max = args.y_max,
        dt = args.dt,
        mod_est_phase = args.mod_est_phase,
        model_order = args.model_order,
        prob_noise_pow = args.prob_noise_pow,
        mod_est_checks = args.mod_est_checks,
        f_man = args.f_man,
        n_man = args.n_man,
        f_min = args.f_min,
        f_max = args.f_max,
        m_min = args.m_min,
        m_max = args.m_max,
        nactor = args.nactor,
        buffer_size = args.buffer_size,
        r_cost_struct = args.r_cost_struct,
        n_critic = args.n_critic,
        gamma = args.gamma,
        critic_struct = args.critic_struct,
        is_log_data = args.is_log_data,
        is_visualization = args.is_visualization,
        is_print_sim_step = args.is_print_sim_step,
        is_disturb = args.is_disturb,
        is_dyn_ctrl = args.is_dyn_ctrl,
        ctrl_mode = args.ctrl_mode

        sim = Simulation(dim_state, dim_input, dim_output, dim_disturb, m, I, t0, t1, n_runs, a_tol, r_tol, x_min, x_max, y_min, y_max, dt, mod_est_phase, model_order, prob_noise_pow, mod_est_checks, f_man,
                         n_man, f_min, f_max, m_min, m_max, nactor, buffer_size, r_cost_struct, n_critic, gamma, critic_struct, is_log_data, is_visualization, is_print_sim_step, is_disturb, is_dyn_ctrl, ctrl_mode)

    else:
        # environment
        sys = System()
        nominalCtrl = NominalController(ctrlGain=0.5, samplTime=0.5)
        agent = Controller()
        sim = Simulation()
        simulator = sim.create_simulator(sys.closedLoop)
        sim.run_simulation(sys, agent, nominalCtrl, simulator)


    sim.run_sim()

if __name__ == "__main__":
    command_line_args = sys.argv[1:]
    main(command_line_args)
