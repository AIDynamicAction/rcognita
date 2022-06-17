import os, sys
import pickle5 as pickle
import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pytest

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)

from rcognita import controllers, visuals, simulator, systems, loggers
from rcognita.utilities import on_key_press


def env_to_pickle():
    with open("./refs/env_3wrobot.pickle", "wb") as env_description_out:
        pickle.dump(env_description_3wrobot(), env_description_out)


def env_description_3wrobot():
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
    args.I = 1  # [kg m^

    return args


def get_env():
    files = os.listdir("./refs")

    if "env_3wrobot.pickle" not in files:
        env_to_pickle()
    with open("./refs/env_3wrobot.pickle", "rb") as f:
        args = pickle.load(f)

    return args


class PresetPipeline:
    def setup_env(self, args={}):
        self.__dict__.update(vars(get_env()))
        self.__dict__.update(args)
        print(self.__dict__)
        self.trajectory = []

    def system_initialization(self):
        self.my_sys = systems.Sys3WRobot(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[self.m, self.I],
            ctrl_bnds=self.ctrl_bnds,
            is_dyn_ctrl=self.is_dyn_ctrl,
            is_disturb=self.is_disturb,
            pars_disturb=[],
        )

    def state_predictor_initialization(self):
        self.state_predictor = controllers.EulerStatePredictor(
            self.pred_step_size,
            self.my_sys._state_dyn,
            self.my_sys.out,
            self.dim_output,
            self.Nactor,
        )

    def controller_initialization(self):
        self.my_ctrl_nominal = controllers.CtrlNominal3WRobot(
            self.m,
            self.I,
            ctrl_gain=5,
            ctrl_bnds=self.ctrl_bnds,
            t0=self.t0,
            sampling_time=self.dt,
        )

        self.my_ctrl_opt_pred = controllers.CtrlOptPred(
            self.dim_input,
            self.dim_output,
            self.ctrl_mode,
            ctrl_bnds=self.ctrl_bnds,
            action_init=[],
            t0=self.t0,
            sampling_time=self.dt,
            Nactor=self.Nactor,
            actor_opt_method="SLSQP",
            pred_step_size=self.pred_step_size,
            sys_rhs=self.my_sys._state_dyn,
            sys_out=self.my_sys.out,
            state_sys=self.state_init,
            state_predictor=self.state_predictor,
            prob_noise_pow=self.prob_noise_pow,
            is_est_model=self.is_est_model,
            model_est_stage=self.model_est_stage,
            model_est_period=self.model_est_period,
            buffer_size=self.buffer_size,
            model_order=self.model_order,
            model_est_checks=self.model_est_checks,
            gamma=self.gamma,
            Ncritic=self.Ncritic,
            critic_period=self.critic_period,
            critic_struct=self.critic_struct,
            stage_obj_struct=self.stage_obj_struct,
            stage_obj_pars=[self.R1],
            observation_target=[],
        )

        self.my_ctrl_RL_stab = controllers.CtrlRLStab(
            self.dim_input,
            self.dim_output,
            self.ctrl_mode,
            ctrl_bnds=self.ctrl_bnds,
            action_init=self.action_init,
            t0=self.t0,
            sampling_time=self.dt,
            Nactor=self.Nactor,
            pred_step_size=self.pred_step_size,
            sys_rhs=self.my_sys._state_dyn,
            sys_out=self.my_sys.out,
            state_sys=self.state_init,
            prob_noise_pow=self.prob_noise_pow,
            is_est_model=self.is_est_model,
            model_est_stage=self.model_est_stage,
            model_est_period=self.model_est_period,
            buffer_size=self.buffer_size,
            model_order=self.model_order,
            model_est_checks=self.model_est_checks,
            gamma=self.gamma,
            Ncritic=self.Ncritic,
            critic_period=self.critic_period,
            critic_struct=self.critic_struct,
            actor_struct=self.actor_struct,
            stage_obj_struct=self.stage_obj_struct,
            stage_obj_pars=[self.R1],
            observation_target=[],
            safe_ctrl=self.my_ctrl_nominal,
            safe_decay_rate=1e-4,
        )

        self.my_ctrl_benchm = self.my_ctrl_opt_pred

    def simulator_initialization(self):
        self.my_simulator = simulator.Simulator(
            sys_type="diff_eqn",
            closed_loop_rhs=self.my_sys.closed_loop_rhs,
            sys_out=self.my_sys.out,
            state_init=self.state_init,
            disturb_init=[],
            action_init=self.action_init,
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            max_step=self.dt / 2,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dyn_ctrl=self.is_dyn_ctrl,
        )

    def logger_initialization(self):
        self.datafiles = [None] * self.Nruns
        self.my_logger = loggers.Logger3WRobot()

    def main_loop_visual(self):
        self.state_full_init = self.my_simulator.state_full

        my_animator = visuals.Animator3WRobot(
            objects=(
                self.my_simulator,
                self.my_sys,
                self.my_ctrl_nominal,
                self.my_ctrl_benchm,
                self.datafiles,
                controllers.ctrl_selector,
                self.my_logger,
            ),
            pars=(
                self.state_init,
                self.action_init,
                self.t0,
                self.t1,
                self.state_full_init,
                self.xMin,
                self.xMax,
                self.yMin,
                self.yMax,
                self.ctrl_mode,
                self.action_manual,
                self.Fmin,
                self.Mmin,
                self.Fmax,
                self.Mmax,
                self.Nruns,
                self.is_print_sim_step,
                self.is_log_data,
                0,
                [],
            ),
        )

        anm = animation.FuncAnimation(
            my_animator.fig_sim,
            my_animator.animate,
            init_func=my_animator.init_anim,
            blit=False,
            interval=self.dt / 1e6,
            repeat=False,
        )

        my_animator.get_anm(anm)

        cId = my_animator.fig_sim.canvas.mpl_connect(
            "key_press_event", lambda event: on_key_press(event, anm)
        )

        anm.running = True
        my_animator.fig_sim.tight_layout()

        plt.show()

    def main_loop_raw(self, save_trajectory):
        run_curr = 1
        datafile = self.datafiles[0]

        while True:

            self.my_simulator.sim_step()

            (t, _, observation, state_full,) = self.my_simulator.get_sim_step_data()

            if save_trajectory:
                self.trajectory.append(state_full)

            action = controllers.ctrl_selector(
                t,
                observation,
                self.action_manual,
                self.my_ctrl_nominal,
                self.my_ctrl_benchm,
                self.ctrl_mode,
            )

            self.my_sys.receive_action(action)
            self.my_ctrl_benchm.receive_sys_state(self.my_sys._state)
            self.my_ctrl_benchm.upd_accum_obj(observation, action)

            xCoord = state_full[0]
            yCoord = state_full[1]
            alpha = state_full[2]
            v = state_full[3]
            omega = state_full[4]

            stage_obj = self.my_ctrl_benchm.stage_obj(observation, action)
            accum_obj = self.my_ctrl_benchm.accum_obj_val

            if self.is_print_sim_step:
                self.my_logger.print_sim_step(
                    t, xCoord, yCoord, alpha, v, omega, stage_obj, accum_obj, action
                )

            if self.is_log_data:
                self.my_logger.log_data_row(
                    datafile,
                    t,
                    xCoord,
                    yCoord,
                    alpha,
                    v,
                    omega,
                    stage_obj,
                    accum_obj,
                    action,
                )

            if t >= self.t1:
                if self.is_print_sim_step:
                    print(
                        ".....................................Run {run:2d} done.....................................".format(
                            run=run_curr
                        )
                    )

                run_curr += 1

                if run_curr > self.Nruns:
                    break

                if self.is_log_data:
                    datafile = self.datafiles[run_curr - 1]

                # Reset simulator
                self.my_simulator.status = "running"
                self.my_simulator.t = self.t0
                self.my_simulator.observation = self.state_full_init

                if self.ctrl_mode != "nominal":
                    self.my_ctrl_benchm.reset(self.t0)
                else:
                    self.my_ctrl_nominal.reset(self.t0)

                accum_obj = 0

    def pipeline_execution(self, args={}, save_trajectory=False):
        self.setup_env(args)
        self.system_initialization()
        self.state_predictor_initialization()
        self.controller_initialization()
        self.simulator_initialization()
        self.logger_initialization()

        if self.is_visualization & ~save_trajectory:
            self.main_loop_visual()
        else:
            self.main_loop_raw(save_trajectory)

    def generate_trajectory(self):
        self.pipeline_execution(save_trajectory=True)
        with open("./refs/trajectory_3wrobot.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


if __name__ == "__main__":
    env_to_pickle()
    PresetPipeline().generate_trajectory()
