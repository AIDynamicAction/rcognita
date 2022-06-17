import os, sys
import pickle5 as pickle
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from configs import config_3wrobot

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)

from rcognita import controllers, visuals, simulator, systems, loggers
from rcognita.utilities import on_key_press


def config_to_pickle():
    with open("./refs/env_3wrobot.pickle", "wb") as env_description_out:
        pickle.dump(config_3wrobot(), env_description_out)

def get_env():
    files = os.listdir("./refs")

    if "env_3wrobot.pickle" not in files:
        config_to_pickle()
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
    config_to_pickle()
    PresetPipeline().generate_trajectory()
