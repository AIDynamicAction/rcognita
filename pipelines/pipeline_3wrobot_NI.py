import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import pathlib
import warnings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
import rcognita
import numpy as np

from config_blueprints import Config3WRobotNI
from pipeline_blueprints import PipelineWithDefaults

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita import (
    controllers,
    visuals,
    simulator,
    systems,
    loggers,
    state_predictors,
    optimizers,
    objectives,
    models,
)
from datetime import datetime
from rcognita.utilities import on_key_press
from rcognita.actors import (
    ActorSTAG,
    ActorMPC,
    ActorRQL,
    ActorSQL,
)

from rcognita.critics import (
    CriticActionValue,
    CriticSTAG,
)


class Pipeline3WRobotNI(PipelineWithDefaults):
    config = Config3WRobotNI

    def initialize_system(self):
        self.my_sys = systems.Sys3WRobotNI(
            sys_type="diff_eqn",
            dim_state=self.dim_state,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_disturb=self.dim_disturb,
            pars=[],
            control_bounds=self.control_bounds,
            is_dyn_ctrl=self.is_dyn_ctrl,
            is_disturb=self.is_disturb,
            pars_disturb=np.array([[200 * self.dt, 200 * self.dt], [0, 0], [0.3, 0.3]]),
        )

    def initialize_safe_controller(self):
        self.my_ctrl_nominal = controllers.CtrlNominal3WRobotNI(
            ctrl_gain=0.5,
            control_bounds=self.control_bounds,
            t0=self.t0,
            sampling_time=self.dt,
        )

    def initialize_logger(self):
        if (
            os.path.basename(os.path.normpath(os.path.abspath(os.getcwd())))
            == "presets"
        ):
            self.data_folder = "../simdata"
        else:
            self.data_folder = "simdata"

        pathlib.Path(self.data_folder).mkdir(parents=True, exist_ok=True)

        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%Hh%Mm%Ss")
        self.datafiles = [None] * self.Nruns

        for k in range(0, self.Nruns):
            self.datafiles[k] = (
                self.data_folder
                + "/"
                + self.my_sys.name
                + "__"
                + self.control_mode
                + "__"
                + date
                + "__"
                + time
                + "__run{run:02d}.csv".format(run=k + 1)
            )

            if self.is_log:
                print("Logging data to:    " + self.datafiles[k])

                with open(self.datafiles[k], "w", newline="") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(["System", self.my_sys.name])
                    writer.writerow(["Controller", self.control_mode])
                    writer.writerow(["dt", str(self.dt)])
                    writer.writerow(["state_init", str(self.state_init)])
                    writer.writerow(["is_est_model", str(self.is_est_model)])
                    writer.writerow(["model_est_stage", str(self.model_est_stage)])
                    writer.writerow(
                        [
                            "model_est_period_multiplier",
                            str(self.model_est_period_multiplier),
                        ]
                    )
                    writer.writerow(["model_order", str(self.model_order)])
                    writer.writerow(["prob_noise_pow", str(self.prob_noise_pow)])
                    writer.writerow(["Nactor", str(self.Nactor)])
                    writer.writerow(
                        [
                            "pred_step_size_multiplier",
                            str(self.pred_step_size_multiplier),
                        ]
                    )
                    writer.writerow(["buffer_size", str(self.buffer_size)])
                    writer.writerow(["stage_obj_struct", str(self.stage_obj_struct)])
                    writer.writerow(["R1_diag", str(self.R1_diag)])
                    writer.writerow(["R2_diag", str(self.R2_diag)])
                    writer.writerow(["Ncritic", str(self.Ncritic)])
                    writer.writerow(["gamma", str(self.gamma)])
                    writer.writerow(
                        ["critic_period_multiplier", str(self.critic_period_multiplier)]
                    )
                    writer.writerow(["critic_struct", str(self.critic_struct)])
                    writer.writerow(["actor_struct", str(self.actor_struct)])
                    writer.writerow(
                        [
                            "t [s]",
                            "x [m]",
                            "y [m]",
                            "alpha [rad]",
                            "stage_obj",
                            "accum_obj",
                            "v [m/s]",
                            "omega [rad/s]",
                        ]
                    )

        # Do not display annoying warnings when print is on
        if not self.no_print:
            warnings.filterwarnings("ignore")

        self.my_logger = loggers.Logger3WRobotNI()

    def main_loop_visual(self):
        state_full_init = self.my_simulator.state_full

        my_animator = visuals.Animator3WRobotNI(
            objects=(
                self.my_simulator,
                self.my_sys,
                self.my_ctrl_nominal,
                self.my_ctrl_benchm,
                self.datafiles,
                self.my_logger,
                self.actor_optimizer,
                self.critic_optimizer,
                self.stage_objective,
            ),
            pars=(
                self.state_init,
                self.action_init,
                self.t0,
                self.t1,
                state_full_init,
                self.xMin,
                self.xMax,
                self.yMin,
                self.yMax,
                self.control_mode,
                self.action_manual,
                self.v_min,
                self.omega_min,
                self.v_max,
                self.omega_max,
                self.Nruns,
                self.no_print,
                self.is_log,
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

    def main_loop_raw(self):

        run_curr = 1
        self.accum_obj_val = 0
        datafile = self.datafiles[0]
        t = t_prev = 0

        while True:

            self.my_simulator.sim_step()

            t_prev = t

            (t, _, observation, state_full,) = self.my_simulator.get_sim_step_data()

            delta_t = t - t_prev
            # DEBUG ===================================================================
            # if self.save_trajectory:
            #     self.trajectory.append([state_full.extend(t)])
            # /DEBUG ===================================================================

            if self.control_mode == "nominal":
                action = self.my_ctrl_nominal.compute_action_sampled(t, observation)
            else:
                action = self.my_ctrl_benchm.compute_action_sampled(t, observation)

            self.my_sys.receive_action(action)

            xCoord = state_full[0]
            yCoord = state_full[1]
            alpha = state_full[2]

            stage_obj = self.stage_objective(observation, action)
            self.upd_accum_obj(observation, action, delta_t)
            accum_obj = self.accum_obj_val

            if not self.no_print:
                self.my_logger.print_sim_step(
                    t, xCoord, yCoord, alpha, stage_obj, accum_obj, action
                )

            if self.is_log:
                self.my_logger.log_data_row(
                    datafile, t, xCoord, yCoord, alpha, stage_obj, accum_obj, action,
                )

            if t >= self.t1:
                if not self.no_print:
                    print(
                        ".....................................Run {run:2d} done.....................................".format(
                            run=run_curr
                        )
                    )

                run_curr += 1

                if run_curr > self.Nruns:
                    break

                if self.is_log:
                    datafile = self.datafiles[run_curr - 1]

                # Reset simulator
                self.my_simulator.status = "running"
                self.my_simulator.t = self.t0
                self.my_simulator.observation = self.state_full_init

                if self.control_mode != "nominal":
                    self.my_ctrl_benchm.reset(self.t0)
                else:
                    self.my_ctrl_nominal.reset(self.t0)

                accum_obj = 0


if __name__ == "__main__":

    Pipeline3WRobotNI().execute_pipeline()
