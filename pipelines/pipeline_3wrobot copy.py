import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)


import rcognita
from pipeline_blueprints import AbstractPipeline

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
    models,
)
from rcognita.rl_tools import Actor, CriticActionValue


class Pipeline3WRobot(AbstractPipeline):
    def rl_components_initialization(self):

        self.v_critic = CriticActionValue(
            Ncritic=self.Ncritic,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            buffer_size=self.buffer_size,
            stage_obj=self.objectives.stage_obj,
            gamma=self.gamma,
            critic_optimizer=self.critic_optimizer,
            critic_model=models.ModelPolynomial(model_name="quad-nomix"),
        )

        self.v_actor = Actor(
            self.Nactor,
            self.dim_input,
            self.dim_output,
            self.ctrl_mode,
            state_predictor=self.state_predictor,
            actor_optimizer=self.actor_optimizer,
            critic=self.v_critic,
            stage_obj=self.objectives.stage_obj,
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

        self.my_ctrl_RL_stab = controllers.CtrlRLStab(
            action_init=self.action_init,
            t0=self.t0,
            sampling_time=self.dt,
            pred_step_size=self.pred_step_size,
            state_dyn=self.my_sys._state_dyn,
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
            critic_period=self.critic_period,
            actor=self.v_actor,
            critic=self.v_critic,
            safe_ctrl=self.my_ctrl_nominal,
            safe_decay_rate=1e-4,
            stage_obj_pars=[self.R1],
            observation_target=[],
        )

        self.my_ctrl_benchm = self.my_ctrl_RL_stab


def main():
    Pipeline3WRobot().pipeline_execution()


if __name__ == "__main__":
    main()
