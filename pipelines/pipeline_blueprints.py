from abc import ABCMeta, abstractmethod
from rcognita import (
    controllers,
    simulator,
    state_predictors,
    optimizers,
    models,
    objectives,
)
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


class AbstractPipeline(metaclass=ABCMeta):
    @property
    @abstractmethod
    def config(self):
        return self.config

    def load_config(self):
        self.env_config = self.config()

    def setup_env(self):
        self.__dict__.update(self.env_config.get_env())
        self.trajectory = []

    def config_to_pickle(self):
        self.env_config.config_to_pickle()

    @abstractmethod
    def initialize_system(self):
        pass

    @abstractmethod
    def initialize_state_predictor(self):
        pass

    @abstractmethod
    def initialize_controller(self):
        pass

    @abstractmethod
    def initialize_controller(self):
        pass

    @abstractmethod
    def initialize_simulator(self):
        pass

    @abstractmethod
    def initialize_logger(self):
        pass

    @abstractmethod
    def main_loop_raw(self):
        pass

    @abstractmethod
    def execute_pipeline(self):
        pass

    def upd_accum_obj(self, observation, action, delta):

        """
        Sample-to-sample accumulated (summed up or integrated) stage objective. This can be handy to evaluate the performance of the agent.
        If the agent succeeded to stabilize the system, ``accum_obj`` would converge to a finite value which is the performance mark.
        The smaller, the better (depends on the problem specification of course - you might want to maximize objective instead).
        
        """

        self.accum_obj_val += self.stage_objective(observation, action) * delta


class PipelineWithDefaults(AbstractPipeline):
    def initialize_state_predictor(self):
        self.state_predictor = state_predictors.EulerStatePredictor(
            self.pred_step_size,
            self.my_sys._state_dyn,
            self.my_sys.out,
            self.dim_output,
            self.Nactor,
        )

    def initialize_models(self):
        if self.critic_struct == "NN":
            self.critic_model = models.ModelNN(
                self.dim_output, self.dim_input, dim_hidden=2
            )
        else:
            self.critic_model = models.ModelPolynomial(model_name=self.critic_struct)
        self.stage_obj_model = models.ModelQuadForm(R1=self.R1, R2=self.R2)

    def initialize_objectives(self):

        self.stage_objective = objectives.StageObjective(
            stage_obj_model=self.stage_obj_model
        )

    def initialize_optimizers(self):
        opt_options = {
            "maxiter": 200,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }
        self.actor_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options
        )
        self.critic_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options,
        )

    def initialize_actor_critic(self):

        if self.control_mode == "RLSTAB":
            self.critic = CriticSTAG(
                Ncritic=self.Ncritic,
                dim_input=self.dim_input,
                dim_output=self.dim_output,
                buffer_size=self.buffer_size,
                stage_obj=self.stage_objective,
                gamma=self.gamma,
                optimizer=self.critic_optimizer,
                critic_model=self.critic_model,
                safe_ctrl=self.my_ctrl_nominal,
                state_predictor=self.state_predictor,
            )
            Actor = ActorSTAG

        else:
            self.critic = CriticActionValue(
                Ncritic=self.Ncritic,
                dim_input=self.dim_input,
                dim_output=self.dim_output,
                buffer_size=self.buffer_size,
                stage_obj=self.stage_objective,
                gamma=self.gamma,
                optimizer=self.critic_optimizer,
                critic_model=self.critic_model,
            )
            if self.control_mode == "MPC":
                Actor = ActorMPC
            elif self.control_mode == "RQL":
                Actor = ActorRQL
            elif self.control_mode == "SQL":
                Actor = ActorSQL

        self.actor = Actor(
            self.Nactor,
            self.dim_input,
            self.dim_output,
            self.control_mode,
            self.control_bounds,
            state_predictor=self.state_predictor,
            optimizer=self.actor_optimizer,
            critic=self.critic,
            stage_obj=self.stage_objective,
        )

    def initialize_controller(self):
        self.my_ctrl_benchm = controllers.CtrlOptPred(
            action_init=self.action_init,
            t0=self.t0,
            sampling_time=self.dt,
            pred_step_size=self.pred_step_size,
            state_dyn=self.my_sys._state_dyn,
            sys_out=self.my_sys.out,
            prob_noise_pow=self.prob_noise_pow,
            is_est_model=self.is_est_model,
            model_est_stage=self.model_est_stage,
            model_est_period=self.model_est_period,
            buffer_size=self.buffer_size,
            model_order=self.model_order,
            model_est_checks=self.model_est_checks,
            critic_period=self.critic_period,
            actor=self.actor,
            critic=self.critic,
            observation_target=[],
        )

    def initialize_simulator(self):
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
            max_step=self.dt / 10,
            first_step=1e-6,
            atol=self.atol,
            rtol=self.rtol,
            is_disturb=self.is_disturb,
            is_dyn_ctrl=self.is_dyn_ctrl,
        )

    def execute_pipeline(self, **kwargs):
        self.load_config()
        self.setup_env()
        self.__dict__.update(kwargs)
        self.initialize_system()
        self.initialize_state_predictor()
        self.initialize_models()
        self.initialize_objectives()
        self.initialize_optimizers()
        self.initialize_safe_controller()
        self.initialize_actor_critic()
        self.initialize_controller()
        self.initialize_simulator()
        self.initialize_logger()
        if not self.no_visual and not self.save_trajectory:
            self.main_loop_visual()
        else:
            self.main_loop_raw()

