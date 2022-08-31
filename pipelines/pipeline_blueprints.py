from abc import ABCMeta, abstractmethod
from rcognita import controllers, simulator, state_predictors, optimizers, objectives

from rcognita.utilities import rc
from rcognita.actors import (
    ActorSTAG,
    ActorMPC,
    ActorRQL,
    ActorSQL,
)

from rcognita.critics import CriticActionValue, CriticSTAG, CriticMPC

from rcognita.models import (
    ModelQuadLin,
    ModelQuadratic,
    ModelQuadNoMix,
    ModelNN,
    ModelQuadForm,
    ModelSS,
)

# from rcognita.estimators import Estimator


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

        self.accum_obj_val += self.running_objective(observation, action) * delta


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
        if self.control_mode == "STAG":
            self.dim_critic_model_input = self.dim_output
        else:
            self.dim_critic_model_input = self.dim_input + self.dim_output

        if self.critic_struct == "NN":
            self.critic_model = ModelNN(self.dim_output, self.dim_input, dim_hidden=3)
        else:
            if self.critic_struct == "quad-lin":
                self.critic_model = ModelQuadLin(self.dim_critic_model_input)
            elif self.critic_struct == "quad-nomix":
                self.critic_model = ModelQuadNoMix(self.dim_critic_model_input)
            elif self.critic_struct == "quadratic":
                self.critic_model = ModelQuadratic(self.dim_critic_model_input)

        if self.actor_struct == "NN":
            self.critic_model = ModelNN(self.dim_output, dim_hidden=3)
        else:
            if self.actor_struct == "quad-lin":
                self.actor_model = ModelQuadLin(self.dim_output)
            elif self.actor_struct == "quad-nomix":
                self.actor_model = ModelQuadNoMix(self.dim_output)
            elif self.actor_struct == "quadratic":
                self.actor_model = ModelQuadratic(self.dim_output)

        self.running_obj_model = ModelQuadForm(R1=self.R1, R2=self.R2)

        A = rc.zeros([self.model_order, self.model_order])
        B = rc.zeros([self.model_order, self.dim_input])
        C = rc.zeros([self.dim_output, self.model_order])
        D = rc.zeros([self.dim_output, self.dim_input])
        x0est = rc.zeros(self.model_order)

        self.model_SS = ModelSS(A, B, C, D, x0est)

    # def estimator_initialization(self):
    #     self.estimator = Estimator(
    #         model_est_checks=self.model_est_checks, model=self.model_SS
    #     )

    def initialize_objectives(self):

        self.running_objective = objectives.RunningObjective(
            running_obj_model=self.running_obj_model
        )

    def initialize_optimizers(self):
        opt_options = {
            "maxiter": 500,
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
                running_obj=self.running_objective,
                gamma=self.gamma,
                optimizer=self.critic_optimizer,
                model=self.critic_model,
                safe_ctrl=self.my_ctrl_nominal,
                state_predictor=self.state_predictor,
            )
            Actor = ActorSTAG

        else:
            if self.control_mode == "MPC":
                Actor = ActorMPC
                self.critic = CriticMPC()
            else:
                self.critic = CriticActionValue(
                    Ncritic=self.Ncritic,
                    dim_input=self.dim_input,
                    dim_output=self.dim_output,
                    buffer_size=self.buffer_size,
                    running_obj=self.running_objective,
                    gamma=self.gamma,
                    optimizer=self.critic_optimizer,
                    model=self.critic_model,
                )
                if self.control_mode == "RQL":
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
            running_obj=self.running_objective,
            model=self.actor_model,
        )

    def initialize_controller(self):
        self.my_ctrl_benchm = controllers.RLController(
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

            if self.save_trajectory:
                self.trajectory.append(rc.concatenate((state_full, t), axis=None))
            if self.control_mode == "nominal":
                action = self.my_ctrl_nominal.compute_action_sampled(t, observation)
            else:
                action = self.my_ctrl_benchm.compute_action_sampled(t, observation)

            self.my_sys.receive_action(action)

            running_obj = self.running_objective(observation, action)
            self.upd_accum_obj(observation, action, delta_t)
            accum_obj = self.accum_obj_val

            if not self.no_print:
                self.my_logger.print_sim_step(
                    t, state_full, action, running_obj, accum_obj
                )

            if self.is_log:
                self.my_logger.log_data_row(
                    datafile, t, state_full, action, running_obj, accum_obj,
                )

            if t >= self.t1:
                if not self.no_print:
                    print(
                        ".....................................Run {run:2d} done.....................................".format(
                            run=run_curr
                        )
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

