import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
from npcasadi_api import SymbolicHandler
import numpy as np
from utilities import dss_sim
from abc import ABC
from models import ModelPolynomial


class Actor:
    def __init__(
        self,
        Nactor,
        dim_input,
        dim_output,
        ctrl_mode,
        action_init=[],
        state_predictor=[],
        actor_optimizer=None,
        critic=[],
        stage_obj=[],
        actor_model=ModelPolynomial(model_name="quad-lin"),
        gamma=1,
    ):
        self.Nactor = Nactor
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.ctrl_mode = ctrl_mode
        self.actor_optimizer = actor_optimizer
        self._critic = critic
        self.stage_obj = stage_obj
        self.actor_model = actor_model
        self.state_predictor = state_predictor
        self.gamma = gamma

        npcsd = SymbolicHandler(actor_optimizer.is_symbolic)

        if self.actor_model.model_name == "quad-lin":
            self.dim_actor_per_input = int(
                (self.dim_output + 1) * self.dim_output / 2 + self.dim_output
            )
        elif self.actor_model.model_name == "quadratic":
            self.dim_actor_per_input = int((self.dim_output + 1) * self.dim_output / 2)
        elif self.actor_model.model_name == "quad-nomix":
            self.dim_actor_per_input = self.dim_output

        self.dim_actor = self.dim_actor_per_input * self.dim_input

        if self.ctrl_mode != "MPC" and self._critic == []:
            raise ValueError(
                f"Critic should be passed to actor in {self.ctrl_mode} mode"
            )
        elif self.ctrl_mode == "MPC" and self.stage_obj == []:
            raise ValueError(
                f"Stage objective should be passed to actor in {self.ctrl_mode} mode"
            )

        if isinstance(self.actor_optimizer.bounds, list):
            self.action_min = np.array(self.actor_optimizer.bounds[0][: self.dim_input])
        else:
            self.action_min = np.array(self.actor_optimizer.bounds.lb[: self.dim_input])

        if len(action_init) == 0:
            self.action_curr = self.action_min / 10
            self.action_sqn_init = npcsd.rep_mat(self.action_min / 10, 1, self.Nactor)
        else:
            self.action_curr = action_init
            self.action_sqn_init = npcsd.rep_mat(action_init, 1, self.Nactor)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_mpc_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :], my_action_sqn[k, :], is_symbolic=is_symbolic
            )
        return J

    def compute_rql_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor - 1):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :].T, my_action_sqn[k, :].T, is_symbolic=is_symbolic
            )
        J += self._critic(
            observation_sqn[-1, :].T, my_action_sqn[-1, :].T, is_symbolic=is_symbolic
        )
        return J

    def compute_sql_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor):
            Q = self._critic(
                observation_sqn[k, :].T, my_action_sqn[k, :].T, is_symbolic=is_symbolic,
            )

            J += Q
        return J

    def _actor_cost(self, action_sqn, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = npcsd.reshape(action_sqn, [self.Nactor, self.dim_input])

        observation_sqn = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn, is_symbolic=is_symbolic
        )

        if self.ctrl_mode == "MPC":
            J = self.compute_mpc_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )
        elif (
            self.ctrl_mode == "RQL"
        ):  # RL: Q-learning with Ncritic-1 roll-outs of stage objectives
            J = self.compute_rql_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )
        elif self.ctrl_mode == "SQL":  # RL: stacked Q-learning
            J = self.compute_sql_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )

        return J

    def _actor_optimizer(
        self, observation, is_symbolic=False,
    ):
        npcsd = SymbolicHandler(is_symbolic)

        my_action_sqn_init = npcsd.reshape(
            npcsd.array(self.action_sqn_init, ignore=True, array_type="SX"),
            [self.Nactor * self.dim_input,],
        )

        cost_function, symbolic_var = npcsd.create_cost_function(
            self._actor_cost, observation, x0=my_action_sqn_init
        )

        action_sqn = self.actor_optimizer.optimize(
            cost_function, my_action_sqn_init, symbolic_var=symbolic_var,
        )

        return action_sqn[: self.dim_input]

    def forward(self, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic=is_symbolic)
        regressor_actor = self.actor_model(observation, is_symbolic=is_symbolic)
        w_actor_reshaped = npcsd.reshape(
            self.w_actor, (self.dim_input, self.dim_actor_per_input)
        )
        result = w_actor_reshaped @ regressor_actor.T
        return result


class Critic(ABC):
    def __init__(
        self,
        Ncritic,
        dim_input,
        dim_output,
        buffer_size,
        critic_optimizer=None,
        critic_model=ModelPolynomial(model_name="quad-lin"),
        stage_obj=[],
        gamma=1,
        observation_target=[],
    ):
        self.observation_target = observation_target
        self.Ncritic = Ncritic
        self.buffer_size = buffer_size
        self.Ncritic = np.min([self.Ncritic, self.buffer_size - 1])
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.action_buffer = np.zeros([buffer_size, dim_input])
        self.observation_buffer = np.zeros([buffer_size, dim_output])
        self.gamma = gamma
        self.stage_obj = stage_obj
        self.critic_optimizer = critic_optimizer
        self.critic_model = critic_model

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _critic_cost(self, w_critic, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0

        for k in range(self.buffer_size - 1, self.buffer_size - self.Ncritic, -1):
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # Temporal difference

            critic_prev = self.forward(
                observation_prev, action_prev, is_symbolic=is_symbolic
            )
            critic_next = self.forward(
                observation_next, action_next, is_symbolic=is_symbolic,
            )

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.stage_obj(
                    npcsd.array(observation_prev),
                    npcsd.array(action_prev),
                    is_symbolic=is_symbolic,
                )
            )

            Jc += 1 / 2 * e ** 2

        return Jc

    def _critic_optimizer(self, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._critic_cost`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        cost_function, symbolic_var = npcsd.create_cost_function(
            self._critic_cost, x0=npcsd.array(self.w_critic_init)
        )

        w_critic = self.critic_optimizer.optimize(
            cost_function, self.w_critic_init, symbolic_var=symbolic_var,
        )

        return w_critic

    def forward(self, observation, action, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            regressor_critic = self.critic_model(
                npcsd.array(observation, array_type="SX"),
                npcsd.array(action, array_type="SX"),
                is_symbolic=is_symbolic,
            )
        else:
            observation_new = observation - self.observation_target
            regressor_critic = self.critic_model(
                observation_new.T,
                npcsd.array(action, array_type="SX").T,
                is_symbolic=is_symbolic,
            )

        return npcsd.dot(self.w_critic, regressor_critic)


class CriticAction(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.critic_model.model_name == "quad-lin":
            self.dim_critic = int(
                ((self.dim_output + self.dim_input) + 1)
                * (self.dim_output + self.dim_input)
                / 2
                + (self.dim_output + self.dim_input)
            )
            self.Wmin = -1e3 * np.ones(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quadratic":
            self.dim_critic = int(
                ((self.dim_output + self.dim_input) + 1)
                * (self.dim_output + self.dim_input)
                / 2
            )
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quad-nomix":
            self.dim_critic = self.dim_output + self.dim_input
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quad-mix":
            self.dim_critic = int(
                self.dim_output + self.dim_output * self.dim_input + self.dim_input
            )
            self.Wmin = -1e3 * np.ones(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)

        self.w_critic = self.Wmin
        self.w_critic_prev = self.Wmin
        self.w_critic_init = np.ones(self.dim_critic)


class CriticActionValue(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.critic_model.model_name == "quad-lin":
            self.dim_critic = int(
                (self.dim_output + 1) * self.dim_output / 2 + self.dim_output
            )
            self.Wmin = -1e3 * np.ones(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quadratic":
            self.dim_critic = int((self.dim_output + 1) * self.dim_output / 2)
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quad-nomix":
            self.dim_critic = self.dim_output
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)

        self.w_critic = self.Wmin
        self.w_critic_prev = self.Wmin
        self.w_critic_init = np.ones(self.dim_critic)
