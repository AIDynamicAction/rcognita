import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
from npcasadi_api import SymbolicHandler
import numpy as np
from abc import ABC, abstractmethod
from models import ModelPolynomial
from casadi import Function
from tabulate import tabulate


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
        self.g_actor_values = []

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
        self.w_actor = npcsd.zeros(self.dim_actor)

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
            self.action_prev = self.action_min / 10
            self.action_sqn_init = npcsd.rep_mat(self.action_min / 10, 1, self.Nactor)
        else:
            self.action_prev = action_init
            self.action_sqn_init = npcsd.rep_mat(action_init, 1, self.Nactor)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_MPC_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :].T, my_action_sqn[k, :].T, is_symbolic=is_symbolic
            )
        return J

    def compute_RQL_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor - 1):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :].T, my_action_sqn[k, :].T, is_symbolic=is_symbolic
            )
        J += self._critic(
            self._critic.weights,
            observation_sqn[-1, :].T,
            my_action_sqn[-1, :].T,
            is_symbolic=is_symbolic,
        )
        return J

    def compute_SQL_cost(self, observation_sqn, my_action_sqn, is_symbolic=False):
        J = 0
        for k in range(self.Nactor):
            Q = self._critic(
                self._critic.weights,
                observation_sqn[k, :].T,
                my_action_sqn[k, :].T,
                is_symbolic=is_symbolic,
            )

            J += Q
        return J

    def compute_VI_cost(self, observation_sqn, action, is_symbolic=False):
        J = self.stage_obj(
            observation_sqn[0, :].T, action.T, is_symbolic=is_symbolic
        ) + self._critic(
            self._critic.weights, observation_sqn[1, :].T, is_symbolic=is_symbolic,
        )
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
            J = self.compute_MPC_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )
        elif (
            self.ctrl_mode == "RQL"
        ):  # RL: Q-learning with Ncritic-1 roll-outs of stage objectives
            J = self.compute_RQL_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )
        elif self.ctrl_mode == "SQL":  # RL: stacked Q-learning
            J = self.compute_SQL_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )

        elif self.ctrl_mode == "RLSTAB":
            J = self.compute_VI_cost(
                observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            )
            # J = self.compute_MPC_cost(
            #     observation_sqn, my_action_sqn, is_symbolic=is_symbolic
            # )

        return J

    def _actor_optimizer(
        self, observation, is_symbolic=False, constraints=(), return_grad=False, t=None
    ):
        npcsd = SymbolicHandler(is_symbolic)

        my_action_sqn_init = npcsd.reshape(
            npcsd.array(self.action_prev, ignore=True, array_type="SX"),
            [self.Nactor * self.dim_input,],
        )

        cost_function, symbolic_var = npcsd.create_cost_function(
            self._actor_cost, observation, x0=my_action_sqn_init
        )

        if isinstance(constraints, tuple) and len(constraints) > 0:
            constraints = npcsd.concatenate(
                tuple([func(symbolic_var) for func in constraints])
            )
        elif isinstance(constraints, type(lambda x: 0)):
            constraints = constraints(symbolic_var)

        action_sqn = self.actor_optimizer.optimize(
            cost_function,
            my_action_sqn_init,
            constraints=constraints,
            symbolic_var=symbolic_var,
        )

        ##### DEBUG
        g1 = Function("g1", [symbolic_var], [constraints])
        self.g_actor_values.append([g1(action_sqn), t])

        # row_header = ["g1"]
        # row_data = [g1(action_sqn)]
        # row_format = "8.3f"
        # table = tabulate(
        #     [row_header, row_data],
        #     floatfmt=row_format,
        #     headers="firstrow",
        #     tablefmt="grid",
        # )
        # print(table)
        ##### DEBUG

        if return_grad:
            return (
                action_sqn[: self.dim_input],
                npcsd.autograd(cost_function, symbolic_var),
            )
        else:
            return action_sqn[: self.dim_input]

    def forward(self, w_actor, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic=is_symbolic)

        w_actor_reshaped = npcsd.reshape(
            w_actor, (self.dim_input, self.dim_actor_per_input)
        )

        result = self.actor_model(
            w_actor_reshaped, observation, observation, is_symbolic=is_symbolic
        )

        return result


class Critic(ABC):
    def __init__(
        self,
        Ncritic,
        dim_input,
        dim_output,
        buffer_size,
        critic_optimizer=None,
        critic_model=ModelPolynomial(model_name="quad-nomix"),
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
        self.g_critic_values = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def _critic_cost(self):
        pass

    def _critic_optimizer(
        self, constraints=(), is_symbolic=False, return_grad=False, t=None
    ):
        npcsd = SymbolicHandler(is_symbolic)
        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred._critic_cost`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        cost_function, symbolic_var = npcsd.create_cost_function(
            self._critic_cost, x0=npcsd.array(self.weights_prev)
        )

        if isinstance(constraints, tuple):
            if len(constraints) > 0:
                constraints = npcsd.concatenate(
                    tuple([func(symbolic_var) for func in constraints])
                )
        else:
            constraints = constraints(symbolic_var)

        weights = self.critic_optimizer.optimize(
            cost_function,
            self.weights_init,
            constraints=constraints,
            symbolic_var=symbolic_var,
        )

        #### DEBUG

        g2 = Function("g2", [symbolic_var], [constraints])
        self.g_critic_values.append([g2(weights), t])

        # g1 = Function("g1", [symbolic_var], [constraints])

        # row_header = ["g1"]
        # row_data = [g1(weights)]
        # row_format = "8.3f"
        # table = tabulate(
        #     [row_header, row_data],
        #     floatfmt=row_format,
        #     headers="firstrow",
        #     tablefmt="grid",
        # )
        # print(table)
        #### DEBUG

        if return_grad:
            return weights, npcsd.autograd(cost_function, symbolic_var)
        else:
            return weights

    @abstractmethod
    def forward(self):
        pass

    def grad_observation(self, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)

        observation_symbolic = npcsd.array_symb(
            npcsd.shape(npcsd.array(observation)), literal="x"
        )
        weights_symbolic = npcsd.array_symb(
            npcsd.shape(npcsd.array(self.weights)), literal="w"
        )

        critic_func = self.forward(
            weights_symbolic, observation_symbolic, is_symbolic=is_symbolic
        )

        f = Function("f", [observation_symbolic, weights_symbolic], [critic_func])

        gradient = npcsd.autograd(f, observation_symbolic, weights_symbolic)

        gradient_evaluated = gradient(observation, weights_symbolic)

        # Lie_derivative = npcsd.dot(v, gradient(observation_symbolic))
        return gradient_evaluated, weights_symbolic


class CriticValue(Critic):
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

        self.weights_prev = self.Wmin
        self.weights_init = np.ones(self.dim_critic)
        self.weights = self.weights_init

    def forward(self, weights, observation, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            critic_res = self.critic_model(
                weights,
                npcsd.array(observation, array_type="SX"),
                npcsd.array([], array_type="SX"),
                is_symbolic=is_symbolic,
            )
        else:
            observation_new = observation - self.observation_target
            critic_res = self.critic_model(
                weights,
                observation_new.T,
                npcsd.array([], array_type="SX"),
                is_symbolic=is_symbolic,
            )

        return critic_res

    def _critic_cost(self, weights, is_symbolic=False):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0

        for k in range(self.Ncritic - 1, 0, -1):
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]

            # Temporal difference

            critic_prev = self.forward(
                weights, observation_prev, is_symbolic=is_symbolic
            )
            critic_next = self.forward(
                self.weights_prev, observation_next, is_symbolic=is_symbolic
            )

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.stage_obj(observation_prev, action_prev)
            )

            Jc += 1 / 2 * e ** 2

        return Jc


class CriticActionValue(Critic):
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

        self.weights_prev = self.Wmin
        self.weights_init = np.ones(self.dim_critic)
        self.weights = self.weights_init

    def forward(self, weights, observation, action, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """
        if npcsd.shape(observation)[0] == 1:
            observation = npcsd.array(observation, array_type="SX").T
            action = npcsd.array(action, array_type="SX").T
            # self.observation_target = npcsd.array(self.observation_target).T

        if self.observation_target == []:
            critic_res = self.critic_model(
                weights, observation, action, is_symbolic=is_symbolic,
            )
        else:
            observation_new = observation - self.observation_target
            critic_res = self.critic_model(
                weights, observation_new, action, is_symbolic=is_symbolic,
            )

        return critic_res

    def _critic_cost(self, weights, is_symbolic=False):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        Jc = 0

        for k in range(self.Ncritic - 1, 0, -1):
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # Temporal difference

            critic_prev = self.forward(
                weights, observation_prev.T, action_prev.T, is_symbolic=is_symbolic
            )
            critic_next = self.forward(
                self.weights_prev,
                observation_next,
                action_next,
                is_symbolic=is_symbolic,
            )

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.stage_obj(observation_prev.T, action_prev.T)
            )

            Jc += 1 / 2 * e ** 2

        return Jc
