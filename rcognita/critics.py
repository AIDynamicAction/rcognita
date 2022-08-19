import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import numpy as np
from .utilities import nc
from abc import ABC, abstractmethod
from models import ModelPolynomial
from casadi import Function
from tabulate import tabulate
import scipy as sp
from functools import partial


class Critic(ABC):
    def __init__(
        self,
        Ncritic,
        dim_input,
        dim_output,
        buffer_size,
        optimizer=None,
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
        self.optimizer = optimizer
        self.critic_model = critic_model
        self.g_critic_values = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def objective(self):
        pass

    def update_buffers(self, observation, action):
        self.action_buffer = nc.push_vec(self.action_buffer, action)

        self.observation_buffer = nc.push_vec(self.observation_buffer, observation)

    def get_optimized_weights(self, constraint_functions=(), t=None):

        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred.objective`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        weights_init = self.weights_prev

        constraints = ()
        bounds = [self.Wmin, self.Wmax]

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = nc.func_to_lambda_with_params(
                self.objective, x0=weights_init, is_symbolic=True
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var
                )

            optimized_weights = self.optimizer.optimize(
                cost_function,
                weights_init,
                bounds,
                constraints=constraints,
                symbolic_var=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = nc.func_to_lambda_with_params(self.objective)

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                    ),
                    -np.inf,
                    0,
                )

            optimized_weights = self.optimizer.optimize(
                cost_function, weights_init, bounds, constraints=constraints,
            )

        #### DEBUG

        # g2 = Function("g2", [symbolic_var], [constraints])
        # self.g_critic_values.append([g2(weights), t])

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

        return optimized_weights

    @abstractmethod
    def forward(self):
        pass

    def grad_observation(self, observation):

        observation_symbolic = nc.array_symb(nc.shape(observation), literal="x")
        weights_symbolic = nc.array_symb(nc.shape(self.weights), literal="w")

        critic_func = self.forward(weights_symbolic, observation_symbolic)

        f = Function("f", [observation_symbolic, weights_symbolic], [critic_func])

        gradient = nc.autograd(f, observation_symbolic, weights_symbolic)

        gradient_evaluated = gradient(observation, weights_symbolic)

        # Lie_derivative = nc.dot(v, gradient(observation_symbolic))
        return gradient_evaluated, weights_symbolic


class CriticValue(Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.critic_model.model_name == "quad-lin":
            self.dim_critic = int(
                (self.dim_output + 1) * self.dim_output / 2 + self.dim_output
            )
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quadratic":
            self.dim_critic = int((self.dim_output + 1) * self.dim_output / 2)
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)
        elif self.critic_model.model_name == "quad-nomix":
            self.dim_critic = self.dim_output
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)

        self.weights_prev = np.ones(self.dim_critic)
        self.weights_init = np.ones(self.dim_critic)
        self.weights = self.weights_init

    def forward(self, observation, weights):

        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            critic_res = self.critic_model(observation, np.array([]), weights)
        else:
            observation_new = observation - self.observation_target
            critic_res = self.critic_model(observation_new, np.array([]), weights)

        return critic_res

    def objective(self, weights):
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

            # Temporal difference

            critic_prev = self.forward(observation_prev, weights)
            critic_next = self.forward(observation_next, self.weights_prev)

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
            self.Wmin = np.zeros(self.dim_critic)
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
            self.Wmin = np.zeros(self.dim_critic)
            self.Wmax = 1e3 * np.ones(self.dim_critic)

        self.weights_prev = np.ones(self.dim_critic)
        self.weights_init = np.ones(self.dim_critic)
        self.weights = self.weights_init

    def forward(self, observation, action, weights):

        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target == []:
            result = self.critic_model(observation, action, weights)
        else:
            observation_diff = observation - self.observation_target
            result = self.critic_model(observation_diff, action, weights)

        return result

    def objective(self, weights):
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

            critic_prev = self.forward(observation_prev, action_prev, weights)
            critic_next = self.forward(observation_next, action_next, self.weights_prev)

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.stage_obj(observation_prev, action_prev)
            )

            Jc += 1 / 2 * e ** 2

        return Jc


class CriticSTAG(CriticValue):
    def __init__(
        self,
        safe_decay_rate=1e-4,
        safe_ctrl=[],
        state_predictor=[],
        *args,
        eps=0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.safe_decay_rate = safe_decay_rate
        self.safe_ctrl = safe_ctrl
        self.state_predictor = state_predictor
        self.eps = eps

    def get_optimized_weights(self, constraint_functions=(), t=None):

        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.CtrlOptPred.objective`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        weights_init = self.weights_prev

        constraints = ()
        bounds = [self.Wmin, self.Wmax]

        observation = self.observation_buffer[-1, :]

        def constr_stab_decay_w(weights, observation):

            action_safe = self.safe_ctrl.compute_action(observation)

            observation_next = self.state_predictor.predict_state(
                observation, action_safe
            )

            critic_curr = self.forward(observation, self.weights_prev)
            critic_next = self.forward(observation_next, weights)

            return (
                critic_next
                - critic_curr
                + self.state_predictor.pred_step_size * self.safe_decay_rate
            )

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = nc.func_to_lambda_with_params(
                self.objective, x0=weights_init, is_symbolic=True
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var
                )

            lambda_constr = (
                lambda weights: constr_stab_decay_w(weights, observation) - self.eps
            )

            constraints += (nc.lambda2symb(lambda_constr, symbolic_var),)

            optimized_weights = self.optimizer.optimize(
                cost_function,
                weights_init,
                bounds,
                constraints=constraints,
                symbolic_var=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = nc.func_to_lambda_with_params(self.objective)

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                    ),
                    -np.inf,
                    0,
                )

            my_constraints = sp.optimize.NonlinearConstraint(
                lambda weights: constr_stab_decay_w(weights, observation),
                -np.inf,
                self.eps,
            )

            # my_constraints = ()

            optimized_weights = self.optimizer.optimize(
                cost_function, weights_init, bounds, constraints=my_constraints,
            )

        return optimized_weights


class CriticMPC(Critic):
    def forward(self, observation, action, weights):
        pass

    def objective(self, weights):
        pass

    def get_optimized_weights(self, constraint_functions=(), t=None):
        pass
