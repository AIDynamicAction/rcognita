import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
import numpy as np
from .utilities import rc, NUMPY, CASADI, TORCH
from abc import ABC, abstractmethod
from casadi import Function
from tabulate import tabulate
import scipy as sp
from functools import partial
import torch
from copy import deepcopy
from multiprocessing import Pool


class Critic(ABC):
    def __init__(
        self,
        Ncritic,
        dim_input,
        dim_output,
        buffer_size,
        optimizer=None,
        model=None,
        running_obj=[],
        gamma=1,
        observation_target=[],
    ):
        if optimizer.engine == "Torch":
            self.typing = TORCH
        elif optimizer.engine == "CasADi":
            self.typing = CASADI
        else:
            self.typing = NUMPY

        self.Ncritic = Ncritic
        self.buffer_size = buffer_size
        self.Ncritic = np.min([self.Ncritic, self.buffer_size - 1])

        self.action_buffer = rc.zeros([buffer_size, dim_input], rc_type=self.typing)
        self.observation_buffer = rc.zeros(
            [buffer_size, dim_output], rc_type=self.typing
        )
        self.observation_target = observation_target

        self.gamma = gamma
        self.running_obj = running_obj
        self.optimizer = optimizer

        self.g_critic_values = []

        self.model = model

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def objDecorator(objective):
        def wrapper(self, buffer, weights=None):
            if weights:
                print("Weights passed")
                self.model.weights = weights

            return objective(buffer)

        return wrapper

    def update(self, constraint_functions=(), t=None):

        """
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.RLController.objective`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        if self.optimizer.engine == "CasADi":
            self._CasADi_update(constraint_functions)

        elif self.optimizer.engine == "SciPy":
            self._SciPy_update(constraint_functions)

        elif self.optimizer.engine == "Torch":
            self._Torch_update()

    def update_buffers(self, observation, action):
        self.action_buffer = rc.push_vec(
            self.action_buffer, rc.array(action, prototype=self.action_buffer)
        )
        self.observation_buffer = rc.push_vec(
            self.observation_buffer,
            rc.array(observation, prototype=self.observation_buffer),
        )

    def grad_observation(self, observation):

        observation_symbolic = rc.array_symb(rc.shape(observation), literal="x")
        weights_symbolic = rc.array_symb(rc.shape(self.weights), literal="w")

        critic_func = self.forward(weights_symbolic, observation_symbolic)

        f = Function("f", [observation_symbolic, weights_symbolic], [critic_func])

        gradient = rc.autograd(f, observation_symbolic, weights_symbolic)

        gradient_evaluated = gradient(observation, weights_symbolic)

        # Lie_derivative = rc.dot(v, gradient(observation_symbolic))
        return gradient_evaluated, weights_symbolic

    def _SciPy_update(self, constraint_functions=()):

        weights_init = self.model.weights

        constraints = ()
        bounds = [self.model.Wmin, self.model.Wmax]
        experience_replay = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(
            experience_replay, weights=weights
        )

        if constraint_functions:
            constraints = sp.optimize.NonlinearConstraint(
                partial(
                    self.create_constraints, constraint_functions=constraint_functions,
                ),
                -np.inf,
                0,
            )

        optimized_weights = self.optimizer.optimize(
            cost_function, weights_init, bounds, constraints=constraints,
        )

        self.model.weights = optimized_weights

    def _CasADi_update(self, constraint_functions=()):

        weights_init = rc.DM(self.model_prev.weights)
        symbolic_var = rc.array_symb(tup=rc.shape(weights_init), prototype=weights_init)

        constraints = ()
        bounds = [self.model.Wmin, self.model.Wmax]
        experience_replay = {
            "observation_buffer": self.observation_buffer,
            "action_buffer": self.action_buffer,
        }

        cost_function = lambda weights: self.objective(
            experience_replay, weights=weights
        )

        cost_function = rc.lambda2symb(cost_function, symbolic_var)

        if constraint_functions:
            constraints = self.create_constraints(constraint_functions, symbolic_var)

        optimized_weights = self.optimizer.optimize(
            cost_function,
            weights_init,
            bounds,
            constraints=constraints,
            symbolic_var=symbolic_var,
        )

        self.model.weights = optimized_weights

    def _Torch_update(self):

        experience_replay = {
            "observation_buffer": torch.tensor(self.observation_buffer),
            "action_buffer": torch.tensor(self.action_buffer),
        }

        self.optimizer.optimize(
            objective=self.objective, model=self.model, model_input=experience_replay,
        )

        self.model.soft_update(1)

    @abstractmethod
    def objective(self):
        pass

    @abstractmethod
    def forward(self):
        pass


class CriticValue(Critic):
    def forward(self, observation):

        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target != []:
            observation = observation - self.observation_target

        result = self.model(observation)

        return result

    def objective(self, data_buffer, weights=None):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """
        observation_buffer = data_buffer.observation_buffer
        action_buffer = data_buffer.action_buffer

        Jc = 0

        for k in range(self.buffer_size - 1, self.buffer_size - self.Ncritic, -1):
            observation_prev = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_prev = action_buffer[k - 1, :]

            # Temporal difference

            critic_prev = self.model(observation_prev, weights)
            critic_next = self.model(observation_next, use_fixed_weights=True)

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.running_obj(observation_prev, action_prev)
            )

            Jc += 1 / 2 * e ** 2

        return Jc


class CriticActionValue(Critic):
    def forward(self, observation, action, use_fixed_weights=False):

        """
        Critic: a routine that models something related to the objective, e.g., value function, Q-function, advantage etc.
        
        Currently, this implementation is for linearly parametrized models.

        """

        if self.observation_target != []:
            observation = observation - self.observation_target

        chi = rc.concatenate((observation, action))
        result = self.model(chi, use_fixed_weights=use_fixed_weights)

        return result

    def objective(self, data_buffer, weights=None):
        """
        Cost function of the critic.
        
        Currently uses value-iteration-like method.  
        
        Customization
        -------------        
        
        Introduce your critic part of an RL algorithm here. Don't forget to provide description in the class documentation. 
       
        """

        observation_buffer = data_buffer["observation_buffer"]
        action_buffer = data_buffer["action_buffer"]

        Jc = 0

        for k in range(self.buffer_size - 1, self.buffer_size - self.Ncritic, -1):
            observation_prev = observation_buffer[k - 1, :]
            observation_next = observation_buffer[k, :]
            action_prev = action_buffer[k - 1, :]
            action_next = action_buffer[k, :]

            # Temporal difference
            chi_prev = rc.concatenate((observation_prev, action_prev))
            chi_next = rc.concatenate((observation_next, action_next))

            critic_prev = self.model(chi_prev, weights)
            critic_next = self.model(chi_next, use_fixed_weights=True)

            e = (
                critic_prev
                - self.gamma * critic_next
                - self.running_obj(observation_prev, action_prev)
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
        This method is merely a wrapper for an optimizer that minimizes :func:`~controllers.RLController.objective`.

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP, trust-constr, Powell

        weights_init = self.model_prev.weights

        constraints = ()
        bounds = [self.Wmin, self.Wmax]

        observation = self.observation_buffer[-1, :]

        def constr_stab_decay_w(weights, observation):

            action_safe = self.safe_ctrl.compute_action(observation)

            observation_next = self.state_predictor.predict_state(
                observation, action_safe
            )

            critic_curr = self.forward(observation, self.model_prev.weights)
            critic_next = self.forward(observation_next, weights)

            return (
                critic_next
                - critic_curr
                + self.state_predictor.pred_step_size * self.safe_decay_rate
            )

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = rc.func_to_lambda_with_params(
                self.objective, var_prototype=weights_init
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var
                )

            lambda_constr = (
                lambda weights: constr_stab_decay_w(weights, observation) - self.eps
            )

            constraints += (rc.lambda2symb(lambda_constr, symbolic_var),)

            optimized_weights = self.optimizer.optimize(
                cost_function,
                weights_init,
                bounds,
                constraints=constraints,
                symbolic_var=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = rc.func_to_lambda_with_params(self.objective)

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
    def __init__(self):
        pass

    def forward(self, observation, action, weights):
        pass

    def objective(self, weights):
        pass

    def get_optimized_weights(self, constraint_functions=(), t=None):
        pass

    def update_buffers(self, observation, action):
        pass

    def update(self, constraint_functions=(), t=None):
        pass


class CriticTabular(Critic):
    def __init__(self, dim_state_space, running_obj, state_predictor, model, gamma=1):

        self.value_table = rc.zeros(dim_state_space)
        self.action_table = rc.zeros(dim_state_space)
        self.running_obj = running_obj
        self.state_predictor = state_predictor
        self.model = model
        self.gamma = gamma

    def forward(self, observation, use_fixed_weights=False):
        return self.model(observation)

    def update_single_cell(self, x):
        action = self.model.table[x]
        self.model.table[x] = (
            self.running_obj(x, action)
            + self.gamma
            * self.model.table[self.state_predictor.predict_state(x, action)]
        )

    def update(self):
        with Pool(self.n_process) as p:
            self.model.update_values(
                p.map(
                    self.update_single_cell,
                    np.nditer(self.model.table, flags=["external_loop"]),
                )[0]
            )

    def objective(self):
        pass
        # self.value_table =  pool.map(lambda row: self.row_value_update(action, row), new_table)

