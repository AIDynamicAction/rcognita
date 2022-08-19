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


class Actor:
    def __init__(
        self,
        Nactor,
        dim_input,
        dim_output,
        control_mode,
        control_bounds=[],
        action_init=[],
        state_predictor=[],
        optimizer=None,
        critic=[],
        stage_obj=[],
        actor_model=ModelPolynomial(model_name="quad-lin"),
        gamma=1,
    ):
        self.Nactor = Nactor
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.control_mode = control_mode
        self.control_bounds = control_bounds
        self.optimizer = optimizer
        self.critic = critic
        self.stage_obj = stage_obj
        self.actor_model = actor_model
        self.state_predictor = state_predictor
        self.gamma = gamma
        self.g_actor_values = []

        if self.actor_model.model_name == "quad-lin":
            self.dim_actor_per_input = int(
                (self.dim_output + 1) * self.dim_output / 2 + self.dim_output
            )
        elif self.actor_model.model_name == "quadratic":
            self.dim_actor_per_input = int((self.dim_output + 1) * self.dim_output / 2)
        elif self.actor_model.model_name == "quad-nomix":
            self.dim_actor_per_input = self.dim_output

        self.dim_actor = self.dim_actor_per_input * self.dim_input
        self.w_actor = nc.zeros(self.dim_actor)

        if self.control_mode != "MPC" and self.critic == []:
            raise ValueError(
                f"Critic should be passed to actor in {self.control_mode} mode"
            )
        elif self.control_mode == "MPC" and self.stage_obj == []:
            raise ValueError(
                f"Stage objective should be passed to actor in {self.control_mode} mode"
            )

        if isinstance(self.control_bounds, (list, np.ndarray)):
            self.action_min = np.array(self.control_bounds)[:, 0]
        else:
            self.action_min = np.array(self.control_bounds.lb[: self.dim_input])

        if len(action_init) == 0:
            self.action_prev = self.action_min / 10
            self.action_sqn_init = nc.rep_mat(self.action_min / 10, 1, self.Nactor + 1)
        else:
            self.action_prev = action_init
            self.action_sqn_init = nc.rep_mat(action_init, 1, self.Nactor + 1)

        self.action_sqn_min = nc.rep_mat(self.action_min, 1, Nactor + 1)
        self.action_sqn_max = nc.rep_mat(-self.action_min, 1, Nactor + 1)
        self.control_bounds = [self.action_sqn_min, self.action_sqn_max]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def create_constraints(self, constraint_functions, my_action_sqn, observation):
        current_observation = observation

        constraint_violations_result = [0 for _ in range(self.Nactor - 1)]
        constraint_violations_buffer = [0 for _ in constraint_functions]

        for constraint_function in constraint_functions:
            constraint_violations_buffer[0] = constraint_function(current_observation)

        max_constraint_violation = nc.max(constraint_violations_buffer)

        max_constraint_violation = -1
        action_sqn = nc.reshape(my_action_sqn, [self.Nactor, self.dim_input])
        predicted_state = current_observation

        for i in range(1, self.Nactor):

            current_action = action_sqn[i - 1, :]
            current_state = predicted_state

            predicted_state = self.state_predictor.predict_state(
                current_state, current_action
            )

            constraint_violations_buffer = []
            for constraint in constraint_functions:
                constraint_violations_buffer.append(constraint(predicted_state))

            max_constraint_violation = nc.max(constraint_violations_buffer)
            constraint_violations_result[i - 1] = max_constraint_violation

        for i in range(2, self.Nactor - 1):
            constraint_violations_result[i] = nc.if_else(
                constraint_violations_result[i - 1] > 0,
                constraint_violations_result[i - 1],
                constraint_violations_result[i],
            )

        return constraint_violations_result

    def get_optimized_action(self, observation, constraint_functions=(), t=None):

        rep_action_prev = nc.rep_mat(self.action_prev / 10, 1, self.Nactor + 1)

        my_action_sqn_init = nc.reshape(
            rep_action_prev, [(self.Nactor + 1) * self.dim_input,],
        )

        constraints = ()

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = nc.func_to_lambda_with_params(
                self.objective, observation, x0=my_action_sqn_init, is_symbolic=True
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var, observation
                )

            action_sqn_optimized = self.optimizer.optimize(
                cost_function,
                my_action_sqn_init,
                self.control_bounds,
                constraints=constraints,
                symbolic_var=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = nc.func_to_lambda_with_params(
                self.objective, observation, x0=my_action_sqn_init
            )

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_constraints,
                        constraint_functions=constraint_functions,
                        observation=observation,
                    ),
                    -np.inf,
                    0,
                )

            action_sqn_optimized = self.optimizer.optimize(
                cost_function,
                my_action_sqn_init,
                self.control_bounds,
                constraints=constraints,
            )

        ##### DEBUG
        # g1 = Function("g1", [symbolic_var], [constraints])
        # self.g_actor_values.append([g1(action_sqn), t])

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
        return action_sqn_optimized[: self.dim_input]

    def forward(self, w_actor, observation):

        w_actor_reshaped = nc.reshape(
            w_actor, (self.dim_input, self.dim_actor_per_input)
        )

        result = self.actor_model(w_actor_reshaped, observation, observation)

        return result


class ActorMPC(Actor):
    def objective(
        self, action_sqn, observation,
    ):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = nc.reshape(action_sqn, [self.Nactor + 1, self.dim_input])

        observation_sqn = [observation]

        observation_sqn_predicted = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn
        )

        observation_sqn = nc.vstack(
            (nc.reshape(observation, [1, self.dim_output]), observation_sqn_predicted)
        )

        J = 0
        for k in range(self.Nactor):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :].T, my_action_sqn[k, :].T
            )
        return J


class ActorSQL(Actor):
    def objective(
        self, action_sqn, observation,
    ):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = nc.reshape(action_sqn, [self.Nactor + 1, self.dim_input])

        observation_sqn = [observation]

        observation_sqn_predicted = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn
        )

        observation_sqn = nc.vstack(
            (nc.reshape(observation, [1, self.dim_output]), observation_sqn_predicted)
        )

        J = 0
        for k in range(self.Nactor + 1):
            Q = self.critic(
                observation_sqn[k, :], my_action_sqn[k, :], self.critic.weights,
            )

            J += Q
        return J


class ActorRQL(Actor):
    def objective(
        self, action_sqn, observation,
    ):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = nc.reshape(action_sqn, [self.Nactor + 1, self.dim_input])

        observation_sqn = [observation]

        observation_sqn_predicted = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn
        )

        observation_sqn = nc.vstack(
            (nc.reshape(observation, [1, self.dim_output]), observation_sqn_predicted)
        )

        J = 0
        for k in range(self.Nactor):
            J += self.gamma ** k * self.stage_obj(
                observation_sqn[k, :], my_action_sqn[k, :]
            )
        J += self.critic(
            observation_sqn[-1, :], my_action_sqn[-1, :], self.critic.weights
        )
        return J


class ActorVI(Actor):
    def objective(
        self, action_sqn, observation,
    ):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = nc.reshape(action_sqn, [self.Nactor + 1, self.dim_input])

        observation_sqn = [observation]

        observation_sqn_predicted = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn
        )

        observation_sqn = nc.vstack(
            (nc.reshape(observation, [1, self.dim_output]), observation_sqn_predicted)
        )

        J = self.stage_obj(observation_sqn[0, :], my_action_sqn[0, :]) + self.critic(
            observation_sqn[1, :].T, self.critic.weights
        )
        return J


class ActorSTAG(ActorVI):
    def __init__(self, *args, eps=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def get_optimized_action(self, observation, constraint_functions=(), t=None):

        rep_action_prev = nc.rep_mat(self.action_prev / 10, 1, self.Nactor + 1)

        my_action_sqn_init = nc.reshape(
            rep_action_prev, [(self.Nactor + 1) * self.dim_input,],
        )

        constraints = ()

        def constr_stab_decay_action(action, observation):

            # action_safe = self.safe_ctrl.compute_action_vanila(observation)

            observation_next = self.state_predictor.predict_state(observation, action)

            critic_curr = self.critic(observation, self.critic.weights_prev)
            critic_next = self.critic(observation_next, self.critic.weights)

            return (
                critic_next
                - critic_curr
                + self.state_predictor.pred_step_size * self.critic.safe_decay_rate
            )

        if self.optimizer.engine == "CasADi":
            cost_function, symbolic_var = nc.func_to_lambda_with_params(
                self.objective, observation, x0=my_action_sqn_init, is_symbolic=True
            )

            if constraint_functions:
                constraints = self.create_constraints(
                    constraint_functions, symbolic_var, observation
                )

            lambda_constr = (
                lambda action: constr_stab_decay_action(action, observation) - self.eps
            )

            constraints += (nc.lambda2symb(lambda_constr, symbolic_var),)

            action_sqn_optimized = self.optimizer.optimize(
                cost_function,
                my_action_sqn_init,
                self.control_bounds,
                constraints=constraints,
                symbolic_var=symbolic_var,
            )

        elif self.optimizer.engine == "SciPy":
            cost_function = nc.func_to_lambda_with_params(
                self.objective, observation, x0=my_action_sqn_init
            )

            if constraint_functions:
                constraints = sp.optimize.NonlinearConstraint(
                    partial(
                        self.create_scipy_constraints,
                        constraint_functions=constraint_functions,
                        observation=observation,
                    ),
                    -np.inf,
                    0,
                )
            my_constraints = sp.optimize.NonlinearConstraint(
                lambda action: constr_stab_decay_action(action, observation),
                -np.inf,
                self.eps,
            )

            # my_constraints = ()

            action_sqn_optimized = self.optimizer.optimize(
                cost_function,
                my_action_sqn_init,
                self.control_bounds,
                constraints=my_constraints,
            )

        return action_sqn_optimized[: self.dim_input]

    def objective(
        self, action_sqn, observation,
    ):
        """
        See class documentation.
        
        Customization
        -------------        
        
        Introduce your mode and the respective actor loss in this method. Don't forget to provide description in the class documentation.

        """

        my_action_sqn = nc.reshape(action_sqn, [self.Nactor + 1, self.dim_input])

        observation_sqn = [observation]

        observation_sqn_predicted = self.state_predictor.predict_state_sqn(
            observation, my_action_sqn
        )

        observation_sqn = nc.vstack(
            (nc.reshape(observation, [1, self.dim_output]), observation_sqn_predicted)
        )

        J = self.stage_obj(observation_sqn[0, :], my_action_sqn[0, :]) + self.critic(
            observation_sqn[1, :], self.critic.weights
        )
        return J
