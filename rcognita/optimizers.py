from rcognita.utilities import rep_mat, nc
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from casadi import vertcat, nlpsol, DM, SX, Function
from abc import ABC, abstractmethod
import time


class BaseOptimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def engine(self):
        return "engine_name"

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass


class SciPyOptimizer(BaseOptimizer):
    engine = "SciPy"

    def __init__(self, opt_method, opt_options):
        self.opt_method = opt_method
        self.opt_options = opt_options

    def optimize(self, objective, x0, bounds, constraints=()):

        bounds = sp.optimize.Bounds(bounds[0], bounds[1], keep_feasible=True)

        opt_result = minimize(
            objective,
            x0=x0,
            method=self.opt_method,
            bounds=bounds,
            options=self.opt_options,
            constraints=constraints,
            tol=1e-7,
        )

        return opt_result.x


class CasADiOptimizer(BaseOptimizer):
    engine = "CasADi"

    def __init__(self, opt_method, opt_options):
        self.opt_method = opt_method
        self.opt_options = opt_options

    def optimize(
        self, objective, x0, bounds, constraints=(), symbolic_var=None, verbose=True,
    ):
        optimization_problem = {
            "f": objective,
            "x": vertcat(symbolic_var),
            "g": vertcat(constraints),
        }

        if isinstance(constraints, tuple):
            ubg = [0 for _ in constraints]
        elif isinstance(constraints, (SX, DM, int, float)):
            ubg = [0]

        try:
            solver = nlpsol(
                "solver", self.opt_method, optimization_problem, self.opt_options,
            )
        except Exception as e:
            print(e)
            return x0

        start = time.time()
        if not ubg is None:
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1], ubg=ubg)
        else:
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])
        final_time = time.time() - start

        result_time = final_time - start

        if verbose:
            print(result_time)

        ##### DEBUG
        # g1 = Function("g1", [symbolic_var], [constraints])

        # print(g1(result["x"]))
        ##### DEBUG

        return result["x"]


class GradientOptimizer(BaseOptimizer):
    def __init__(self, objective, learning_rate, n_steps, grad_norm_ub=1e-2):
        self.engine = "CasADi"
        self.objective = objective
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.grad_norm_ub = grad_norm_ub

    def substitute_args(self, x0, *args):
        cost_function, symbolic_var = nc.function2SX(
            self.objective, x0=x0, force=True, *args
        )

        return cost_function, symbolic_var

    def grad_step(self, x0, *args):
        cost_function, symbolic_var = self.substitute_args(x0, *args)
        cost_function = Function("f", [symbolic_var], [cost_function])
        gradient = nc.autograd(cost_function, symbolic_var)
        grad_eval = gradient(x0)
        norm_grad = nc.norm_2(grad_eval)
        if norm_grad > self.grad_norm_ub:
            grad_eval = grad_eval / norm_grad * self.grad_norm_ub

        x0_res = x0 - self.learning_rate * grad_eval
        return x0_res

    def optimize(self, x0, *args):
        for _ in range(self.n_steps):
            x0 = self.grad_step(x0, *args)

        return x0

