from rcognita.utilities import rc
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from casadi import vertcat, nlpsol, DM, SX, Function
from abc import ABC, abstractmethod
import time
import torch.optim as optim
import torch
from multiprocessing import Pool


class BaseOptimizer(ABC):
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

    def verbose(opt_func):
        def wrapper(self, *args, **kwargs):
            tic = time.time()
            result = opt_func(self, *args, **kwargs)
            toc = time.time()
            if self.verbose:
                print(f"result optimization time:{toc-tic} \n")

            return result

        return wrapper


class SciPyOptimizer(BaseOptimizer):
    engine = "SciPy"

    def __init__(self, opt_method, opt_options, verbose=True):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @BaseOptimizer.verbose
    def optimize(self, objective, x0, bounds, constraints=(), verbose=True):

        bounds = sp.optimize.Bounds(bounds[0], bounds[1], keep_feasible=True)

        before_opt = objective(x0)
        opt_result = minimize(
            objective,
            x0=x0,
            method=self.opt_method,
            bounds=bounds,
            options=self.opt_options,
            constraints=constraints,
            tol=1e-7,
        )
        if verbose:
            print(f"before:{before_opt},\nafter:{opt_result.fun}")

        return opt_result.x


class CasADiOptimizer(BaseOptimizer):
    engine = "CasADi"

    def __init__(self, opt_method, opt_options, verbose=True):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.verbose = verbose

    @BaseOptimizer.verbose
    def optimize(
        self, objective, x0, bounds, constraints=(), symbolic_var=None,
    ):
        optimization_problem = {
            "f": objective,
            "x": vertcat(symbolic_var),
            "g": vertcat(*constraints),
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

        if ubg is not None and len(ubg) > 0:
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1], ubg=ubg)
        else:
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])

        ##### DEBUG
        # g1 = Function("g1", [symbolic_var], [constraints])

        # print(g1(result["x"]))
        ##### DEBUG

        return result["x"]


class GradientOptimizer(CasADiOptimizer):
    def __init__(
        self, objective, learning_rate, n_steps, grad_norm_ub=1e-2, verbose=True
    ):
        self.objective = objective
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.grad_norm_ub = grad_norm_ub
        self.verbose = verbose

    def substitute_args(self, x0, *args):
        cost_function, symbolic_var = rc.function2MX(
            self.objective, x0=x0, force=True, *args
        )

        return cost_function, symbolic_var

    def grad_step(self, x0, *args):
        cost_function, symbolic_var = self.substitute_args(x0, *args)
        cost_function = Function("f", [symbolic_var], [cost_function])
        gradient = rc.autograd(cost_function, symbolic_var)
        grad_eval = gradient(x0)
        norm_grad = rc.norm_2(grad_eval)
        if norm_grad > self.grad_norm_ub:
            grad_eval = grad_eval / norm_grad * self.grad_norm_ub

        x0_res = x0 - self.learning_rate * grad_eval
        return x0_res

    @BaseOptimizer.verbose
    def optimize(self, x0, *args):
        for _ in range(self.n_steps):
            x0 = self.grad_step(x0, *args)

        return x0


class TorchOptimizer(BaseOptimizer):
    engine = "Torch"

    def __init__(
        self, opt_options, iterations=1, opt_method=torch.optim.SGD, verbose=False
    ):
        self.opt_method = opt_method
        self.opt_options = opt_options
        self.iterations = iterations
        self.verbose = verbose

    @BaseOptimizer.verbose
    def optimize(self, objective, model, model_input):
        optimizer = self.opt_method(model.parameters(), **self.opt_options)
        # optimizer.zero_grad()

        for _ in range(self.iterations):
            optimizer.zero_grad()
            loss = objective(model_input)
            loss.backward()
            optimizer.step()
            if self.verbose:
                print(objective(model_input))


class BruteForceOptimizer(BaseOptimizer):
    engine = "Parallel"

    def __init__(self, n_pools, possible_variants):
        self.n_pools = n_pools
        self.possible_variants = possible_variants

    def element_wise_maximization(self, x):
        reward_func = lambda variant: self.objective(variant, x)
        reward_func = np.vectorize(reward_func)
        values = reward_func(self.possible_variants)
        return self.possible_variants[np.argmax(values)]

    def optimize(self, objective, table):
        self.table = table
        self.objective = objective
        with Pool(self.n_pools) as p:
            result_table = p.map(
                self.element_wise_maximization,
                np.nditer(self.table, flags=["external_loop"]),
            )[0]
        return result_table

