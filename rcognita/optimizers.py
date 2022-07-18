from .utilities import dss_sim
from rcognita.utilities import rep_mat
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from casadi import SX, cos, sin, fmin, vertcat, nlpsol
from numpy.random import randn
import time


class RcognitaOptimizer:
    def __init__(self, opt_method, optimizer, bounds, options, is_symbolic=False):
        self.is_symbolic = is_symbolic
        self.opt_method = opt_method
        self.optimizer = optimizer
        self.bounds = bounds
        self.options = options

    def optimize(self, cost, x0, constraints=(), symbolic_var=None):
        return self.optimizer(
            cost=cost,
            x0=x0,
            opt_method=self.opt_method,
            options=self.options,
            bounds=self.bounds,
            constraints=constraints,
            symbolic_var=symbolic_var,
        )

    def from_scipy_transformer(optimizer):
        def opt_wrapper(
            cost, x0, opt_method, options, bounds, constraints=(), symbolic_var=None
        ):
            result = optimizer(
                cost,
                x0=x0,
                opt_method=opt_method,
                options=options,
                bounds=bounds,
                constraints=constraints,
            ).x
            return result

        return opt_wrapper

    @staticmethod
    def standard_actor_optimizer(actor_opt_method, **kwargs):
        ctrl_bnds = kwargs.get("ctrl_bnds")
        Nactor = kwargs.get("Nactor")
        action_min = np.array(ctrl_bnds[:, 0])
        action_max = np.array(ctrl_bnds[:, 1])
        action_sqn_min = rep_mat(action_min, 1, Nactor)
        action_sqn_max = rep_mat(action_max, 1, Nactor)

        if actor_opt_method == "trust-constr":
            actor_opt_options = {
                "maxiter": 300,
                "disp": False,
            }
        else:
            actor_opt_options = {
                "maxiter": 300,
                "maxfev": 5000,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-7,
                "fatol": 1e-7,
            }

        bnds = sp.optimize.Bounds(action_sqn_min, action_sqn_max, keep_feasible=True)

        @RcognitaOptimizer.from_scipy_transformer
        def minimizer(actor_cost, x0, opt_method, options, bounds, constraints=()):
            return minimize(
                lambda w_actor: actor_cost(w_actor),
                x0=x0,
                method=opt_method,
                bounds=bounds,
                options=options,
                constraints=constraints,
                tol=1e-7,
            )

        return RcognitaOptimizer(actor_opt_method, minimizer, bnds, actor_opt_options)

    @staticmethod
    def standard_critic_optimizer(critic_opt_method, **kwargs):

        critic_struct = kwargs.get("critic_struct")
        dim_input = kwargs.get("dim_input")
        dim_output = kwargs.get("dim_output")

        if critic_struct == "quad-lin":
            dim_critic = int(
                ((dim_output + dim_input) + 1) * (dim_output + dim_input) / 2
                + (dim_output + dim_input)
            )
            Wmin = -1e3 * np.ones(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)

        elif critic_struct == "quadratic":
            dim_critic = int(
                ((dim_output + dim_input) + 1) * (dim_output + dim_input) / 2
            )
            Wmin = np.zeros(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quad-nomix":
            dim_critic = dim_output + dim_input
            Wmin = np.zeros(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quad-mix":
            dim_critic = int(dim_output + dim_output * dim_input + dim_input)
            Wmin = -1e3 * np.ones(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        if critic_opt_method == "trust-constr":
            critic_opt_options = {
                "maxiter": 200,
                "disp": False,
            }
        else:
            critic_opt_options = {
                "maxiter": 200,
                "maxfev": 1500,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-7,
                "fatol": 1e-7,
            }

        bnds = sp.optimize.Bounds(Wmin, Wmax, keep_feasible=True)

        @RcognitaOptimizer.from_scipy_transformer
        def minimizer(
            critic_cost,
            x0,
            opt_method,
            options,
            bounds,
            constraints=(),
            symbolic_var=None,
        ):
            result = minimize(
                lambda w_critic: critic_cost(w_critic),
                x0=x0,
                method=opt_method,
                bounds=bounds,
                options=options,
                constraints=constraints,
                tol=1e-7,
            )

            return result

        return RcognitaOptimizer(critic_opt_method, minimizer, bnds, critic_opt_options)

    @staticmethod
    def casadi_actor_optimizer(opt_method="ipopt", max_iter=120, **kwargs):
        ctrl_bnds = kwargs.get("ctrl_bnds")
        Nactor = kwargs.get("Nactor")
        action_min = np.array(ctrl_bnds[:, 0])
        action_max = np.array(ctrl_bnds[:, 1])
        action_sqn_min = rep_mat(action_min, 1, Nactor)
        action_sqn_max = rep_mat(action_max, 1, Nactor)
        bounds = [action_sqn_min, action_sqn_max]

        options = {
            "print_time": 0,
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-4,
        }

        def optimizer(
            cost,
            x0,
            opt_method="ipopt",
            options=options,
            bounds=bounds,
            constraints=(),
            symbolic_var=None,
            verbose=True,
        ):

            qp_prob = {
                "f": cost,
                "x": vertcat(symbolic_var),
                "g": vertcat(*constraints),
            }

            solver = nlpsol("solver", opt_method, qp_prob, options)
            start = time.time()
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])
            final_time = time.time() - start
            return result["x"]

        return RcognitaOptimizer(
            opt_method, optimizer, bounds, options, is_symbolic=True
        )

    @staticmethod
    def casadi_critic_optimizer(opt_method="ipopt", max_iter=120, **kwargs):
        critic_struct = kwargs.get("critic_struct")
        dim_input = kwargs.get("dim_input")
        dim_output = kwargs.get("dim_output")

        if critic_struct == "quad-lin":
            dim_critic = int(
                ((dim_output + dim_input) + 1) * (dim_output + dim_input) / 2
                + (dim_output + dim_input)
            )
            Wmin = -1e3 * np.ones(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)

        elif critic_struct == "quadratic":
            dim_critic = int(
                ((dim_output + dim_input) + 1) * (dim_output + dim_input) / 2
            )
            Wmin = np.zeros(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quad-nomix":
            dim_critic = dim_output + dim_input
            Wmin = np.zeros(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quad-mix":
            dim_critic = int(dim_output + dim_output * dim_input + dim_input)
            Wmin = -1e3 * np.ones(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)

        options = {
            "print_time": 0,
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-4,
        }

        bounds = [Wmin, Wmax]

        def optimizer(
            cost,
            x0,
            opt_method="ipopt",
            options=options,
            bounds=bounds,
            constraints=(),
            symbolic_var=None,
            verbose=True,
        ):

            qp_prob = {
                "f": cost,
                "x": vertcat(symbolic_var),
                "g": vertcat(*constraints),
            }

            solver = nlpsol("solver", opt_method, qp_prob, options)
            start = time.time()
            result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])
            final_time = time.time() - start
            return result["x"]

        return RcognitaOptimizer(
            opt_method, optimizer, bounds, options, is_symbolic=True
        )

