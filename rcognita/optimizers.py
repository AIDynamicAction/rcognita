from numpy.random import rand
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from rcognita.utilities import rep_mat


class RcognitaOptimizer:
    def __init__(self, opt_method, optimizer, bounds, options, tol=1e-7):
        self.opt_method = opt_method
        self.optimizer = optimizer
        self.bounds = bounds
        self.options = options
        self.tol = tol

    def optimize(self, cost, x0, constraints=()):
        method = self.opt_method
        options = self.options
        bounds = self.bounds
        tol = self.tol
        result = self.optimizer(
            cost=cost,
            x0=x0,
            bounds=bounds,
            options=options,
            tol=tol,
            opt_method=method,
            constraints=constraints,
        )
        return result

    def from_scipy_transformer(optimizer):
        def opt_wrapper(cost, x0, opt_method, options, bounds, tol, constraints=()):
            result = optimizer(
                cost,
                x0=x0,
                opt_method=opt_method,
                options=options,
                bounds=bounds,
                tol=tol,
                constraints=constraints,
            ).x
            print(result)
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
        def minimizer(actor_cost, x0, opt_method, options, bounds, tol, constraints=()):
            return minimize(
                lambda w_actor: actor_cost(w_actor),
                x0=x0,
                method=opt_method,
                tol=tol,
                bounds=bounds,
                options=options,
                constraints=constraints,
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
            critic_cost, x0, opt_method, options, bounds, tol, constraints=()
        ):
            result = minimize(
                lambda w_critic: critic_cost(w_critic),
                x0=x0,
                method=opt_method,
                tol=tol,
                bounds=bounds,
                options=options,
                constraints=constraints,
            )

            return result

        return RcognitaOptimizer(critic_opt_method, minimizer, bnds, critic_opt_options)
