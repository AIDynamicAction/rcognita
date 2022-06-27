from numpy.random import rand
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import NonlinearConstraint


class RcognitaOptimizer:
    def __init__(self, opt_method, optimizer, bounds, options, tol=1e-7):
        self.opt_method = opt_method
        self.optimizer = optimizer
        self.bounds = bounds
        self.options = options
        self.tol = tol

    def optimize(self, cost, x0):
        method = self.opt_method
        options = self.options
        bounds = self.bounds
        tol = self.tol
        # print(method, options, bounds, tol, x0)
        result = self.optimizer(
            cost=cost, x0=x0, bounds=bounds, options=options, tol=tol, opt_method=method
        )
        return result

    def from_scipy_transformer(optimizer):
        def opt_wrapper(cost, x0, opt_method, options, bounds, tol):
            result = optimizer(
                cost,
                x0=x0,
                opt_method=opt_method,
                options=options,
                bounds=bounds,
                tol=tol,
            ).x
            return result

        return opt_wrapper

    @staticmethod
    def standard_actor_optimizer(actor_opt_method, action_sqn_min, action_sqn_max):

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
        def minimizer(actor_cost, x0, opt_method, options, bounds, tol):
            # print(locals())
            # raise ValueError
            return minimize(
                lambda w_actor: actor_cost(w_actor),
                x0=x0,
                method=opt_method,
                tol=tol,
                bounds=bounds,
                options=options,
            )

        return RcognitaOptimizer(actor_opt_method, minimizer, bnds, actor_opt_options)

    @staticmethod
    def standard_critic_optimizer(critic_opt_method, Wmin, Wmax):
        if critic_opt_method == "trust-constr":
            critic_opt_options = {
                "maxiter": 200,
                "disp": False,
            }  #'disp': True, 'verbose': 2}
        else:
            critic_opt_options = {
                "maxiter": 200,
                "maxfev": 1500,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-7,
                "fatol": 1e-7,
            }  # 'disp': True, 'verbose': 2}

        bnds = sp.optimize.Bounds(Wmin, Wmax, keep_feasible=True)

        @RcognitaOptimizer.from_scipy_transformer
        def minimizer(critic_cost, x0, opt_method, options, bounds, tol):
            result = minimize(
                lambda w_critic: critic_cost(w_critic),
                x0=x0,
                method=opt_method,
                tol=tol,
                bounds=bounds,
                options=options,
            )

            return result

        return RcognitaOptimizer(critic_opt_method, minimizer, bnds, critic_opt_options)
