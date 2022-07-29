from rcognita.utilities import rep_mat, nc
import scipy as sp
from scipy.optimize import minimize
import numpy as np
from casadi import vertcat, optimization_problemsol, DM, SX, Function
import time

MAX_ITER = 250


class RcognitaOptimizer:
    def __init__(self, opt_method, optimizer, bounds, options):
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
    def create_scipy_actor_optimizer(actor_opt_method, **kwargs):
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
    def create_scipy_critic_optimizer(critic_opt_method, **kwargs):

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
                lambda weights: critic_cost(weights),
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
    def casadi_actor_optimizer(opt_method="ipopt", max_iter=MAX_ITER, **kwargs):
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
            "ipopt.constr_viol_tol": 1e-7,
            "ipopt.theta_max_fact": 1e-3,
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
            optimization_problem = {
                "f": cost,
                "x": vertcat(symbolic_var),
                "g": vertcat(constraints),
            }

            if isinstance(constraints, tuple):
                ubg = nc.zeros(len(constraints))
            elif isinstance(constraints, SX):
                ubg = nc.zeros(1)

            if nc.shape(constraints)[0] > 0:
                ubg = nc.zeros(nc.shape(constraints))
            try:
                solver = optimization_problemsol(
                    "solver", opt_method, optimization_problem, options
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
            ##### DEBUG
            # g1 = Function("g1", [symbolic_var], [constraints])

            # print(g1(result["x"]))
            ##### DEBUG

            return result["x"]

        return RcognitaOptimizer(opt_method, optimizer, bounds, options)

    @staticmethod
    def casadi_q_critic_optimizer(opt_method="ipopt", max_iter=MAX_ITER, **kwargs):
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
            if constraints is None:
                constraints = ()
            optimization_problem = {
                "f": cost,
                "x": vertcat(symbolic_var),
                "g": vertcat(constraints),
            }

            ubg = None
            if nc.shape(constraints)[0] > 0:
                ubg = nc.zeros(constraints)
            try:
                solver = optimization_problemsol(
                    "solver", opt_method, optimization_problem, options
                )
            except:
                return x0
            start = time.time()
            if not ubg is None:
                result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1], ubg=ubg)
            else:
                result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])
            final_time = time.time() - start
            return result["x"]

        return RcognitaOptimizer(opt_method, optimizer, bounds, options)

    def casadi_v_critic_optimizer(opt_method="ipopt", max_iter=MAX_ITER, **kwargs):
        critic_struct = kwargs.get("critic_struct")
        dim_input = kwargs.get("dim_input")
        dim_output = kwargs.get("dim_output")

        if critic_struct == "quad-lin":
            dim_critic = int((dim_output + 1) * dim_output / 2 + dim_output)
            Wmin = -1e3 * np.ones(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quadratic":
            dim_critic = int((dim_output + 1) * dim_output / 2)
            Wmin = np.zeros(dim_critic)
            Wmax = 1e3 * np.ones(dim_critic)
        elif critic_struct == "quad-nomix":
            dim_critic = dim_output
            Wmin = np.ones(dim_critic) * 0.1
            Wmax = 1e3 * np.ones(dim_critic)

        options = {
            "print_time": 0,
            "ipopt.max_iter": max_iter,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-4,
            "ipopt.constr_viol_tol": 1e-6,
            "ipopt.theta_max_fact": 1e-3,
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
            if constraints is None:
                constraints = ()

            optimization_problem = {
                "f": cost,
                "x": vertcat(symbolic_var),
                "g": vertcat(constraints),
            }

            ubg = None
            if nc.shape(constraints)[0] > 0:
                ubg = nc.zeros(nc.shape(constraints))
            try:
                solver = optimization_problemsol(
                    "solver", opt_method, optimization_problem, options
                )
            except:
                return x0
            start = time.time()
            if not ubg is None:
                result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1], ubg=ubg)
            else:
                result = solver(x0=x0, lbx=bounds[0], ubx=bounds[1])
            final_time = time.time() - start
            return result["x"]

        return RcognitaOptimizer(opt_method, optimizer, bounds, options)


class GradientOptimizer:
    def __init__(self, objective, learning_rate, n_steps, len_ub=1e-2):
        self.objective = objective
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.len_ub = len_ub

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
        if norm_grad > 1:
            grad_eval = grad_eval / norm_grad * self.len_up

        x0_res = x0 - self.learning_rate * grad_eval
        return x0_res

    def optimize(self, x0, *args):
        for _ in range(self.n_steps):
            x0 = self.grad_step(x0, *args)

        return x0

