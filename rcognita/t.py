# from casadi import SX, MX, DM, Function, gradient, substitute, norm_1, nlpsol, norm_2

# x = SX.sym("x", 2)
# y = SX([3, 3])

# g = norm_1(x - y) - 0.1

# f = 1  # norm_2(x)
# # f = Function("f", [x], [x ** 2 + 3 * x])

# options = {
#     "print_time": 0,
#     "ipopt.max_iter": 200,
#     "ipopt.print_level": 0,
#     "ipopt.acceptable_tol": 1e-7,
#     "ipopt.acceptable_obj_change_tol": 1e-4,
# }

# optimization_problem = {}
# optimization_problem["x"] = x
# optimization_problem["f"] = f
# optimization_problem["g"] = g

# solver = nlpsol("solver", "ipopt", optimization_problem, options)

# result = solver(x0=DM([10, 20]), ubx=DM([20, 50]), lbx=-DM([10, 20]), ubg=0)["x"]

# g = Function("g", [x], [g])


# print(g(result), result)

# from optimizers import GradientOptimizer
# import numpy as np
# import matplotlib.pyplot as plt

# parabola = lambda x: 0.5 * x ** 2 + 2 * np.sqrt(5) * x + 1

# x = np.linspace(-10, 15, 200)
# y = parabola(x)
# x0 = [10]

# optimizer = GradientOptimizer(parabola, 0.002, 10000)

# x_new = optimizer.optimize(x0)

# plt.figure(figsize=(10, 10))
# plt.plot(x, y, c="g", label="plotted function")
# plt.axvline(x0, c="r")
# plt.axhline(parabola(x0[0]), c="r")
# plt.axvline(np.array(x_new), c="b")
# plt.axhline(parabola(np.array(x_new)), c="b")
# plt.grid()
# plt.legend()
# plt.show()

# import os, sys
# import numpy as np

# PARENT_DIR = os.path.abspath(__file__ + "/../../")
# sys.path.insert(0, PARENT_DIR)
# CUR_DIR = os.path.abspath(__file__ + "/..")
# sys.path.insert(0, CUR_DIR)

# from utilities import nc


# def func(x):
#     return x ** 2


# a = [1, 2, 3]

# p = np.array([2])
# x_sym = nc.array()
# x_num = np.array([1, 2])
# norm = nc.norm_1(x_num)
# print(norm)

from abc import ABC, abstractmethod
import numpy as np
from utilities import nc
from npcasadi_api import typeInferenceDecoratorFunc

Nactor = 10
dim_input = 2
pred_step_size = 0.01
dim_state = 5


def sys_rhs(t, state, action, disturb=[]):

    Dstate = nc.zeros(dim_state, prototype=action)
    Dstate[0] = state[3] * nc.cos(state[2])
    Dstate[1] = state[3] * nc.sin(state[2])
    Dstate[2] = state[4]

    Dstate[3] = 1 / action[0]
    Dstate[4] = 1 / action[1]

    return Dstate


g1 = lambda x: x[0] ** 2 + x[1] ** 2

g2 = lambda x: x[0] ** 2 - x[1] ** 2 + 1

constraints = (g1, g2)

observation = nc.array([5, 5, 0.5, 1, 1], array_type="SX")

u = nc.array_symb((dim_input * Nactor, 1), literal="x")

u = nc.array_symb((dim_input * Nactor, 1), literal="x")


def create_symbolic_constraints(
    my_action_sqn, constraint_functions, observation, is_symbolic=False
):
    current_observation = observation

    constraint_violations_result = [0 for _ in range(Nactor - 1)]
    constraint_violations_buffer = [0 for _ in constraint_functions]

    for constraint_function in constraint_functions:
        constraint_violations_buffer[0] = constraint_function(current_observation)

    max_constraint_violation = nc.max(constraint_violations_buffer)
    start_in_danger = max_constraint_violation > 0.0

    max_constraint_violation = -1
    action_sqn = nc.reshape(my_action_sqn, [Nactor, dim_input])
    predicted_state = current_observation

    for i in range(1, Nactor):

        current_action = action_sqn[i - 1, :]
        predicted_state = predicted_state + pred_step_size * sys_rhs(
            [], predicted_state, current_action
        )

        constraint_violations_buffer = []
        for constraint in constraints:
            constraint_violations_buffer.append(constraint(predicted_state))

        max_constraint_violation = nc.max(constraint_violations_buffer)
        constraint_violations_result[i - 1] = max_constraint_violation

    for i in range(2, Nactor - 1):
        constraint_violations_result[i] = nc.if_else(
            nc.logic_and(constraint_violations_result[i - 1] > 0, start_in_danger),
            constraint_violations_result[i - 1],
            constraint_violations_result[i],
        )

    return constraint_violations_result, my_action_sqn


print(create_symbolic_constraints(u, constraints, observation))
