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

import os, sys
import numpy as np

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from utilities import nc


def func(x):
    return x ** 2


a = [1, 2, 3]

p = np.array([2])
x_sym = nc.array()
x_num = np.array([1, 2])
norm = nc.norm_1(x_num)
print(norm)
