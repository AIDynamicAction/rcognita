from casadi import SX, MX, DM, Function, gradient, substitute, norm_1, nlpsol, norm_2

x = SX.sym("x", 2)
y = SX([3, 3])

g = norm_1(x - y) - 0.1

f = 1  # norm_2(x)
# f = Function("f", [x], [x ** 2 + 3 * x])

options = {
    "print_time": 0,
    "ipopt.max_iter": 200,
    "ipopt.print_level": 0,
    "ipopt.acceptable_tol": 1e-7,
    "ipopt.acceptable_obj_change_tol": 1e-4,
}

nlp = {}
nlp["x"] = x
nlp["f"] = f
nlp["g"] = g

solver = nlpsol("solver", "ipopt", nlp, options)

result = solver(x0=DM([10, 20]), ubx=DM([20, 50]), lbx=-DM([10, 20]), ubg=0)["x"]

g = Function("g", [x], [g])


print(g(result), result)

