import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime, least_squares

def line(x, a, b,c,d):
    """Linear Function"""
    return (a*x + b)/(x*x + c*x + d)

def func_check(x):
    return 1/(x*x - 3*x +2)

def d_a(func, x, y, a, b,c,d):
    """Least squares error function"""
    return np.sum((func(x,a, b, c, d) - y)**2)

epsilon = 0.001
inf = float("inf")

np.random.seed(1) #Fix random parameters to reproduce results on different machines

a = np.random.rand(1)
b = np.random.rand(1)
c = np.random.rand(1)
d = np.random.rand(1)
x_data = np.linspace(0, 1, 1001, endpoint=True)
y_data = np.linspace(0, 1, 1001, endpoint=True)
delta = np.random.normal(size=1001)

for i in range(0,1001):
    if func_check(x_data[i])<(-100):
        y_data[i] = -100 + delta[i]
    elif (func_check(x_data[i])>(-100) and func_check(x_data[i])<100):
        y_data[i] = func_check(x_data[i]) + delta[i]
    else:
        y_data[i] = -100 + delta[i]


if __name__ == "__main__":
    func = line

    x = x_data
    y = y_data
    plt.scatter(x, y, color="royalblue", label="noisy data", marker=".")
    plt.plot(x, func(x_data, a, b, c, d), color="maroon", label="Initial line")
    
    guess = [0.5, 0.5]

    #Conjugated Gradient Descent (CG)
    cg_res = minimize(d_a, guess, method="CG", tol=epsilon)
    plt.plot(x, func(x, *cg_res.x), label="Conjugated Gradient Descent")
    print("Conjugated Gradient Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*cg_res.x, cg_res.nit, cg_res.nfev))

    #Newtons CG
    fprime = lambda x_jac: approx_fprime(x_jac, d_a, epsilon=epsilon) #Callable Jacobian for Newton
    newton_res = minimize(d_a, guess, method="Newton-CG", jac=fprime, tol=epsilon)
    plt.plot(x, func(x, *newton_res.x), label="Newton's CG Descent")
    print("Newtons CG Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*newton_res.x, newton_res.nit, newton_res.nfev))


    def residuals_len_mar(a,b,c,d, x=x, y=y):
        """Least squares error for Len-Mar"""
        return func(x, a, b, c, d) - y


    #Levenberg-Marquardt Method
    lev_mar_res = least_squares(residuals_len_mar, guess, method="lm", xtol=epsilon)
    plt.plot(x, func(x, *lev_mar_res.x), label="Levenberg-Marquardt")
    print("Levenberg-Marquardt\nA: {}    B: {}    Iter: {}    Func: {}".format(*lev_mar_res.x, 666, lev_mar_res.nfev))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()