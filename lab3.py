import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime, least_squares

def line(x, a, b):
    """Linear Function"""
    return a * x + b

def rational(x, a, b):
    """Rational Function"""
    return a / (1 + b * x)

def d_ab(ab):
    """Least squares error function"""
    return np.sum((func(x, ab[0], ab[1]) - y)**2)

epsilon = 0.001
inf = float("inf")

np.random.seed(1) #Fix random parameters to reproduce results on different machines

alpha = np.random.rand(1)
beta = np.random.rand(1)
x_data = np.linspace(0, 1, 101, endpoint=True)
delta = np.random.normal(size=101)
y_data = line(x_data, alpha, beta) + delta



if __name__ == "__main__":
    #Uncomment the line below for linear approximant
    #func = line

    #Uncomment the line below for rational approximant
    func = rational

    x = x_data
    y = y_data
    plt.scatter(x, y, color="royalblue", label="noisy data", marker=".")
    plt.plot(x, func(x, alpha, beta), color="maroon", label="Initial line")
    print("Alpha: {}    Beta: {}".format(alpha, beta))
    
    guess = [0.5, 0.5]

    #Conjugated Gradient Descent (CG)
    cg_res = minimize(d_ab, guess, method="CG", tol=epsilon)
    plt.plot(x, func(x, *cg_res.x), label="Conjugated Gradient Descent")
    print("Conjugated Gradient Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*cg_res.x, cg_res.nit, cg_res.nfev))

    #Newtons CG
    fprime = lambda x_jac: approx_fprime(x_jac, d_ab, epsilon=epsilon) #Callable Jacobian for Newton
    newton_res = minimize(d_ab, guess, method="Newton-CG", jac=fprime, tol=epsilon)
    plt.plot(x, func(x, *newton_res.x), label="Newton's CG Descent")
    print("Newtons CG Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*newton_res.x, newton_res.nit, newton_res.nfev))


    def residuals_len_mar(ab, x=x, y=y):
        """Least squares error for Len-Mar"""
        return func(x, ab[0], ab[1]) - y


    #Levenberg-Marquardt Method
    lev_mar_res = least_squares(residuals_len_mar, guess, method="lm", xtol=epsilon)
    plt.plot(x, func(x, *lev_mar_res.x), label="Levenberg-Marquardt")
    print("Levenberg-Marquardt\nA: {}    B: {}    Iter: {}    Func: {}".format(*lev_mar_res.x, 666, lev_mar_res.nfev))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

