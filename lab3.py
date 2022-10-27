import numpy as np
import random
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


def line_grad(ab):
    """Manually calculated gradient of function d_ab when func == line"""
    first = np.sum(2 * x * (ab[0] * x + ab[1] - y))
    second = np.sum(2 * (ab[0] * x + ab[1] - y))
    return np.array([first, second])


def rational_grad(ab):
    """Manually calculated gradient of function d_ab when func == rational"""
    first = np.sum(- (2 * (-ab[0] + ab[1] * x * y + y)) / ((ab[1] * x + 1)**2))
    second = np.sum(- (2 * ab[0] * x * (ab[0] - y * (ab[1] * x + 1))) / ((ab[1] * x + 1)**3))
    return np.array([first, second])


epsilon = 0.001
inf = float("inf")

alpha = np.random.rand(1)
beta = np.random.rand(1)

x_data = np.linspace(0, 1, 101, endpoint=True)

delta = np.random.normal(size=101)
guess = [0.5, 0.5]


def bar_bor(gradient, last, cur):
    """Barzilai-Borwein beta coefficient"""
    grad_dif = gradient(cur) - gradient(last)
    top = np.abs((cur-last).dot(grad_dif))
    bottom = np.linalg.norm(grad_dif)**2
    return top/bottom


def gradient_descent(gradient, start=guess, learn_rate=0.001, max_iter=10000, tolerance=epsilon):
    """Gradient descent optimization"""
    minimum = np.array(start)
    beta = learn_rate
    for i in range(max_iter):
        diff = -beta * gradient(minimum)
        if np.all(np.abs(diff) <= tolerance):
            break
        last_minimum = np.copy(minimum)
        minimum += diff
        beta = bar_bor(gradient, last_minimum, minimum)
    return minimum, i, 2*i


if __name__ == "__main__":
    #Uncomment the line below for linear approximant
    #func, grad = line, line_grad

    #Uncomment the line below for rational approximant
    func, grad = rational, rational_grad

    y_data = func(x_data, alpha, beta) + delta

    x = x_data
    y = y_data
    plt.plot(x, y, color="royalblue", label="noisy data")
    plt.plot(x, func(x, alpha, beta), color="maroon", label="Initial line")
    #print(func.__doc__)
    print("Alpha: {}    Beta: {}".format(alpha, beta))
    
    #Gradient Descent
    grad_res, grad_iter, grad_nfev = gradient_descent(grad)
    plt.plot(x, func(x, *grad_res), label="Gradient Descent")
    print("Gradient Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*grad_res, grad_iter, grad_nfev))

    #Conjugated Gradient Descent (CG)
    cg_res = minimize(d_ab, guess, method="CG", tol=epsilon)
    plt.plot(x, func(x, *cg_res.x), label="Conjugated Gradient Descent")
    print("Conjugated Gradient Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*cg_res.x, cg_res.nit, cg_res.nfev))

    #Quasi-Newtons CG
    fprime = lambda x_jac: approx_fprime(x_jac, d_ab, epsilon=epsilon) #Callable Jacobian for Newton
    newton_res = minimize(d_ab, guess, method="Newton-CG", jac=fprime, tol=epsilon)
    plt.plot(x, func(x, *newton_res.x), label="Quasi-Newton's CG Descent")
    print("Quasi-Newtons CG Descent\nA: {}    B: {}    Iter: {}    Func: {}".format(*newton_res.x, newton_res.nit, newton_res.nfev))


    def residuals_len_mar(ab, x=x, y=y):
        """Least squares error for Len-Mar"""
        return func(x, ab[0], ab[1]) - y


    #Levenberg-Marquardt Method
    lev_mar_res = least_squares(residuals_len_mar, guess, method="lm", xtol=epsilon)
    plt.plot(x, func(x, *lev_mar_res.x), label="Levenberg-Marquardt")
    print("Levenberg-Marquardt\nA: {}    B: {}    Iter: {}    Func: {}".format(*lev_mar_res.x, lev_mar_res.nfev, lev_mar_res.nfev))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

