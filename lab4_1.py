import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, differential_evolution,dual_annealing



def rational(x, a, b): #новая функция на это место
    """Rational Function"""
    return a / (1 + b * x)


def func_check(x):
    '''function to calculate y'''
    return 1/(x*x - 3*x +2)


def d_ab(ab):
    """Least squares error function"""
    return np.sum((func(x, ab[0], ab[1]) - y)**2)


def rational_grad(ab):
    """Manually calculated gradient of function d_ab when func == rational"""
    first = np.sum(- (2 * (-ab[0] + ab[1] * x * y + y)) / ((ab[1] * x + 1)**2))
    second = np.sum(- (2 * ab[0] * x * (ab[0] - y * (ab[1] * x + 1))) / ((ab[1] * x + 1)**3))
    return np.array([first, second])


epsilon = 0.001
inf = float("inf")

np.random.seed(0) #Fix random parameters to reproduce results on different machines

alpha = np.random.rand(1)
beta = np.random.rand(1)
x_data = np.linspace(0, 1, 1001, endpoint=True)
delta = np.random.normal(size=1001)
y_data = np.linspace(0, 1, 1001, endpoint=True)
for i in range(0,1001):
    if func_check(x_data[i])<(-100):
        y_data[i] = -100 + delta[i]
    elif (func_check(x_data[i])>(-100) and func_check(x_data[i])<100):
        y_data[i] = func_check(x_data[i]) + delta[i]
    else:
        y_data[i] = -100 + delta[i]
guess = [0.5, 0.5]

if __name__ == "__main__":

    #Uncomment the line below for rational approximant
    func, grad = rational, rational_grad

    x = x_data
    y = y_data
    plt.scatter(x, y, color="royalblue", label="noisy data", marker=".")
    plt.plot(x, func(x, alpha, beta), color="maroon", label="Initial line")
    print("Alpha: {}    Beta: {}".format(alpha, beta))


    #Levenberg-Marquardt Method
    def residuals_len_mar(ab, x=x, y=y):
        """Least squares error for Len-Mar"""
        return func(x, ab[0], ab[1]) - y

    lev_mar_res = least_squares(residuals_len_mar, guess, method="lm", xtol=epsilon)
    plt.plot(x, func(x, *lev_mar_res.x), label="Levenberg-Marquardt")
    print("Levenberg-Marquardt\nA: {}    B: {}    Iter: {}    Func: {}".format(*lev_mar_res.x, lev_mar_res.nfev, lev_mar_res.nfev))

    #Nelder-Mead method
    nelder_mead_res = minimize(d_ab, guess, method="Nelder-Mead", tol=epsilon)
    plt.plot(x, func(x, *nelder_mead_res.x), label="Nelder-Mead")
    print("Nelder-Mead search\nA: {}    B: {}    Iter: {}    Func: {}".format(*nelder_mead_res.x, nelder_mead_res.nit, nelder_mead_res.nfev))

    #Differential Evolution
    r_min, r_max = -5.0, 5.0
    bounds = [[r_min, r_max], [r_min, r_max]]
    differential_evolution_res = differential_evolution(d_ab, bounds, strategy='best1bin', tol=epsilon)
    plt.plot(x, func(x, *differential_evolution_res.x), label="Differential Evolution")
    print("Differential Evolution\nA: {}    B: {}    Iter: {}    Func: {}".format(*differential_evolution_res.x, differential_evolution_res.nit, differential_evolution_res.nfev))
    
    #Simulated Annealing
    annealing_res = dual_annealing(d_ab, bounds,initial_temp=5230.0, restart_temp_ratio=2e-04, maxfun=1000.0)
    plt.plot(x, func(x, *annealing_res.x), label="Simulated Annealing",color="gold")
    print("Simulated Annealing\nA: {}    B: {}    Iter: {}    Func: {}".format(*annealing_res.x, annealing_res.nit, annealing_res.nfev))
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()  