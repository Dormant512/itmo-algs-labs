import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def line(x, a, b):
    """Linear Function"""
    return a * x + b

def rational(x, a, b):
    """Rational Function"""
    return a / (1 + b * x)

def d(func, x, y, a, b):
    return np.sum((func(x, a, b) - y)**2)

epsilon = 0.001
inf = float("inf")
alpha = np.random.rand(1)
beta = np.random.rand(1)
x_data = np.linspace(0, 1, 101, endpoint=True)
delta = np.random.normal(size=101)
y_data = line(x_data, alpha, beta) + delta

def dichotomic_search(func, domain, eps=epsilon, iter_count=0, func_count=0):
    iter_count += 1
    func_count += 1
    if domain[0] + epsilon > domain[1]:
        y_left, y_right = func(domain[0]), func(domain[1])
        func_count += 2
        if y_left < y_right:
            return domain[0], y_left, iter_count, func_count
        else:
            return domain[1], y_right, iter_count, func_count
    else:
        delta = eps / 2
        x1 = (domain[0] + domain[1] - delta) / 2
        x2 = (domain[0] + domain[1] + delta) / 2
        y1, y2 = func(x1), func(x2)
        func_count += 2
        if y1 < y2:
            new_domain = (domain[0], x2)
        else:
            new_domain = (x1, domain[1])
        return dichotomic_search(func, new_domain, eps=eps, iter_count=iter_count, func_count=func_count)

def exhaustive_search(x, y, func, a_domain=(0,1), b_domain=(0,1), eps=epsilon):
    D_min = inf
    iter_count = 0
    func_count = 0
    a_data = np.linspace(*a_domain, int((a_domain[1] - a_domain[0]) / eps), endpoint=True)
    b_data = np.linspace(*b_domain, int((b_domain[1] - b_domain[0]) / eps), endpoint=True)
    for a in a_data:
        for b in b_data:
            iter_count += 1
            D = d(func, x, y, a, b)
            func_count += 1
            if D < D_min:
                D_min = D
                a_opt = a
                b_opt = b
    return a_opt, b_opt, iter_count, func_count

def gauss_search(x, y, func, a_domain=(0,1), b_domain=(0,1), a_init=0.5, b_init=0.5, eps=epsilon):
    a, b = a_init, b_init
    iter_count = 0
    func_count = 0
    while True:
        a_last, b_last = a, b
        def d_a(a):
            return np.sum((func(x, a, b) - y)**2)

        def d_b(b):
            return np.sum((func(x, a, b) - y)**2)
        
        a_all = dichotomic_search(d_a, a_domain, eps=eps)
        b_all = dichotomic_search(d_b, b_domain, eps=eps)
        a, b = a_all[0], b_all[0]
        iter_count += (a_all[2] + b_all[2] + 1)
        func_count += (a_all[3] + b_all[3])
        if np.absolute(a_last - a) <= eps and np.absolute(b_last - b) <= eps:
            break
    return a, b, iter_count, func_count

if __name__ == "__main__":
    #func = line
    func = rational
    x = x_data
    y = y_data
    plt.scatter(x, y, color="royalblue", label="noisy data", marker=".")
    plt.plot(x, func(x, alpha, beta), color="maroon", label="Initial line")
    print("Alpha: {}    Beta: {}".format(alpha, beta))
    
    ex_search = exhaustive_search(x, y, func)
    plt.plot(x, func(x, *ex_search[:-2]), label="Exhaustive")
    print("Exhaustive search\nA: {}    B: {}    Iter: {}    Func: {}".format(*ex_search))
    
    ga_search = gauss_search(x, y, func)
    plt.plot(x, func(x, *ga_search[:-2]), label="Gauss")
    print("Gauss search\nA: {}    B: {}    Iter: {}    Func: {}".format(*ga_search))
    
    def d_ab(ab):
        return np.sum((func(x, ab[0], ab[1]) - y)**2)
    
    guess = [0.5, 0.5]
    nelder_mead_res = minimize(d_ab, guess, method="Nelder-Mead", tol=epsilon)
    plt.plot(x, func(x, *nelder_mead_res.x), label="Nelder-Mead")
    print("Nelder-Mead search\nA: {}    B: {}    Iter: {}    Func: {}".format(*nelder_mead_res.x, nelder_mead_res.nit, nelder_mead_res.nfev))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

