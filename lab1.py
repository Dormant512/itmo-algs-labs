import time
import sys
from functools import wraps
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(2500)
v = np.random.rand(2000)
A = np.random.rand(2000, 2000)
B = np.random.rand(2000, 2000)

# timing decorator
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return (result, total_time)
    return timeit_wrapper

# constant function
@timeit
def const_func(x):
    return 1

# sum function
@timeit
def sum_func(x):
    return np.sum(x)

# prod function
@timeit
def prod_func(x):
    return np.prod(x)

# poly direct function
@timeit
def poly_direct_func(x):
    res = 0
    num = 1.5
    n = len(x)
    for i in range(n):
        term = x[i] * (np.power(num, i))
        res += term
    return res

# horner recursive function
def horner_recursive(x, num):
    if len(x) <= 1:
        return x
    else:
        return x[0] + num * horner_recursive(x[1:], num)

# poly Horner function
@timeit
def poly_horner_func(x):
    num = 1.5
    res = horner_recursive(x, num)
    return res

# bubble sort
@timeit
def bubble_sort_func(arr):
    is_changed = True
    while(is_changed):
        is_changed = False
        for i in range(len(arr) - 1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                is_changed = True
    return arr

# quick sort
@timeit
def quick_sort_func(arr):
    res = np.sort(arr, kind="quick sort")
    return res

# timsort, the default sorting algorithm used in python's built in sort() method
@timeit
def timsort_func(arr):
    arr.sort()
    return arr

@timeit
def mat_mul_func(A, B):
    return np.matmul(A, B)

# data generator
def generate_data(func, *args):
    raw_times = [0] * 2000
    if len(args) == 1:
        v = args[0]
        for n in range(1, 2001):
            five_runs = []
            argument = v[:n]
            for i in range(5):
                res = func(argument)[1]
                five_runs.append(res)
            raw_times[n-1] = np.mean(five_runs)
        return raw_times
    elif len(args) == 2:
        A, B = args[0], args[1]
        for n in range(1, 2001):
            five_runs = []
            arg1, arg2 = A[:n, :n], B[:n, :n]
            for i in range(5):
                res = func(arg1, arg2)[1]
                five_runs.append(res)
            raw_times[n-1] = np.mean(five_runs)
        return raw_times

def o_const(x, a):
    return 0 * x + a

def o_linear(x, a, b):
    return a * x + b

def o_nlogn(x, a, b, c):
    return a * x * np.log(b * x) + c

def o_square(x, a, b, c):
    return a * x**2 + b * x + c

def o_cube(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

if __name__ == "__main__":
    # change function to appropriate theoretical; change to:
        #o_const for constant approximation
        #o_linear for linear approximation
        #o_nlogn for logarithmic approximation
        #o_square for quadratic approximation
        #o_cube for cubic approximation
    func = o_const

    # change data source to appropriate; change the first argument to
        #const_func for constant function
        #sum_func for the sum of elements
        #prod_func for the product of elements
        #poly_direct_func for direct polynomial calculation
        #poly_horner_func for Horner’s method
        #bubble_sort_func for bubble sort
        #quick_sort_func for quick sort
        #timsort_func for timsort
        #mat_mul_func for matrix product
    ydata = generate_data(const_func, v)
    
    xdata = np.linspace(1, 2000, 2000, endpoint=True)
    plt.plot(xdata, ydata, color="royalblue", label="empyrical")
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata, func(xdata, *popt), color="maroon", label="theoretical")
    plt.xlabel("n, size of dataset")
    plt.ylabel("average execution time, seconds")
    plt.legend()
    plt.show()
