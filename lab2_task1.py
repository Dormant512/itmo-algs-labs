import numpy as np

epsilon = 0.001
gold = (3 - np.sqrt(5)) / 2
inf = float("inf")
task1_domains = [(0, 1), (0, 1), (0.1, 1)]

#power function
def task1_func1(x):
    """y = x^3"""
    return x**3

#absolute function
def task1_func2(x):
    """y = |x - 0.2|"""
    return np.absolute(x - 0.2)

#trigonometric function
def task1_func3(x):
    """y = x * sin(1/x)"""
    if x == 0:
        return None
    else:
        return x * np.sin(1/x)

task1_functions = [task1_func1, task1_func2, task1_func3]

#exhaustive search function
def exhaustive_search(func, domain, eps=epsilon):
    iter_count = 0
    func_count = 0
    current_x = domain[0]
    current_min_x = current_x
    current_min_y = inf
    while current_x <= domain[1]:
        value = func(current_x)
        if value < current_min_y:
            current_min_x = current_x
            current_min_y = value
        current_x += eps
        iter_count += 1
        func_count += 1
    return current_min_x, current_min_y, iter_count, func_count

#dichotomic search function
def dichotomic_search(func, domain, eps=epsilon, iter_count=0, func_count=0):
    iter_count += 1
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

#golden search function
def golden_search(func, domain, eps=epsilon, x1=None, x2=None, iter_count=0, func_count=0):
    iter_count += 1
    if domain[0] + epsilon > domain[1]:
        y_left, y_right = func(domain[0]), func(domain[1])
        func_count += 2
        if y_left < y_right:
            return domain[0], y_left, iter_count, func_count
        else:
            return domain[1], y_right, iter_count, func_count
    else:
        if not x1:
            x1 = domain[0] + gold * (domain[1] - domain[0])
        if not x2:
            x2 = domain[1] - gold * (domain[1] - domain[0])
        y1, y2 = func(x1), func(x2)
        func_count += 2
        if y1 < y2:
            new_domain = (domain[0], x2)
            x2 = x1
            ans = golden_search(func, new_domain, eps=eps, x2=x2, iter_count=iter_count, func_count=func_count)
        else:
            new_domain = (x1, domain[1])
            x1 = x2
            ans = golden_search(func, new_domain, eps=eps, x1=x1, iter_count=iter_count, func_count=func_count)
        return ans

if __name__ == "__main__":
    for i in range(3):
        print("    ", task1_functions[i].__doc__)

        #exhaustive search result
        ans = exhaustive_search(task1_functions[i], task1_domains[i])
        print("Exhaustive search\nx: {:.4f}    y: {:.4f}    iter: {}    func: {}".format(*ans))

        #dichotomic search result
        ans = dichotomic_search(task1_functions[i], task1_domains[i])
        print("Dichotomic search\nx: {:.4f}    y: {:.4f}    iter: {}    func: {}".format(*ans))

        #golden search result
        ans = golden_search(task1_functions[i], task1_domains[i])
        print("Golden search\nx: {:.4f}    y: {:.4f}    iter: {}    func: {}".format(*ans))

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

wait = input("waiting: ")
