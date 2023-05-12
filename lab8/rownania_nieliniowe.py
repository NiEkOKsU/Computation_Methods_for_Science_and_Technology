import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.float_format = "{:,.12f}".format

def save(results):
    df = pd.DataFrame(results)
    print(df)
    path = r'D:\Computation_Methods_for_Science_and_Technology\lab8\res_newton_10-9.xlsx'
    df.to_excel(path)

def func1(x):
    return x**14 + x**13

def der_func1(x):
    return 14*x**13 + 13*x**12

def newtons_method(func, der, x_0, epsilon, max_iter, stop_condition):
    x_n = x_0
    for n in range(max_iter):
        f_xn = func(x_n)
        der_fxn = der(x_n)
        if der_fxn == 0:
            # Zero derivative. No solution found.
            return None, None
        if stop_condition == "abs" and abs(f_xn) < epsilon:
            # Found solution
            return x_n, n
        elif stop_condition == "points" and abs(f_xn / der_fxn) < epsilon:
            # Found solution
            return x_n, n
        x_n -= f_xn / der_fxn
    # Exceeded maximum number of iterations. No solution found.
    return None, np.inf

def secant(func, x_1, x_2, epsilon, max_iter, stop_condition):
    for n in range(max_iter):
        if func(x_1) == func(x_2):
            # Divided by zero
            return None, None
        x_1, x_2 = x_2, x_2 - (x_2 - x_1) * func(x_2) / (func(x_2) - func(x_1))
        if stop_condition == "abs" and abs(func(x_2)) < epsilon:
            # Found solution
            return x_2, n
        elif stop_condition == "points" and abs(x_1 - x_2) < epsilon:
            # Found solution
            return x_2, n
    # Exceeded maximum number of iterations. No solution found.
    return None, np.inf

def create_dataframe(method_name, epsilon, max_iter, stop_condition):
    X = np.arange(-1.4, 0.6 + 0.1, 0.1)
    result = []
    for x_0 in X:
        if method_name == "newton":
            x, n = newtons_method(func1, der_func1, x_0, epsilon, max_iter, stop_condition)
            result += [x, n, x_0]
        elif method_name == "secant":
                x, n = secant(func1, x_0, 0.6, epsilon, max_iter, stop_condition)
                result += [x, n, x_0]
                x, n = secant(func1, -1.4, x_0, epsilon, max_iter, stop_condition)
                result += [x, n, x_0]
    df = pd.DataFrame(data={"x value": result[::3],
                            "num of iterations": result[1::3],
                            "point": result[2::3]})
    return df


save(create_dataframe("secant", 10**(-9), 1000, "points"))
