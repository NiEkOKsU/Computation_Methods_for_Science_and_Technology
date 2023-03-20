import numpy as np
import math
import matplotlib.pyplot as plt

def fun(x):
    return math.e**(4*math.cos(2*x))

def coef(nodes, values):
    n = len(nodes)
    a = [values[i] for i in range(n)]

    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            a[i] = float(a[i]-a[i-1])/float(nodes[i] - nodes[i - j])

    return np.array(a)

def newton(x, nodes):
    a = coef(nodes, list(map(fun, nodes)))
    n = len(a) - 1
    temp = a[n]
    for i in range(n - 1, -1, -1):
        temp = temp * (x - nodes[i]) + a[i]
    return temp

def equadistant(x0, x1, n):
    result = [x0 + i*(x1-x0)/(n-1) for i in range(n)]
    return result

def chebyshew(x0, x1, n):
    result = [1/2*(x0 + x1) + 1/2*(x1 - x0)*math.cos((2*i - 1)*math.pi/(2*n)) for i in range(1, n+1)]
    return result

def max_diff(fval, intrval, n):
    max_df = 0
    for i in range(n):
        max_df = max(abs(fval[i] - intrval[i]), max_df)
    return max_df

def mean_square_error(fval, intrval, n):
    error = 0
    for i in range(n):
        error += (fval[i] - intrval[i]) ** 2
    error = math.sqrt(error) / n
    return error

def main():
    amount = 50000
    low_bound_x = -math.pi
    up_bound_x = 3 * math.pi
    points = list(map(lambda x: (low_bound_x + x*(up_bound_x-low_bound_x)/amount), range(amount)))
    fun_values = list(map(fun, points))
    nodes_num = [17,20,50]
    for n in nodes_num:
        nodes = chebyshew(low_bound_x, up_bound_x, n)
        interpolated = [newton(points[i], nodes) for i in range(amount)]
        print("Największa różnica: ", max_diff(fun_values, interpolated, amount))
        print("Błąd średniokwadratowy: ", mean_square_error(fun_values, interpolated, amount))
        plt.xlabel('oś x')
        plt.ylabel('oś y')
        plt.title('Interpolacja')
        plt.plot(points, fun_values, 'b-', label='f')
        plt.plot(points, interpolated, 'g-', label='wielomian interpolujący')
        plt.plot(nodes, list(map(fun, nodes)), 'r.', label='punkty')
        plt.legend()
        plt.show()

main()