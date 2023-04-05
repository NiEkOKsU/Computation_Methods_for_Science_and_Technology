import numpy as np
import math
import matplotlib.pyplot as plt

def fun(x):
    return math.e**(4*math.cos(2*x))

def dfun(x, step):
    x1, x2 = x - step, x + step
    if x == -math.pi:
        return (fun(x2) - fun(x)) / step
    if x == 3*math.pi:
        return (fun(x) - fun(x1)) / step
    return (fun(x2) - fun(x1)) / 2*step

def hermie(x, nodes):
    n = len(nodes)
    n2 = 2*n
    z = [nodes[i//2] for i in range(n2)]
    result = 0
    helper = 1
    matrix = np.zeros((n2, n2))
    step = nodes[1] - nodes[0]
    for i in range(n2):
        for j in range(i+1):
            if j == 0:
                matrix[i][j] = fun(z[i])
            elif j == 1 & i % 2 == 1:
                matrix[i][j] = dfun(z[i], step)
            else:
                matrix[i][j] = (matrix[i][j-1] - matrix[i-1][j-1]) / (z[i] - z[i-j])
        result += matrix[i][i] * helper
        helper *= (x - z[i])
    return result

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
    amount = 5000
    low_bound_x = -math.pi
    up_bound_x = 3 * math.pi
    points = list(map(lambda x: (low_bound_x + x*(up_bound_x-low_bound_x)/amount), range(amount)))
    fun_values = list(map(fun, points))
    nodes_num = [15,17,20]
    for n in nodes_num:
        nodes = equadistant(low_bound_x, up_bound_x, n)
        interpolated = [hermie(points[i], nodes) for i in range(amount)]
        print("Największa różnica dla ", n ," węzłów : ", max_diff(fun_values, interpolated, amount))
        print("Błąd średniokwadratowy dla ", n ," węzłów : ", mean_square_error(fun_values, interpolated, amount))
        plt.xlabel('oś x')
        plt.ylabel('oś y')
        plt.title('Interpolacja')
        plt.plot(points, fun_values, 'b-', label='f')
        plt.plot(points, interpolated, 'g-', label='wielomian interpolujący')
        plt.plot(nodes, list(map(fun, nodes)), 'r.', label='punkty')
        plt.legend()
        plt.show()

main()
