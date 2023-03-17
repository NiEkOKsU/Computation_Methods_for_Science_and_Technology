import numpy as np
import math
import matplotlib.pyplot as plt

def fun(x):
    return math.sin(x) * math.sin(x**2/math.pi)

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

def main():
    amount = 1000
    low_bound_x = -math.pi
    up_bound_x = 2 * math.pi
    points = list(map(lambda x: (low_bound_x + x*(up_bound_x-low_bound_x)/amount), range(amount)))
    fun_values = list(map(fun, points))
    n = int(input("Podaj ilość węzłów: "))
    interpolated = [newton(points[i], chebyshew(low_bound_x, up_bound_x, n)) for i in range(amount)]
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Interpolation')
    plt.plot(points, fun_values, 'b-', points, interpolated, 'r-',
         chebyshew(low_bound_x, up_bound_x, n), list(map(fun, chebyshew(low_bound_x, up_bound_x, n))), 'y.')
    plt.show()

main()