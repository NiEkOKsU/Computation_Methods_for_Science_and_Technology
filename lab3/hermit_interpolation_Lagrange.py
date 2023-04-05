import numpy as np
import math
import matplotlib.pyplot as plt

def fun(x):
    return math.e**(4*math.cos(2*x))

def dfun(x):
    return -8*math.e**(4*math.cos(2*x))*math.sin(2*x)

def hermie(x, nodes):
    n = len(nodes)
    funvals = list(map(fun, nodes))
    dfunvals = list(map(dfun, nodes))
    work_2n = [0 for _ in range(2 * n)]
    for i in range(n):
        prod = 1
        sum = 0
        for j in range(n):
            if i != j:
                prod *= (nodes[i] - nodes[j])
                sum += 1 / (nodes[i] - nodes[j])
        work_2n[i] = 1 / prod
        work_2n[n + i] += sum
    sum = 0
    for i in range(n):
        prod = 1
        for j in range(n):
            if i != j:
                prod *= (x - nodes[j])
        
        prod *= work_2n[i]
        prod *= prod

        sum += (1 - 2 * (x - nodes[i]) * work_2n[n + i]) * prod * funvals[i]
        sum += (x - nodes[i]) * prod * dfunvals[i]
    return sum

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
    nodes_num = [2,4,5,7,8,10,12,15,17,20]
    for n in nodes_num:
        nodes = chebyshew(low_bound_x, up_bound_x, n)
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