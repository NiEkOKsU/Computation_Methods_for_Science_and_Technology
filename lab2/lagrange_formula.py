import math
import matplotlib.pyplot as plt

def fun(x):
    return math.e**(4*math.cos(2*x))

def equadistant(x0, x1, n):
    result = [x0 + i*(x1-x0)/(n-1) for i in range(n)]
    return result

def chebyshew(x0, x1, n):
    result = [1/2*(x0 + x1) + 1/2*(x1 - x0)*math.cos((2*i - 1)*math.pi/(2*n)) for i in range(1, n+1)]
    return result

def lagrange(x, n, nodes):
    values = list(map(fun, nodes))
    sum = 0
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term *=  (x - nodes[j]) / (nodes[i] - nodes[j])
        term *= values[i]
        sum += term
    return sum

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
    nodes_num = [2,4,5,7,8,10,13,15,17,20,50]
    for n in nodes_num:
        nodes = equadistant(low_bound_x, up_bound_x, n)
        interpolated = [lagrange(points[i], n, nodes) for i in range(amount)]
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