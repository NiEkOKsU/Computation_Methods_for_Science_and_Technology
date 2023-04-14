import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def save(results):
    df = pd.DataFrame(results)
    print(df)
    df.to_excel("res_eq.xlsx", sheet_name='sheet1')

def chebyshew(x0, x1, n):
    result = []
    for i in range(1, n+1, 1):
        result.append(1/2*(x0 + x1) + 1/2*(x1 - x0)*math.cos((2*i - 1)*math.pi/(2*n)))
    return result


def equadistant(x0, x1, n):
    result = []
    for i in range(n):
        result.append(x0 + i*(x1-x0)/(n-1))
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


def f(x):
    return np.e**(4*np.cos(2*x))


def poly(vec, x):
    result = 0
    for i in range(len(vec)):
        result += vec[i]*x**i
    return result


def regress(f, nodes, degree):
    values = np.array(list(map(f, nodes)))
    n = len(nodes)
    matrix = np.zeros((n, degree+1))
    for i in range(n):
        for j in range(degree+1):
            matrix[i][j] += nodes[i]**j
    return np.linalg.inv(np.transpose(matrix).dot(matrix)).dot(np.transpose(matrix)).dot(values)


def main():
    wezly = [4,10,15,20,30,50,100]
    res = [['Liczba węzłów', 'stopien wielomanu','blad max', 'mse']]
    for n in wezly:
        for degree in range(2,10):
            if n<=degree:
                break
            amount = 10000
            x0 = -math.pi
            x1 = 3 * math.pi
            numbers = range(amount)
            points = list(map(lambda x: (x0 + x*(x1-x0)/amount), numbers))
            values = list(map(f, points))
            nodes = equadistant(x0, x1, n)
            regressed = []

            for i in range(amount):
                regressed.append(poly(regress(f, nodes, degree), points[i]))
            res.append([n, degree, max_diff(regressed, values, n), mean_square_error(regressed, values, n)])

            plt.xlabel('oś X')
            plt.ylabel('oś Y')
            plt.title('Aproksymacja')
            plt.plot(points, values, 'b-', points, regressed, 'r-',
                    nodes, list(map(f, nodes)), 'y.')
            plt.show()

main()