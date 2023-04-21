import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def save(results):
    df = pd.DataFrame(results)
    print(df)
    df.to_excel(r'D:\Computation_Methods_for_Science_and_Technology\lab5\res_eq.xlsx')

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
    q, r = np.linalg.qr(matrix)
    y = np.dot(np.transpose(q), values)
    x = np.linalg.solve(r, y)
    return x


def main():
    wezly = [4,10,15,20,30,50,100]
    res = [['Liczba węzłów', 'stopien wielomanu','blad max', 'mse']]
    amount = 10000
    x0 = -math.pi
    x1 = 3 * math.pi
    numbers = range(amount)
    for n in wezly:
        for degree in range(3,10):
            if n<=degree:
                break
            points = list(map(lambda x: (x0 + x*(x1-x0)/amount), numbers))
            values = list(map(f, points))
            nodes = equadistant(x0, x1, n)
            regressed = [poly(regress(f, nodes, degree), points[i]) for i in range(amount)]
            res.append([n, degree, max_diff(regressed, values, amount), mean_square_error(regressed, values, amount)])
            print(n, degree)
            plt.xlabel('oś X')
            plt.ylabel('oś Y')
            plt.title('Aproksymacja')
            plt.plot(points, values, 'b-', points, regressed, 'grey',
                    nodes, list(map(f, nodes)), 'r.', markersize=10)
            plt.show()
    save(res)
main()