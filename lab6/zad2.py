import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def save(results):
    df = pd.DataFrame(results)
    print(df)
    df.to_excel(r'D:\Computation_Methods_for_Science_and_Technology\lab6\res_eq.xlsx')

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


def get_trig_coeff(m, k, nodes):
    akl = []
    bkl = []
    n = len(nodes)
    for i in range(n):
        akl.append(f(nodes[i]*3/2) * np.cos(k*nodes[i]))
        bkl.append(f(nodes[i]*3/2) * np.sin(k*nodes[i]))
    return (sum(akl) / (n/2)), (sum(bkl) / (n/2))


def approximate_trig(x, nodes, m):  # N is for degree of the approximation
    a0 = get_trig_coeff(m, 0, nodes)[0]
    res = 0
    for k in range(1, m):
        ak, bk = get_trig_coeff(m, k, nodes)
        res += ak * np.cos(2/3 * k * x)
        res += bk * np.sin(2/3 * k * x)
    res += a0 / 2
    return res


def main():
    wezly = [4,10,15,20,30,50,100]
    res = [['Liczba węzłów', 'stopien wielomanu','blad max', 'mse']]
    amount = 10000
    x0 = -math.pi
    x1 = 3 * math.pi
    numbers = range(amount)
    for n in wezly:
        for degree in range(2,10):
            if n<=degree:
                break
            points = list(map(lambda x: (x0 + x*(x1-x0)/amount), numbers))
            values = list(map(f, points))
            nodes = equadistant(x0, x1, n)
            regressed = [regressed.append(approximate_trig(points[n], nodes, degree))]
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
