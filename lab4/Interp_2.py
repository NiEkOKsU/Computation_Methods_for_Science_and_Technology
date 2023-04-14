import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save(results):
    df = pd.DataFrame(results)
    print(df)
    df.to_excel("res_eq.xlsx", sheet_name='sheet1')

def f(x):
    return np.e**(4*np.cos(2*x))


def get_ys(xs):
    return [f(x) for x in xs]

def spline2(x_points, y_points, xs, boundary_cond):
    size = len(x_points) - 1
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j or j == i-1:
                matrix[i][j] = 1

    h = [x_points[i+1] - x_points[i] for i in range(size)]
    g = [2 / h[i] * (y_points[i+1] - y_points[i]) for i in range(size)]

    b = np.linalg.solve(matrix, g)

    b = list(b)
    b = [0] + b

    a = [(b[i+1] - b[i]) / (2 * h[i]) for i in range(size)]
    c = [y_points[i] for i in range(size)]

    if boundary_cond == 2:
        b[0] = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        a[0] = 0

    nr_fun = 0
    ys = []
    for i in range(len(xs)):
        while x_points[nr_fun + 1] < xs[i] < x_points[-1]:
            nr_fun += 1
        ys.append(get_val([c[nr_fun], b[nr_fun], a[nr_fun]], x_points[nr_fun], xs[i]))

    return ys


def get_val(coeff, xi, x):
    val = 0
    for i, elem in enumerate(coeff):
        val += elem * (x - xi) ** i
    return val


def get_norm(y1, y2, which):
    n = len(y1)
    if which == "max":
        max_df = 0
        for i in range(n):
            max_df = max(abs(y1[i] - y2[i]), max_df)
        return max_df
    if which == "eu":
        error = 0
        for i in range(n):
            error += (y1[i] - y2[i]) ** 2
        error = np.sqrt(error) / n
        return error


def chebyshew(a, b, no):
    ret = sorted(np.array([np.cos(((2 * j - 1) / (2 * no)) * np.pi) for j in range(1, no + 1)], dtype=np.float64))
    for i in range(len(ret)):
        ret[i] = 0.5 * (a + b) + 0.5 * (b - a) * ret[i]
    return ret


def equidistant(a, b, n):
    step = (b-a)/(n-1)
    ret = []
    for _ in range(n):
        ret.append(a)
        a += step
    return ret


def main():
    start = -np.pi
    end = 3*np.pi
    n_draw = 5000

    xs = equidistant(start, end, n_draw)
    ys_or = get_ys(xs)
    res = [['Liczba węzłów', 'spl2, nat', 'spl2, lin']]

    for i in range(100, 101):
        print(i)
        plt.figure(figsize=(9, 6))
        xp = equidistant(start, end, i)
        yp = get_ys(xp)
        plt.plot(xp, yp, 'r.', markersize=10)
        plt.plot(xs, ys_or, 'y', label='funkcja interpolowana')
        r = [i]
        #if True: #spline == 2 or spline == 0:
        ys = spline2(xp, yp, xs, 1)
        plt.plot(xs, ys, 'grey', label='2 stopien naturalna')
        r.append(get_norm(ys, ys_or, 'eu'))
        ys = spline2(xp, yp, xs, 2)
        plt.plot(xs, ys, 'm', label='2 stopien, pierwsza funkcja liniowa')
        r.append(get_norm(ys, ys_or, 'eu'))
        res.append(r)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    df = pd.DataFrame(res)
    print(df)

main()
