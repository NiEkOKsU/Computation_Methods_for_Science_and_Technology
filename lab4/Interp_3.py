import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def f(x):
    return np.e**(4*np.cos(2*x))


def get_ys(xs):
    return [f(x) for x in xs]


def spline3(x_points, y_points, xs, boundary_cond):
    size = len(x_points) - 2
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i][i] = 4
        if i - 1 > 0:
             matrix[i][i - 1] = 1
        if i + 1 < size:
            matrix[i][i + 1] = 1

    h = [x_points[i+1] - x_points[i] for i in range(size)]
    g = [6 / (h[i]**2) * (y_points[i] - 2*y_points[i+1] + y_points[i+2]) for i in range(size)]
    h.append(x_points[-1] - x_points[-2])
    z = np.linalg.solve(matrix, g)

    z = list(z)
    if boundary_cond == 1:
        z = [0] + z + [0]
    else:
        z = [z[0]] + z + [z[-1]]

    a = []; b = []; c = []; d = []
    for i in range(size+1):
        a.append((z[i+1] - z[i]) / (6 * h[i]))
        b.append(0.5 * z[i])
        c.append((y_points[i+1] - y_points[i]) / h[i] - (z[i+1] + 2 * z[i]) / 6 * h[i])
        d.append(y_points[i])

    nr_fun = 0
    ys = []
    for i in range(len(xs)):
        while x_points[nr_fun + 1] < xs[i] < x_points[-1]:
            nr_fun += 1
        ys.append(get_val([d[nr_fun], c[nr_fun], b[nr_fun], a[nr_fun]], x_points[nr_fun], xs[i]))

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
    res = [['Liczba węzłów', 'spl3, nat', 'spl3, par']]

    for i in range(100, 101):
        print(i)
        plt.figure(figsize=(9, 6))
        xp = equidistant(start, end, i)
        yp = get_ys(xp)
        plt.plot(xp, yp, 'r.', markersize=10)
        plt.plot(xs, ys_or, 'y', label='funkcja interpolowana')
        r = [i]
        #if True: #spline == 3 or spline == 0:
        ys = spline3(xp, yp, xs, 1)
        plt.plot(xs, ys, 'grey', label='3 stopien naturalna')
        r.append(get_norm(ys, ys_or, 'max'))
        ys = spline3(xp, yp, xs, 2)
        plt.plot(xs, ys, 'm', label='3 stopien paraboliczna')
        r.append(get_norm(ys, ys_or, 'max'))
        res.append(r)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    df = pd.DataFrame(res)
    print(df)

main()
