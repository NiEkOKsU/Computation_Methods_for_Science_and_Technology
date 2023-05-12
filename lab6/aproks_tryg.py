import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

def save(results):
    df = pd.DataFrame(results)
    print(df)
    df.to_excel(r'C:\Users\proks\OneDrive\Pulpit\Computation_Methods_for_Science_and_Technology\lab6\res.xlsx')

class TrigonometricApproximation:
    def __init__(self, X, Y, n, m, start, stop):
        if m > np.floor((n-1)/2):
            raise Exception("m cannot be greater than floor of (n-1)/2")
        self.X = X
        self.Y = Y
        self.n = n
        self.m = m
        self.start = start
        self.stop = stop
        self.A = np.zeros(self.n)
        self.B = np.zeros(self.n)
        self.scale_to_2pi()
        self.compute_A_and_B()
        self.scale_from_2pi()

    def scale_to_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] /= range_length
            self.X[i] *= 2 * np.pi
            self.X[i] += -np.pi - (2 * np.pi * self.start / range_length)

    def compute_A_and_B(self):
        for i in range(self.n):
            ai = sum(self.Y[j] * np.cos(i * self.X[j]) for j in range(self.n))
            bi = sum(self.Y[j] * np.sin(i * self.X[j]) for j in range(self.n))
            self.A[i] = 2 * ai / self.n
            self.B[i] = 2 * bi / self.n

    def scale_from_2pi(self):
        range_length = self.stop - self.start
        for i in range(len(self.X)):
            self.X[i] -= -np.pi - (2 * np.pi * self.start / range_length)
            self.X[i] /= 2 * np.pi
            self.X[i] *= range_length

    def scale_point_to_2pi(self, x):
        range_length = self.stop - self.start
        x /= range_length
        x *= 2 * np.pi
        x += -np.pi - (2 * np.pi * self.start / range_length)
        return x

    def approximate(self, X):
        points = []
        for x in X:
            cp_x = deepcopy(x)
            cp_x = self.scale_point_to_2pi(cp_x)
            approximated_x = 1 / 2 * self.A[0] + sum(self.A[j] * np.cos(j * cp_x) + self.B[j] * np.sin(j * cp_x)
                                                     for j in range(1, self.m + 1))
            points.append(approximated_x)
        return points

def f(x):
    return np.e**(4*np.cos(2*x))

def max_error(Y1, Y2):
    return max([abs(Y1[i] - Y2[i]) for i in range(len(Y1))])

def mean_square_error(Y1, Y2):
    return np.sqrt(sum([(Y1[i] - Y2[i])**2 for i in range(len(Y1))]))/1000

def visualize(x, y, start, stop, n, m, function):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label="Węzły", color="red")
    X = np.arange(start, stop + 0.01, 0.01)
    plt.plot(X, f(X), label="Funkcja aproksymowana", color="blue")
    plt.plot(X, function(X), label="Wielomian aproksymujący", color="grey")
    plt.title(f"Aproksymacji trygonometryczna dla {n} węzłów i m={m}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    
def trig_approximation(start, stop, n, m):
    X = np.linspace(start, stop, n)
    Y = f(X)
    trigonometric_approximation = TrigonometricApproximation(X, Y, n, m, start, stop)
    visualize(X, Y, start, stop, n, m, trigonometric_approximation.approximate)
    
def main():
    wezly = [5, 10, 15, 30,50, 100]
    res = [['Liczba węzłów', 'stopien wielomanu','blad max', 'mse']]
    amount = 1000
    x0 = -np.pi
    x1 = 3 * np.pi
    total_X = np.linspace(x0, x1, amount)
    func_val = f(total_X)
    for n in wezly:
        for f_num in [2,3,4,5,6,7,8,9,10,15,20,25,30]:
            if f_num > np.floor((n-1)/2):
                break
            X = np.linspace(x0, x1, n)
            Y = f(X)
            trigonometric_approximation = TrigonometricApproximation(X, Y, n, f_num, x0, x1)
            trig_appr_result = trigonometric_approximation.approximate(total_X)
            res.append([n, f_num, max_error(trig_appr_result, func_val), mean_square_error(trig_appr_result, func_val)])
            #visualize(X, Y, x0, x1, n, f_num, trigonometric_approximation.approximate)
            print(trigonometric_approximation.A)
    #save(res) żeby zapisać wyniki i je wyświtlić należy odkomentowaćtą linijkę
main()