import numpy as np
import pandas as pd
from time import perf_counter

def save(results):
    print(results)
    path = r'D:\Computation_Methods_for_Science_and_Technology\lab10\resspektral.xlsx'
    results.to_excel(path)

def jacobi_method(A, b, stop_cond, epsilon, max_iters):
    D = np.diag(A)
    R = A - np.diagflat(D)
    X = np.zeros_like(b)
    for i in range(len(X)):
        X[i] = 100
    iters = 0
    for _ in range(max_iters):
        X_new = (b - (R @ X)) / D
        iters += 1
        if stop_cond == 1 and np.linalg.norm(X_new - X) < epsilon:
            break
        elif stop_cond == 2 and np.linalg.norm(A @ X - b) < epsilon:
            break
        X = X_new
    return X, iters

def matrix_create(A, n, k, m):
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                A[i-1][i-1] = k
            elif i-1 == j:
                A[i-1][j-1] = m/i
            elif j > i:
                A[i-1][j-1] = (-1)**j * m/j
    return A

def exercise_1(numbers, epsilon, k, m, max_iters):
    result = []
    for n in numbers:
        A = np.zeros((n, n))
        A = matrix_create(A, n, k, m)
        X_vec = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
        b = A @ X_vec
        # stop condition 1
        start = perf_counter()
        X, first_iters = jacobi_method(A, b, 1, epsilon, max_iters)
        end = perf_counter()
        first_time = end - start
        first_norm = np.linalg.norm(X_vec - X)

        # stop condition 2
        start = perf_counter()
        X, second_iters = jacobi_method(A, b, 2, epsilon, max_iters)
        end = perf_counter()
        second_time = end - start
        second_norm = np.linalg.norm(X_vec - X)

        result += [first_iters, second_iters, first_time, second_time, first_norm, second_norm]
    df = pd.DataFrame(data={"n": numbers,
                            "1st condition iters": result[::6],
                            "2nd condition iters": result[1::6],
                            "1st condition time [s]": result[2::6],
                            "2nd condition time [s]": result[3::6],
                            "1st condition norm": result[4::6],
                            "2nd condition norm": result[5::6]})
    return df

def spectral_radius(A):
    D = np.diag(A)
    R = A - np.diagflat(D)
    S = R / D
    eigvals = np.linalg.eigvals(S)
    return max(abs(i) for i in eigvals)

def exercise_2(numbers, k, m):
    result = []
    for n in numbers:
        A = np.zeros((n, n)).astype(np.float64)
        A = matrix_create(A, n, k, m)
        spec_rad = spectral_radius(A)
        condition = True
        if spec_rad >= 1:
            condition = False
        result += [spec_rad, condition]
    df = pd.DataFrame(data={"n": numbers,
                            "spectral radius": result[::2],
                            "condition": result[1::2]})
    return df

numbers = [3, 4, 5, 7, 10, 12, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 1000]
df_1 = exercise_2(numbers, 7, 1)
save(df_1)