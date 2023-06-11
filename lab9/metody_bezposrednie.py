import numpy as np
import pandas as pd
import time
pd.options.display.float_format = "{:,.12f}".format

def save(results):
    df = pd.DataFrame(results)
    print(df)
    path = r'C:\Users\proks\OneDrive\Pulpit\Computation_Methods_for_Science_and_Technology\lab9\res4.xlsx'
    df.to_excel(path)

def gaussian_elimination(A, B):
    start_time = time.time()
    n = len(A)
    for i in range(n-1):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            B[j] -= factor * B[i]
    
    X = np.zeros(n)
    X[n-1] = B[n-1] / A[n-1][n-1]
    for i in range(n-2, -1, -1):
        sum = B[i]
        for j in range(i+1, n):
            sum -= A[i][j] * X[j]
        X[i] = sum / A[i][i]
    return X, time.time() - start_time

def exercise_1(numbers):
    result = []
    for n in numbers:
        for float_type in [np.float32, np.float64, np.longdouble]:
            A = np.array([[1 / (i + j - 1) if i != 1 else 1 for j in range(1, n + 1)] for i in range(1, n + 1)]).astype(float_type)
            X_vec = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
            B = A @ X_vec
            X, timme = gaussian_elimination(A, B)
            norm = np.linalg.norm(X_vec - X)
            result += [norm]
            result += [timme]
    df = pd.DataFrame(data={"n": numbers,
                            "float32": result[::6],
                            "czas dla float32": result[1::6],
                            "float64": result[2::6],
                            "czas dla float64": result[3::6],
                            "float128": result[4::6],
                            "czas dla float128": result[5::6]})
    return df

def exercise_2(numbers):
    result = []
    for n in numbers:
        for float_type in [np.float32, np.float64, np.longdouble]:
            A = np.zeros((n, n)).astype(float_type)
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if j >= i:
                        A[i - 1][j - 1] = 2 * i / j
                    else:
                        A[i - 1][j - 1] = A[j - 1][i - 1]
            X_vec = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
            B = A @ X_vec
            X, timme = gaussian_elimination(A, B)
            print(X.ndim)
            norm = np.linalg.norm(X_vec - X).astype(float_type)
            result += [norm]
            result += [timme]
    df = pd.DataFrame(data={"n": numbers,
                            "float32": result[::6],
                            "czas dla float32": result[1::6],
                            "float64": result[2::6],
                            "czas dla float64": result[3::6],
                            "float128": result[4::6],
                            "czas dla float128": result[5::6]})
    return df

def norm(A):
    n = len(A)
    return max(sum(A[i][j] for j in range(n)) for i in range(n))

def create_A1(n):
    return np.array([[1 / (i + j - 1) if i != 1 else 1 for j in range(1, n + 1)] for i in range(1, n + 1)])

def create_A2(n):
    A = np.zeros((n, n))
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if j >= i:
                A[i - 1][j - 1] = 2 * i / j
            else:
                A[i - 1][j - 1] = A[j - 1][i - 1]
    return A

def conditioning_factor(A):
    A_inv = np.linalg.inv(A)
    return norm(A_inv) * norm(A)

def condition_number(numbers):
    result = []
    for n in numbers:
        con_num_1 = conditioning_factor(create_A1(n))
        con_num_2 = conditioning_factor(create_A2(n))
        result += [con_num_1, con_num_2]
    df = pd.DataFrame(data={"n":numbers,
                            "ex 1 condition number":result[::2],
                            "ex 2 condition number":result[1::2]})
    return df

def thomas_algorithm(A, B):
    start_time = time.time()
    n = np.shape(A)[0]
    C = np.zeros(n)
    C[0] = A[0][0]

    X = np.zeros(n)
    X[0] = B[0]

    for i in range(1, n):
        ratio = A[i][i - 1] / C[i - 1]
        C[i] = A[i][i] - ratio * A[i - 1][i]
        X[i] = B[i] - ratio * X[i - 1]

    X[n - 1] = X[n - 1] / C[n - 1]
    for i in range(n - 2, -1, -1):
        X[i] = (X[i] - A[i][i + 1] * X[i + 1]) / C[i]
    return X, time.time() - start_time

def exercise_3(numbers, k, m):
    result = []
    for n in numbers:
        A = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    A[i - 1][j - 1] = -m * i - k
                elif j == i + 1:
                    A[i - 1][j - 1] = i
                elif i > j == i - 1:
                    A[i - 1][j - 1] = m / i
        X_vec = np.array([1 if i % 2 == 0 else -1 for i in range(n)])
        B = A @ X_vec
        X_gaussian, gaussian_time = gaussian_elimination(A, B)
        norm_gaussian = np.linalg.norm(X_vec - X_gaussian)

        X_thomas, thomas_time = thomas_algorithm(A, B)
        norm_thomas = np.linalg.norm(X_vec - X_thomas)
        result += [norm_gaussian, norm_thomas, gaussian_time, thomas_time]
    df = pd.DataFrame(data={"n": numbers,
                            "gaussian norm": result[::4],
                            "thomas norm": result[1::4],
                            "gaussian time [s]": result[2::4],
                            "thomas time [s]": result[3::4]})
    return df

numbers = [1000]
save(exercise_3(numbers,7,3))
