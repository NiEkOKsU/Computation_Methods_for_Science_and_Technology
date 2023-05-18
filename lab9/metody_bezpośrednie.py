import numpy as np
import pandas as pd
import time
pd.options.display.float_format = "{:,.12f}".format

def save(results):
    df = pd.DataFrame(results)
    print(df)
    path = r'D:\Computation_Methods_for_Science_and_Technology\lab9\res.xlsx'
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

numbers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 50, 100, 200]
save(exercise_1(numbers))
