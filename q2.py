import numpy as np

# 給定矩陣 A
A = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
], dtype=float)

n = A.shape[0]

L = np.eye(n)
U = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
    for j in range(i + 1, n):
        L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

def forward_substitution(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def backward_substitution(U, y):
    x = np.zeros_like(y)
    for i in reversed(range(len(y))):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

inv_A = np.zeros((n, n))
I = np.eye(n)

for col in range(n):
    y = forward_substitution(L, I[:, col])
    x = backward_substitution(U, y)
    inv_A[:, col] = x

# 輸出結果
np.set_printoptions(precision=5, suppress=True)
print("A 的反矩陣為：")
print(inv_A)
