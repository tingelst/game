import numpy as np


def gp_tensor():
    gp_table = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 2, 1, 7, -6, 8, -4, 3, 5, 3, -7, 1, 5, 4, 8,
        -2, 6, 4, 6, -5, 1, -3, 2, 8, 7, 5, 8, -4, 3, -1, -7, 6, -2, 6, 4, 8,
        -2, 7, -1, -5, -3, 7, -3, 2, 8, -6, 5, -1, -4, 8, 5, 6, 7, -2, -3, -4,
        -1
    ]).reshape(8, 8)
    tensor = np.zeros((8, 8, 8))
    for k in range(8):
        for i in range(8):
            for j in range(8):
                val = gp_table[i, j]
                if abs(val) == k + 1:
                    tensor[k, i, j] = np.sign(val)
    return tensor


def ip_tensor():
    ip_table = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 3, 5, 0, 0, 1, 0, 4, 0, -2,
        6, 0, 0, 0, 1, -3, 2, 0, 7, 0, 0, -4, 3, -1, 0, 0, -2, 0, 4, 0, -2, 0,
        -1, 0, -3, 0, -3, 2, 0, 0, 0, -1, -4, 0, 5, 6, 7, -2, -3, -4, -1
    ]).reshape(8, 8)
    tensor = np.zeros((8, 8, 8))
    for k in range(8):
        for i in range(8):
            for j in range(8):
                val = ip_table[i, j]
                if abs(val) == k + 1:
                    tensor[k, i, j] = np.sign(val)
    return tensor


def op_tensor():
    op_table = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 2, 0, 7, -6, 8, 0, 0, 0, 3, -7, 0, 5, 0, 8, 0,
        0, 4, 6, -5, 0, 0, 0, 8, 0, 5, 8, 0, 0, 0, 0, 0, 0, 6, 0, 8, 0, 0, 0,
        0, 0, 7, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(8, 8)
    tensor = np.zeros((8, 8, 8))
    for k in range(8):
        for i in range(8):
            for j in range(8):
                val = op_table[i, j]
                if abs(val) == k + 1:
                    tensor[k, i, j] = np.sign(val)
    return tensor


def dual_tensor():
    dual_table = np.array([-8, -5, -6, -7, 2, 3, 4, 1])
    tensor = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            val = dual_table[j]
            if abs(val) == i + 1:
                tensor[i, j] = np.sign(val)
    return tensor


Gkij = gp_tensor()
Ikij = ip_tensor()
Okij = op_tensor()
Dji = dual_tensor()

np.save('Gkij.txt', Gkij)

ai = np.array([0, 1, 2, 3, 0, 0, 0, 0])
bj = np.array([0, 0, 1, 0, 0, 0, 0, 0])
ci = np.array([0, 0, 0, 0, 1, 2, 3, 0])

# print(ai)
# print(bj)
# print(np.einsum('i,j,kij->k', ai, bj, Gkij))
# print(np.einsum('i,j,kij->k', ai, bj, Ikij))
# print(np.einsum('i,j,kij->k', ai, bj, Okij))

print(np.einsum('i,ji', ci, Dji))
