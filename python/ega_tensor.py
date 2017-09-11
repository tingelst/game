import sys
sys.path.append('../build')

import versor as vsr
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

# print(np.einsum('i,ji', ci, Dji))


Gkij2 = np.zeros((8, 8, 8))
# print(Gkij2)
for k in range(8):
    ek = vsr.EGA(0, 0, 0, 0, 0, 0, 0, 0)
    ek[k] = 1.0
    for i in range(8):
        ei = vsr.EGA(0, 0, 0, 0, 0, 0, 0, 0)
        ei[i] = 1.0
        A = Gkij2[k]
        A[i, :] = np.array(ek * ei)

M = np.zeros((8, 8))
mask = np.array([1, 2, 3, 4, 5, 6, 7, 8])
for i in range(8):
    W = np.zeros((8, 8))
    for j in range(8):
        a = vsr.EGA(0, 0, 0, 0, 0, 0, 0, 0)
        b = vsr.EGA(0, 0, 0, 0, 0, 0, 0, 0)
        a[i] = 1.
        b[j] = 1.
        M[i, j] = np.dot(mask, np.array(a * b))
        M = M.astype(np.int)


def cga_gp_tensor():
    dim = 32
    M = np.zeros((dim, dim))
    mask = np.arange(1, dim + 1)
    for i in range(dim):
        for j in range(dim):
            a = vsr.CGA(*[0] * 32)
            b = vsr.CGA(*[0] * 32)
            a[i] = 1.
            b[j] = 1.
            M[i, j] = np.dot(mask, np.array(a * b))
    tensor = np.zeros((dim, dim, dim))
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                val = M[i, j]
                if abs(val) == k + 1:
                    tensor[k, i, j] = np.sign(val)
    return tensor

def ega_gp_tensor():
    dim = 8
    M = np.zeros((dim, dim))
    mask = np.arange(1, dim + 1)
    for i in range(dim):
        for j in range(dim):
            a = vsr.EGA(*[0] * dim)
            b = vsr.EGA(*[0] * dim)
            a[i] = 1.
            b[j] = 1.
            # print(a * b)
            M[i, j] = np.dot(mask, np.array(a * b))
    tensor = np.zeros((dim, dim, dim))
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                val = M[i, j]
                if abs(val) == k + 1:
                    tensor[k, i, j] = np.sign(val)
    return tensor


Gkij = cga_gp_tensor()
a = vsr.CGA(*np.arange(1,33)) * 0.1
b = vsr.CGA(*np.arange(1,33)) * 0.1
# print(a * b)
# print(vsr.CGA(*np.einsum('i,j,kij->k', a, b, Gkij)))


Gkij = ega_gp_tensor()
for i in range(8):
    a = vsr.EGA(*np.arange(1,9))
    b = vsr.EGA(*[0]*8)
    b[i] = 1.0
    c = a * b
    d = vsr.EGA(*np.einsum('i,j,kij->k', a, b, Gkij))
    print(c)
    print(d)

Gkij = cga_gp_tensor()
for i in range(32):
    a = vsr.CGA(*np.arange(1,33))
    b = vsr.CGA(*[0]*32)
    b[i] = 1.0
    c = a * b
    d = vsr.CGA(*np.einsum('i,j,kij->k', a, b, Gkij))
    print(c)
    print(d)