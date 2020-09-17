import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt, e, c, epsilon_0
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate
import numpy.linalg as LA


def V12(x1, x2):
    return e ** 2 / (4.*math.pi*epsilon0)/(abs(x2-x1)) / eV


def varphi1(n1, x, R, L):
    kn = math.pi * (n1 + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2. + R / 2.))


def varphi2(n2, x, R, L):
    kn = math.pi * (n2 + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2. - R / 2.))


def varphi12(n1, n2, x1, x2, R, L):
    return varphi1(n1, x1, R, L) * varphi2(n2, x2, R, L)


def integral_V12(x2, x1, n1, n2, m1, m2, R, L, W, V_H):
    return varphi12(n1, n2, x1, x2, R, L)*bar_V12(x1, x2, R, L, W, V_H)*varphi12(m1, m2, x1, x2, R, L)


def Energy0_12(n1, n2, L):
    kn1 = math.pi * (n1 + 1) / L
    kn2 = math.pi * (n2 + 1) / L
    En1 = hbar ** 2 * kn1 ** 2 / (2.0 * me)
    En2 = hbar ** 2 * kn2 ** 2 / (2.0 * me)
    return (En1 + En2) / eV


def Qbit_varphi(x1, x2, an, n_max, R, L):
    phi = 0
    for m1 in range(0, DIM):
        for m2 in range(0, DIM):
            m = m1 * (n_max + 1) + m2
            phi += an[m] * varphi12(m1, m2, x1, x2, R, L)
    return phi.real


def integral_varphi_x1(x1, n_max, x, R, L, an):
    varphi = Qbit_varphi(x1, x, an, n_max, R, L)
    return abs(varphi) ** 2


def integral_varphi_x2(x2, n_max, x, R, L, an):
    varphi = Qbit_varphi(x, x2, an, n_max, R, L)
    return abs(varphi) ** 2


def rho(x, an, n_max, R, L):
    if (x <= -R / 2.0 + L / 2.0):
        x_min = R / 2 - L / 2
        x_max = R / 2 + L / 2
        result = integrate.quad(
            integral_varphi_x2,
            x_min, x_max,
            args=(n_max, x, R, L, an)
        )
    elif (R / 2.0 - L / 2.0 <= x):
        x_min = -R / 2 - L / 2
        x_max = -R / 2 + L / 2
        result = integrate.quad(
            integral_varphi_x1,
            x_min, x_max,
            args=(n_max, x, R, L, an)
        )
    real = result[0]
    return real


def bar_V1(x, R, L, W, V_H):
    if (x <= -R / 2.0 - W / 2.0):
        return 0
    elif (x <= -R / 2.0 + W / 2.0):
        return V_H
    else:
        return 0


def bar_V2(x, R, L, W, V_H):
    if (x <= R / 2.0 - W / 2.0):
        return 0
    elif (x <= R / 2.0 + W / 2.0):
        return V_H
    else:
        return 0


def bar_V12(x1, x2, R, L, W, V_H):
    return (V12(x1, x2) + bar_V1(x1, R, L, W, V_H) +
            bar_V2(x2, R, L, W, V_H) +
            phi1(x1, Ex) + phi2(x2, Ex))


def phi1(x1, Ex):
    return e * Ex * x1 / eV


def phi2(x2, Ex):
    return e * Ex * x2 / eV


def integral_Xnm(x1, x2, n1, n2, R, L, n_max, an):
    X = Qbit_varphi(x1, x2, an[n1], n_max, R, L) * (x1 + x2) * \
        Qbit_varphi(x1, x2, an[n2], n_max, R, L)
    return X.real


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 物理定数
h = Planck
hbar = hbar
me = m_e
eV = electron_volt
e = e
c = c
epsilon0 = epsilon_0
I = 0+1j

L = 1.e-9
R = 2.e-9
x_min = -L / 2.
x_max = L / 2.
n_max = 5
DIM = n_max + 1
N = 4

NX = 500
dx = 1.e-9

W = L / 5.
V_H = 30.
Ex = 10.e7

DIM1 = n_max + 1
DIM2 = DIM1 * DIM1


xs = []
phi = [0] * N
for n in range(len(phi)):
    phi[n] = [0] * (NX + 1)


x1_min = -R / 2.0 - L / 2.0
x1_max = -R / 2.0 + L / 2.0
x2_min = R / 2.0 - L / 2.0
x2_max = R / 2.0 + L / 2.0

matrix = [[0] * DIM2 for i in range(DIM2)]

for m1 in range(DIM1):
    for m2 in range(DIM1):
        for n1 in range(DIM1):
            for n2 in range(DIM1):
                result = integrate.dblquad(
                    integral_V12,
                    x1_min, x1_max,
                    lambda x: x2_min, lambda x: x2_max,
                    args=(n1, n2, m1, m2, R, L, W, V_H)
                )
                V_real = result[0]
                V_imag = 0j
                if (m2*n1*n2 == 0):
                    print(f'{m1},{m2},{n1},{n2}, {V_real}')
                nn = n1 * DIM1 + n2
                mm = m1 * DIM1 + m2
                matrix[mm][nn] = V_real + V_imag
                if (nn == mm):
                    matrix[mm][nn] += Energy0_12(n1, n2, L)

matrix = np.array(matrix)
result = LA.eig(matrix)
eig = result[0]
vec = result[1]
index = np.argsort(eig)
eigenvalues = eig[index]
vec = vec.T
vectors = vec[index]

for n in range(N):
    x_min = -R / 2 - L / 2
    x_max = R / 2 + L / 2

    for nx in range(NX + 1):
        x = x_min + (x_max - x_min) * nx / NX
        if (n == 0):
            xs.append(x / dx)
        if (x <= -R / 2.0 + L / 2.0):
            phi[n][nx] = rho(x, vectors[n], n_max, R, L) * 1.e-9
        elif (R / 2.0 - L / 2.0 <= x):
            phi[n][nx] = rho(x, vectors[n], n_max, R, L) * 1.e-9
        else:
            phi[n][nx] = 0

dE = (eigenvalues[1] - eigenvalues[0]) * eV
omega = dE / hbar
_lambda = 2. * math.pi * c / omega
print('エネルギー(基底状態)' + str(eigenvalues[0] / eV) + '(eV)')
print('エネルギー(励起状態)' + str(eigenvalues[1] / eV) + '(eV)')
print('エネルギー差' + str(dE / eV) + '(eV)')
print('電磁波の波長' + str(_lambda / 1.e-9) + '(nm)')

for n1 in range(N):
    for n2 in range(N):
        x1_min = -R / 2.0 - L / 2.0
        x1_max = -R / 2.0 + L / 2.0
        x2_min = R / 2.0 - L / 2.0
        x2_max = R / 2.0 + L / 2.0
        result = integrate.dblquad(
            integral_Xnm,
            x_min, x_max,
            lambda x: x2_min, lambda x: x2_max,
            args=(n1, n2, R, L, n_max, vectors)
        )
        real = result[0]
        imag = 0

        if (abs(real / L) < L):
            real = 0
        print('(' + str(n1) + ' ,' + str(n2) + ') ' + str(real / L))

for n in range(N):
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(xs, phi[n])

plt.show()
