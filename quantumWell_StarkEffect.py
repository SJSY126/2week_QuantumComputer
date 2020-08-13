import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt, e
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate
import numpy.linalg as LA


def wavenumber(E, m):
    return np.sqrt(2.0 * m * E / (hbar ** 2))


def verphi(n, x):
    kn = math.pi * (n + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2.))


def Energy(n):
    kn = math.pi * (n + 1) / L
    return (hbar * kn) ** 2 / (2 * me)


def phi(n, x, t):
    kn = math.pi * (n + 1) / L
    omega = hbar * kn**2 / (2 * me)
    return verphi(n, x) * cmath.exp(-I * omega * t)


def V(x, Ex):
    return (e * Ex * x)


def integral_matrixElement(x, n1, n2, Ex):
    return verphi(n1, x) * V(x, Ex) * verphi(n2, x) / eV


def average_x(x, a):
    sum = 0
    for n in range(n_max + 1):
        sum += a[n] * verphi(n, x)
    return x * sum**2


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 物理定数
h = Planck
hbar = hbar
me = m_e
eV = electron_volt
e = e
I = 0+1j

L = 10 ** -9
x_min = -L / 2
x_max = L / 2
n_max = 10
NEx = 10
Ex_max = 1e10
Ex_min = 0

NX = 500
dx = 10 ** -9
ts = -50
te = 50

DIM = n_max + 1

eigenvalues = [0] * (NEx + 1)
vectors = [0] * (NEx + 1)
for nEx in range(NEx + 1):
    eigenvalues[nEx] = []
    vectors[nEx] = []

xs = []
phi = [0] * (NEx + 1)
for nEx in range(NEx + 1):
    phi[nEx] = [0] * 2
    for n in range(len(phi[nEx])):
        phi[nEx][n] = [0] * (NX + 1)

averageX = [0] * 2
for n in range(len(averageX)):
    averageX[n] = [0] * (NEx + 1)

for nEx in range(NEx + 1):
    print('電場強度:' + str(nEx * 100 / NEx) + '%')
    Ex = Ex_max / NEx * nEx

    matrix = []

    for n1 in range(n_max + 1):
        col = []
        for n2 in range(n_max + 1):
            result = integrate.quad(
                integral_matrixElement,
                x_min, x_max,
                args=(n1, n2, Ex)
            )
            real = result[0]
            imag = 0j
            #print('('+str(n1)+','+str(n2)+') '+str(real))
            En = Energy(n1) / eV if (n1 == n2) else 0
            col.append(En + real)
        matrix.append(col)

    matrix = np.array(matrix)
    result = LA.eig(matrix)
    eig = result[0]
    vec = result[1]

    index = np.argsort(eig)
    eigenvalues[nEx] = eig[index]
    vec = vec.T
    vectors[nEx] = vec[index]

    # for i in range(DIM):
    # print(f'{i}番目の固有値:{eigenvalues[nEx][i]}')
    # print(f'{i}番目の固有値に対応する固有ベクトル:\n{vectors[nEx][i]}')

    # 検算
    sum = 0
    for i in range(DIM):
        v = matrix@vectors[nEx][i] - eigenvalues[nEx][i]*vectors[nEx][i]
        for j in range(DIM):
            sum += abs(v[j]) ** 2

    #print("MA-EA: " + str(sum))

    for nx in range(NX+1):
        x = x_min + (x_max - x_min) * nx / NX
        if (nEx == 0):
            xs.append(x/dx)
        for n in range(len(phi[nEx])):
            for m in range(n_max + 1):
                phi[nEx][n][nx] += vectors[nEx][n][m] * verphi(m, x)

            phi[nEx][n][nx] = abs(phi[nEx][n][nx]) ** 2 / (1e9)

    for n in range(len(averageX)):
        result = integrate.quad(
            average_x,
            x_min, x_max,
            args=(vectors[nEx][n])
        )
        averageX[n][nEx] = result[0]*(1e9)

fig1 = plt.figure(figsize=(10, 6))
plt.title('Energy at Electric field strength')
plt.xlabel('Electric field strength(V/m)')
plt.ylabel('Energy(eV)')
plt.xlim([0, 10])
exs = range(NEx + 1)
En_0 = []
En_1 = []
for nRx in range(NEx + 1):
    En_0.append(eigenvalues[nEx][0])
    En_1.append(eigenvalues[nEx][1])

plt.plot(exs, En_0, marker='o', linewidth=3)
plt.plot(exs, En_1, marker='o', linewidth=3)

fig2 = plt.figure(figsize=(10, 6))
plt.title('Existence probability at Position(n=0)')
plt.xlabel("Position(nm)")
plt.ylabel('|phi|^2')
plt.xlim([-0.5, 0.5])
plt.ylim([0, 4.])
for nEx in range(NEx + 1):
    plt.plot(xs, phi[nEx][0], linewidth=3)

fig3 = plt.figure(figsize=(10, 6))
plt.title('Existence probability at Position(n=1)')
plt.xlabel('Position(nm)')
plt.ylabel('|phi|^2')
plt.xlim([-0.5, 0.5])
plt.ylim([0, 3.])
for nEx in range(NEx + 1):
    plt.plot(xs, phi[nEx][1], linewidth=3)

fig4 = plt.figure(figsize=(10, 6))
plt.title('Position at Electric field strength')
plt.xlabel('Electric field strength(V/m)')
plt.ylabel('Possint(nm)')
plt.xlim([0, 10])
exs = range(NEx + 1)
plt.plot(exs, averageX[0], marker='o', linewidth=3)
plt.plot(exs, averageX[1], marker='o', linewidth=3)

plt.show()
