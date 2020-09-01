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


def Qbit_varphi(n, x, an, n_max):
    phi = 0
    for m in range(n_max + 1):
        phi += an[n][m] * verphi(m, x)

    return phi


def Energy(n):
    kn = math.pi * (n + 1) / L
    return (hbar * kn) ** 2 / (2 * me)


def V(x, Ex):
    if (abs(x) <= W / 2):
        return (e * Ex * x) + V_max
    else:
        return (e * Ex * x)


def integral_matrixElement(x, n1, n2, Ex):
    return verphi(n1, x) * V(x, Ex) * verphi(n2, x) / eV


def integral_Xnm(x, n1, n2, n_max, an):
    return Qbit_varphi(n1, x, an, n_max) * x * Qbit_varphi(n2, x, an, n_max)


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
n_max = 20
DIM = n_max + 1

NX = 500
dx = 10 ** -9

W = L / 5
V_max = 30.0 * eV

Ex = 1.0e8
N = 2

eigenvalues = []
vectors = []

xs = []
phi = [0] * N
for n in range(N):
    phi[n] = [0] * (NX + 1)

averageX = [0] * N
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
eigenvalues = eig[index]
vec = vec.T
vectors = vec[index]

# for i in range(DIM):
# print(f'{i}番目の固有値:{eigenvalues[nEx][i]}')
# print(f'{i}番目の固有値に対応する固有ベクトル:\n{vectors[nEx][i]}')

# 検算
sum = 0
for i in range(DIM):
    v = matrix@vectors[i] - eigenvalues[i]*vectors[i]
    for j in range(DIM):
        sum += abs(v[j]) ** 2

print("MA-EA: " + str(sum))

for nx in range(NX+1):
    x = x_min + (x_max - x_min) * nx / NX
    xs.append(x/dx)
    for n in range(len(phi)):
        for m in range(n_max + 1):
            phi[n][nx] = abs(Qbit_varphi(n, x, vectors, n_max))**2/(1e9)


for n in range(len(averageX)):
    result = integrate.quad(
        average_x,
        x_min, x_max,
        args=(vectors[n])
    )
    averageX[n] = result[0]*(1e9)

for n1 in range(N):
    for n2 in range(N):
        result = integrate.quad(
            integral_Xnm,
            x_min, x_max,
            args=(n1, n2, n_max, vectors)
        )
        real = result[0]
        imag = 0
        if (abs(real / L) < L):
            real = 0
        print('('+str(n1)+' ,'+str(n2)+') '+str(real / L))

fig1 = plt.figure(figsize=(10, 6))
plt.title('Energy eigenvalues')
plt.xlabel('Level')
plt.ylabel('Energy(eV)')
plt.xlim([0, n_max])
exs = range(n_max + 1)
En = []
plt.plot(exs, eigenvalues, marker='o', linewidth=3)

fig2 = plt.figure(figsize=(10, 6))
plt.title('Existence probability at Position')
plt.xlabel("Position(nm)")
plt.ylabel('|phi|^2')
plt.xlim([-0.5, 0.5])
plt.ylim([0, 5.])
plt.plot(xs, phi[0], linewidth=3)
plt.plot(xs, phi[1], linewidth=3)


fig3 = plt.figure(figsize=(10, 6))
plt.title('Position at Electric field strength')
plt.xlabel('Electric field strength(V/m)')
plt.ylabel('Possint(nm)')
plt.xlim([0, 10])
print(averageX)
exs = range(1)

plt.plot(exs, averageX[0], marker='o', linewidth=3)
plt.plot(exs, averageX[1], marker='o', linewidth=3)

plt.show()
