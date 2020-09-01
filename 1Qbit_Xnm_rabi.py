import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt, e, c
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate
import numpy.linalg as LA


def verphi(n, x):
    kn = math.pi * (n + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2.))


def V(x, Ex):
    if (abs(x) <= W / 2):
        return (e * Ex * x) + V_max
    else:
        return (e * Ex * x)


def Energy(n):
    kn = math.pi * (n + 1) / L
    return (hbar * kn) ** 2 / (2 * me)


def Qbit_verphi(n, x, an, n_max):
    phi = 0
    for m in range(n_max + 1):
        phi += an[n][m] * verphi(m, x)
    return phi


def integral_matrixElement(x, n1, n2, Ex):
    return verphi(n1, x) * V(x, Ex) * verphi(n2, x) / eV


def integral_Xnm(x, n1, n2, n_max, an):
    return Qbit_verphi(n1, x, an, n_max) * x * Qbit_verphi(n2, x, an, n_max)


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
c = c
I = 0+1j

L = 1.e-9
x_min = -L / 2
x_max = L / 2
n_max = 30
DIM = n_max + 1

NX = 500
dx = 1.e-9

W = L / 5
V_max = 30.0 * eV

Ex = 1.e8
N = 2

dt = 1.e-17
Tn = 30000000
skip = 100000
A0 = 1.e-8

eigenvalues = []
vectors = []

xs = []
phi = [0] * N
for n in range(N):
    phi[n] = [0] * (NX + 1)

averageX = [0] * N
matrix = []


class RK4:
    def __init__(self, N, dt):
        self.dt = dt
        self.N = N
        self.A0 = 0
        self.omega = 0
        self.Energy = [0] * N
        self.bn = np.array([0+0j] * N)
        self.dbn = np.array([0+0j] * N)
        Xnm = [0+0j]*N
        for i in range(N):
            Xnm[i] = [0+0j] * N
        self.Xnm = np.array(Xnm)
        self.__a1 = np.array([0+0j] * N)
        self.__a2 = np.array([0+0j] * N)
        self.__a3 = np.array([0+0j] * N)
        self.__a4 = np.array([0+0j] * N)

    def Db(self, t, bn, out_bn):
        for n in range(N):
            out_bn[n] = self.Energy[n] / (I * hbar) * bn[n]
            for m in range(N):
                out_bn[n] += self.A0 * e / (hbar**2) * math.cos(self.omega*t) * \
                    (self.Energy[n]-self.Energy[m])*self.Xnm[n][m] * bn[m]

    def timeEvolution(self, t):
        self.Db(t, self.bn, self.__a1)
        self.Db(t, self.bn + self.__a1 * 0.5 * self.dt, self.__a2)
        self.Db(t, self.bn + self.__a2 * 0.5 * self.dt, self.__a3)
        self.Db(t, self.bn + self.__a3 * self.dt, self.__a4)
        self.dbn = (self.__a1 + 2. * self.__a2 + 2. *
                    self.__a3 + self.__a4) * self.dt / 6.


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
    v = matrix @ vectors[i] - eigenvalues[i]*vectors[i]
    for j in range(DIM):
        sum += abs(v[j]) ** 2

print("MA-EA: " + str(sum))

for nx in range(NX+1):
    x = x_min + (x_max - x_min) * nx / NX
    xs.append(x/dx)
    for n in range(len(phi)):
        phi[n][nx] = abs(Qbit_verphi(n, x, vectors, n_max))**2/(1.e9)


for n in range(len(averageX)):
    result = integrate.quad(
        average_x,
        x_min, x_max,
        args=(vectors[n])
    )
    averageX[n] = result[0] * (1.e9)

dE = (eigenvalues[1] - eigenvalues[0]) * eV
omega = dE / hbar
_lambda = 2. * math.pi * c / omega

print('エネルギー（基底状態）:' + str(eigenvalues[0]) + '(eV)')
print('エネルギー（励起状態）:' + str(eigenvalues[1]) + '(eV)')
print('エネルギー差:' + str(dE / eV) + '(eV)')
print('エネルギー差に対応する光子の角振動数:' + str(omega) + '(rad/s)')
print('エネルギー差に対応する光子の角振動数に対する周期:' + str(2.*math.pi/omega) + '(s)')
print('電磁波の波長:' + str(_lambda / 1.e-9) + '(nm)')

rk4 = RK4(N, dt)
rk4.omega = omega
rk4.A0 = A0
rk4.Energy[0] = eigenvalues[0] * eV
rk4.Energy[1] = eigenvalues[1] * eV

for n1 in range(N):
    for n2 in range(N):
        result = integrate.quad(
            integral_Xnm,
            x_min, x_max,
            args=(n1, n2, n_max, vectors)
        )
        real = result[0]
        imag = 0
        rk4.Xnm[n1][n2] = real + 1j * imag
        if (abs(real / L) < L):
            real = 0
        print('(' + str(n1) + ' ,' + str(n2) + ') ' + str(real / L))


rk4.bn = np.array(
    [1+0j, 0+0j]
)
ts = []
b0s = []
b1s = []

for tn in range(Tn + 1):
    t_real = dt * tn
    if (tn % skip == 0):
        print("t=" + str(tn / skip) + ' ' +
              str(abs(rk4.bn[0]) ** 2)+' '+str(abs(rk4.bn[1]) ** 2)+' '+str(abs(rk4.bn[0]) ** 2+abs(rk4.bn[1]) ** 2))
        ts.append(tn / skip)
        b0s.append(abs(rk4.bn[0]) ** 2)
        b1s.append(abs(rk4.bn[1]) ** 2)

    rk4.timeEvolution(t_real)
    rk4.bn += rk4.dbn

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


fig4 = plt.figure(figsize=(10, 6))
plt.title('Expansion coefficient')
plt.xlabel('time(s)')
plt.ylabel('Expansion coefficient')
plt.xlim([0, Tn/skip])
plt.ylim([0, 1])
plt.plot(ts, b0s, marker='o', linewidth=3)
plt.plot(ts, b1s, marker='o', linewidth=3)

plt.show()
