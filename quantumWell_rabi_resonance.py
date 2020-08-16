import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt, e, c
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate


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
    if (abs(x) <= W / 2):
        return (e * Ex * x) + V_max
    else:
        return (e * Ex * x)


def integral_matrixElement(x, n1, n2, Ex):
    return verphi(n1, x) * V(x, Ex) * verphi(n2, x) / eV


def integral_Xnm(x, n1, n2):
    return verphi(n1, x) * x * verphi(n2, x)


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

L = 10 ** -9
x_min = -L / 2
x_max = L / 2

n_max = 3
DIM = n_max + 1

dt = 1e-18
Tn = 1000000


A0 = 1e-8


class RK4:
    def __init__(self, DIM, dt):
        self.dt = dt
        self.DIM = DIM
        self.bn = np.array([0+0j] * DIM)
        self.dbn = np.array([0+0j] * DIM)
        self.__a1 = np.array([0+0j] * DIM)
        self.__a2 = np.array([0+0j] * DIM)
        self.__a3 = np.array([0+0j] * DIM)
        self.__a4 = np.array([0+0j] * DIM)
        Xnm = [0+0j]*DIM
        for i in range(DIM):
            Xnm[i] = [0+0j] * DIM
        self.Xnm = np.array(Xnm)

    def Db(self, t, bn, out_bn):
        for n in range(DIM):
            out_bn[n] = Energy(n) / (I * hbar) * bn[n]
            for m in range(DIM):
                out_bn[n] += A0 * e / (hbar**2) * math.cos(self.omega*t) * \
                    (Energy(n)-Energy(m))*self.Xnm[n][m] * bn[m]

    def timeEvolution(self, t):
        self.Db(t, self.bn, self.__a1)
        self.Db(t, self.bn + self.__a1 * 0.5 * self.dt, self.__a2)
        self.Db(t, self.bn + self.__a2 * 0.5 * self.dt, self.__a3)
        self.Db(t, self.bn + self.__a3 * self.dt, self.__a4)
        self.dbn = (self.__a1 + 2 * self.__a2 + 2 *
                    self.__a3 + self.__a4) * self.dt / 6


omegas = []
maxB1s = []
dE = Energy(1) - Energy(0)
omega = dE / hbar
_lambda = 2. * math.pi * c / omega

Nomega = 100
for n_omega in range(Nomega + 1):
    omega0 = dE / hbar
    omega = omega0 * (1.0 + 0.1 * (n_omega - Nomega / 2) / Nomega)

    omegas.append(omega / omega0 * 100)

    rk4 = RK4(DIM, dt)
    rk4.omega = omega
    rk4.A0 = A0

    for n1 in range(n_max + 1):
        for n2 in range(n_max + 1):
            result = integrate.quad(
                integral_Xnm,
                x_min, x_max,
                args=(n1, n2)
            )
            real = result[0]
            imag = 0
            rk4.Xnm[n1][n2] = real + 1j * imag

            if (abs(real / L) < L):
                real = 0

            print('(' + str(n1) + ", " + str(n2) + ") " + str(real / L))

    rk4.bn = np.array(
        [1+0j, 0+0j, 0+0j, 0+0j]
    )

    maxB1 = 0
    ts = []
    b0s = []
    b1s = []

    for tn in range(Tn + 1):
        t_real = dt * tn
        rk4.timeEvolution(t_real)
        rk4.bn += rk4.dbn
        if (maxB1 < abs(rk4.bn[1] ** 2)):
            maxB1 = abs(rk4.bn[1]) ** 2
        #print(str(tn) + '/' + str(Tn + 1))

    print(str(omega / omega0 * 100) + ' ' + str(maxB1))
    maxB1s.append(maxB1)


print('エネルギー（基底状態）:' + str(Energy(0) / eV) + '(eV)')
print('エネルギー（励起状態）:' + str(Energy(1) / eV) + '(eV)')
print('エネルギー差:' + str(dE / eV) + '(eV)')
print('エネルギー差に対応する光子の角振動数:' + str(omega) + '(rad/s)')
print('エネルギー差に対応する光子の角振動数に対する周期:' + str(2.*math.pi/omega) + '(s)')
print('電磁波の波長:' + str(_lambda/1e-9) + '(nm)')


plt.figure(figsize=(8, 6))
plt.title('Max Existence probability at omega')
plt.xlabel('omega')
plt.ylabel('Max Existence probability')
plt.xlim([95, 105])
plt.ylim([0, 1])
plt.plot(omegas, maxB1s, marker="o",  linewidth=3.0)
plt.show()
