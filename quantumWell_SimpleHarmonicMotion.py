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
    if (abs(x) <= W / 2):
        return (e * Ex * x) + V_max
    else:
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

n_max = 2
DIM = n_max + 1

dt = 1e-16
Tn = 300
skip = 1

T0 = 2.*math.pi*hbar/Energy(0)
print('E0 = ' + str(Energy(0) / eV) + '(eV)')
print('T0 = ' + str(T0) + '(s)')


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

    def Db(self, t, bn, out_bn):
        for n in range(DIM):
            out_bn[n] = Energy(n) / (I * hbar) * bn[n]

    def timeEvolution(self, t):
        self.Db(t, self.bn, self.__a1)
        self.Db(t, self.bn + self.__a1 * 0.5 * self.dt, self.__a2)
        self.Db(t, self.bn + self.__a2 * 0.5 * self.dt, self.__a3)
        self.Db(t, self.bn + self.__a3 * self.dt, self.__a4)
        self.dbn = (self.__a1 + 2 * self.__a2 + 2 *
                    self.__a3 + self.__a4) * self.dt / 6


ts = []
b0s = []

rk4 = RK4(DIM, dt)
rk4.bn = np.array(
    [1.+0.j, 0.+0.j, 0.+0.j]
)

for tn in range(Tn + 1):
    t_real = dt * tn
    if (tn % skip == 0):
        ts.append(tn)
        b0s.append(rk4.bn[0])

    rk4.timeEvolution(t_real)
    rk4.bn += rk4.dbn

plt.figure(figsize=(8, 6))
plt.title("Expansion coefficient at time")
plt.xlabel("time(s)")
plt.ylabel('Expansion coefficient')
plt.xlim([0, Tn / skip])
plt.ylim([-1, 1])
plt.plot(ts, np.real(b0s), marker="o",  linewidth=3.0)
plt.plot(ts, np.imag(b0s), marker="o",  linewidth=3.0)
plt.show()
