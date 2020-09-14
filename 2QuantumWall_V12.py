import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt, e, c, epsilon_0
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate
import numpy.linalg as LA


def V12(x1, x2):
    return e ** 2 / (4.*math.pi*epsilon0)/(abs(x2-x1))/eV


def verphi1(n1, x, R, L):
    kn = math.pi * (n1 + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2. + R / 2.))


def verphi2(n2, x, R, L):
    kn = math.pi * (n2 + 1) / L
    return math.sqrt(2. / L) * math.sin(kn * (x + L / 2. - R / 2.))


def verphi12(n1, n2, x1, x2, R, L):
    return verphi1(n1, x1, R, L) * verphi2(n2, x2, R, L)


def integral_V12(x2, x1, n1, n2, m1, m2, R, L):
    return verphi12(n1, n2, x1, x2, R, L)*V12(x1, x2)*verphi12(m1, m2, x1, x2, R, L)


def Energy0_12(n1, n2, L):
    kn1 = math.pi * (n1 + 1) / L
    kn2 = math.pi * (n2 + 1) / L
    En1 = hbar ** 2 * kn1 ** 2 / (2.0 * me)
    En2 = hbar ** 2 * kn2 ** 2 / (2.0 * me)
    return (En1 + En2) / eV


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
R = 1.2e-9
x_min = -L / 2
x_max = L / 2
n_max = 5
DIM = n_max + 1
N = 4

R_min = L + 0.1e-9
R_max = L + 2.0e-9
NR = 19

x1_min = -R / 2.0 - L / 2.0
x1_max = -R / 2.0 + L / 2.0
x2_min = R / 2.0 - L / 2.0
x2_max = R / 2.0 + L / 2.0


for m1 in range(DIM):
    for m2 in range(DIM):
        for n1 in range(DIM):
            for n2 in range(DIM):
                result = integrate.dblquad(
                    integral_V12,
                    x1_min, x1_max,
                    lambda x: x2_min, lambda x: x2_max,
                    args=([n1, n2, m1, m2, R, L])
                )
                V_real = result[0]
                print(f'{m1},{m2},{n1},{n2}, {V_real}')
