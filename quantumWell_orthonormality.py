import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt
import matplotlib.pyplot as plt
import cmath
import scipy.integrate as integrate


def wavenumber(E, m):
    return np.sqrt(2.0 * m * E / (hbar ** 2))


def verphi(n, x):
    kn = math.pi * (n + 1) / L
    return math.sqrt(2 / L) * math.sin(kn * (x + L / 2.))


def Energy(n):
    kn = math.pi * (n + 1) / L
    return (hbar * kn) ** 2 / (2 * me)


def phi(n, x, t):
    kn = math.pi * (n + 1) / L
    omega = hbar * kn**2 / (2 * me)
    return verphi(n, x) * cmath.exp(-I * omega * t)


fig = plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 物理定数
h = Planck
hbar = hbar
me = m_e
eV = electron_volt
I = 0+1j

L = 10 ** -9
x_min = -L / 2
x_max = L / 2
n_max = 5


def integral_orthonomality(x, n1, n2):
    return verphi(n1, x) * verphi(n2, x)


for n1 in range(n_max + 1):
    for n2 in range(n_max + 1):
        result = integrate.quad(
            integral_orthonomality,
            x_min, x_max,
            args=(n1, n2)
        )
        print('('+str(n1)+','+str(n2)+')' +
              str(result[0] if result[0] > 1e-15 else 0))
