import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmath


def wavenumber(E, m):
    return np.sqrt(2.0 * m * E / (hbar ** 2))


def verphi(n, x):
    kn = math.pi * (n + 1) / L
    return math.sqrt(2 / L) * math.sin(kn * (x + L / 2))


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
NX = 500
dx = 10 ** -9
ts = -50
te = 50

T0 = 2 * math.pi * hbar / Energy(0)
dt = T0 / (te - ts + 1)

imgs = []

for tn in range(ts, te):
    t = dt * tn
    xl = []
    nl = [[] for _ in range(n_max + 1)]

    for ix in range(NX):
        x = x_min + (x_max - x_min) * ix / NX
        xl.append(x / dx)
        for n in range(n_max + 1):
            nl[n].append(phi(n, x, t).real / math.sqrt(2 / L) * 0.5 + n)

    img = plt.plot(xl, nl[0], 'blue')
    img += plt.plot(xl, nl[1], 'green')
    img += plt.plot(xl, nl[2], 'red')
    img += plt.plot(xl, nl[3], 'yellow')
    img += plt.plot(xl, nl[4], 'black')
    img += plt.plot(xl, nl[5], 'cyan')

    imgs.append(img)

plt.title('Quantum Well', fontsize=16)
plt.xlabel('Position(nm)', fontsize=16)
plt.ylabel('Eigenfunction', fontsize=16)
plt.xlim([-0.5, 0.5])
ani = animation.ArtistAnimation(fig, imgs, interval=50)
plt.show()
