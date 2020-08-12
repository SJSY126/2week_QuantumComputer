import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 物理定数
h = Planck
hbar = hbar
me = m_e
eV = electron_volt

E1 = 0.25 * eV
E2 = 1.0 * eV
E3 = 4.0 * eV


def wavenumber(E, m):
    return np.sqrt(2.0 * m * E / (hbar ** 2))


k1 = wavenumber(E1, me)
k2 = wavenumber(E2, me)
k3 = wavenumber(E3, me)

omega1 = E1 / hbar
omega2 = E2 / hbar
omega3 = E3 / hbar

dt = 1 * 10 ** -16
dx = 1 * 10 ** -9
XN = 400
TN = 1000
x_min = -2.0
x_max = 2.0

imgs = []

for tn in range(TN):
    t = tn * dt
    xl = []
    psi1l = []
    psi2l = []
    psi3l = []
    for ix in range(XN):
        x = (x_min + (x_max - x_min) * ix / XN) * dx
        psi1 = math.cos(k1 * x - omega1 * t)
        psi2 = math.cos(k2 * x - omega2 * t)
        psi3 = math.cos(k3 * x - omega3 * t)

        xl = np.append(xl, x / dx)
        psi1l = np.append(psi1l, psi1)
        psi2l = np.append(psi2l, psi2)
        psi3l = np.append(psi3l, psi3)

    img = plt.plot(xl, psi1l, color='red', linewidth=3, label='E_1')
    img += plt.plot(xl, psi2l, color='green', linewidth=3, label='E_2')
    img += plt.plot(xl, psi3l, color='blue', linewidth=3, label='E_3')

    imgs.append(img)
plt.title('Plane wave')
plt.xlabel('Position')
plt.ylabel('Real prat of Wave fanction')
ani = animation.ArtistAnimation(fig, imgs, interval=10)
ani.save('output.html', writer=animation.HTMLWriter())
plt.show()
