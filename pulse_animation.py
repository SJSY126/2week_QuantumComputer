import math
import numpy as np
from scipy.constants import Planck, hbar, m_e, electron_volt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmath


def wavenumber(E, m):
    return np.sqrt(2.0 * m * E / (hbar ** 2))


fig = plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 物理定数
h = Planck
hbar = hbar
me = m_e
eV = electron_volt

I = 0.0+1.0j

NX = 500
dx = 1.0 * 10 ** -9
x_min = -10 * dx
x_max = 10 * dx
NK = 200
delta_x = 2.0 * 10 ** -9
sigma = 2.0 * math.sqrt(math.log(2)) / delta_x
dk = 20.0 / (delta_x * NK)
E0 = 1.0 * eV
k0 = wavenumber(E0, me)
omega0 = hbar / (2.0 * me) * k0 ** 2

ts = -50
te = 30
dt = 1.0 * 10 ** -16

imgs = []

for tn in range(ts, te):
    t_real = dt * tn
    xl = []
    Psi_real = []
    Psi_imag = []
    Psi_abs = []
    for nx in range(NX):
        x = x_min + (x_max - x_min) * nx / NX
        Psi = 0.0+0.0j
        for kn in range(NK):
            k = k0 + dk * (kn - NK / 2)
            omega = hbar / (2.0 * me) * k ** 2
            Psi += cmath.exp(I * (k * x - omega * t_real)) * \
                cmath.exp(-((k - k0) / (2.0 * sigma)) ** 2)

        Psi = Psi * dk * dx / 10

        # print(f'x/dx:{x/dx}\nPsi_real:{Psi.real}\nPsi_imag:{Psi.imag}\nPsi_abs:{abs(Psi)}')

        xl.append(x / dx)
        Psi_real.append(Psi.real)
        Psi_imag.append(Psi.imag)
        Psi_abs.append(abs(Psi))

    img = plt.plot(xl, Psi_real, 'red')
    img += plt.plot(xl, Psi_imag, 'green')
    img += plt.plot(xl, Psi_abs, 'blue')
    time = plt.text(0.9, 1.03, 't:{:.2e}'.format(
        t_real), transform=plt.gca().transAxes, ha='center', va='center')
    img.append(time)
    imgs.append(img)

kl = []
a_kl = []

for kn in range(NK):
    k = k0 + dk * (kn - NK / 2)
    a_k = math.exp(-((k - k0) / (2.0 * sigma)) ** 2)
    kl.append(k/dk)
    a_kl.append(a_k)

plt.title('Gaussian wave packet(Spatial distribution)')
plt.xlabel('Position[nm]', fontsize=16)
plt.ylabel('Probability amplitude', fontsize=16)
plt.xlim([-10, 10])
plt.ylim([-0.3, 0.3])
ani = animation.ArtistAnimation(fig, imgs, interval=50)
ani.save('output.html', writer=animation.HTMLWriter())

fig2 = plt.figure(figsize=(5, 5))
plt.title('Gaussian wave packet(Wave number distribution)')
plt.xlabel('Wave number', fontsize=16)
plt.ylabel('distribution', fontsize=16)
plt.plot(kl, a_kl)

plt.tight_layout()
plt.show()
