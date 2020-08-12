import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import matplotlib
print(matplotlib.matplotlib_fname())

fig = plt.figure(figsize=(16, 9))

x_min = -1.
x_max = 1.

ims = []
N = 100
AN = 30

for a in range(AN):
    phi = 2.0 * math.pi * a / AN
    xl = []
    yl = []
    for i in range(N + 1):
        x = x_min + (x_max - x_min) * i / N
        y = math.sin(math.pi * x + phi)
        xl.append(x)
        yl.append(y)

    img = plt.plot(xl, yl, color='blue', linewidth=3, linestyle='solid')
    ims.append(img)

plt.title('sin function')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.xlim([-1., 1])
plt.ylim([-1., 1])

ani = animation.ArtistAnimation(fig, ims, interval=50)
ani.save('output.html', writer=animation.HTMLWriter())

plt.show()
