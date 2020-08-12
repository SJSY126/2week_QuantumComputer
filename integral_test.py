import math
import scipy.integrate as integrate

x_min = 0
x_max = 1


def integrand(x):
    return math.sin(math.pi * x)


exact = 2. / math.pi

result = integrate.quad(integrand, x_min, x_max)

print('積分結果:' + str(result[0]))
print('計算誤差:' + str(result[0] - exact) + '(推定誤差:' + str(result[1]) + ')')
