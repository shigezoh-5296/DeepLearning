import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def function_tmp1(x0):
    return x0*x0 + 4.0**2.0


def function_tmp2(x1):
    return 3.0**2.0 + x1*x1


def nemerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.xlim(0, 20)
# plt.ylim(0, 6)
# plt.plot(x, y)
# plt.show()

# print(nemerical_diff(function_tmp1, 3.0))
# print(nemerical_diff(function_tmp2, 4.0))


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
