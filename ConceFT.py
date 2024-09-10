from collections.abc import Callable
from scipy.special import eval_hermite
import mne

import numpy as np

e, pi = np.e, np.pi
N = 10  # Time ticks
K = 1  # ?
M = 1  # ?
Q = 1  # ?
J = 3  # ?
nu = 5  # ?


def f_gen(raw: mne.io.BaseRaw):
    def f(i: int, c: int):
        return raw.get_data[c, i]

    return f


def hermite_window(degree: int, N: int = 2 * K + 1):
    x = np.linspace(-10, 10, N)
    H = eval_hermite(degree, x)

    exp_term = np.exp(-0.5 * x ** 2)
    H_derivative = eval_hermite(degree - 1, x) if degree > 0 else np.zeros_like(x)
    return np.exp(-0.5 * x ** 2) * H, exp_term * H_derivative - x * H * exp_term, x


h1, h1p, x = hermite_window(1)
h2, h2p, _ = hermite_window(2)
h3, h3p, _ = hermite_window(3)

h_s = [h1, h2, h3]
hp_s = [h1p, h2p, h3p]

zJ = [[e ** (np.random.random() * pi * 2j) for __ in range(J)] for _ in range(Q)]


def g_gen(i, hw_s) -> Callable[[int], complex]:
    def g(x: int) -> complex:
        s = 0
        for j in range(1, J + 1):
            s += zJ[i] * hw_s[j][x]
        return s

    return g


def V_gen(g: Callable[[int], complex], f: Callable[[int], float]) -> Callable[[int, int], complex]:
    def V(n: int, m: int) -> complex:
        s = 0
        for k in range(1, 2 * K + 2):
            s += f(n + k - K - 1) * g(k) * e ** (-pi * (k - 1) * m * 2j) / M
        return s

    return V


def Omega(n: int, m: int, i: int, f):
    Vi = V_gen(g_gen(i, h_s), f)
    Vip = V_gen(g_gen(i, hp_s), f)
    return -np.imag(N * Vip(n, m) / (2 * pi * Vi(n, m)))


def S_i(n: int, m: int, i: int, f: Callable):
    s = 0
    Vi = V_gen(g_gen(i, h_s), f)
    for l in range(int(Omega(n, m, i, f) - nu), int(Omega(n, m, i, f) + nu) + 1):
        s += Vi(n, l)
    return s


def CFT(n: int, m: int, f: Callable):
    s = 0
    for i in range(Q):
        s += np.linalg.norm(S_i(n, m, i, f))
    return s / Q
