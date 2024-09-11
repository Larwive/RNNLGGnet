from collections.abc import Callable
from scipy.special import eval_hermite
import mne

import numpy as np

e, pi = np.e, np.pi
N = 10  # Time ticks
K = 0  # ?
M = 4000  # ?
Q = 30  # ?
J = 3  # ?
nu = 5  # ?

freqs = np.linspace(0, 20, M)


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


def amp_sigma(n: int, CFTf):
    sigmask = np.where(12 <= freqs <= 15)
    return (freqs[1] - freqs[0]) * np.sum(CFTf[n, sigmask])


def power_band(n, min_freq, max_freq, CFTf):
    assert max_freq > min_freq
    bandmask = np.where(min_freq <= freqs <= max_freq)
    return (freqs[1] - freqs[0]) * np.sum(abs(CFTf[n, bandmask]) ** 2) / (max_freq - min_freq)


def p_sigma(n: int, CFTf):
    return power_band(n, c_star - .1, c_star + .1, CFTf)


def p_delta(n: int, CFTf):
    return power_band(n, .5, 4, CFTf)


def p_theta(n: int, CFTf):
    return power_band(n, 4, 8, CFTf)


def p_alpha(n: int, CFTf):
    return power_band(n, 8, 12, CFTf)


def norm_p_sigma(n: int, CFTf):
    return p_sigma(n, CFTf) / (p_delta(n, CFTf) + p_theta(n, CFTf) + p_alpha(n, CFTf) + p_sigma(n, CFTf))


def regularization_term(c, n):
    reg_term = 0
    for l in range(1, n):
        delta_c = c[l] - c[l - 1]
        reg_term += abs(delta_c) ** 2
    return reg_term


def compute_R(l, c_l, CFTf):
    numer = abs(CFTf[l, c_l])
    denom = np.sum(abs(CFTf))
    return np.log(numer / denom)


def objective_function(c, CFTf, lambda_penalty):
    R_sum = sum([compute_R(l, c[l], CFTf) for l in range(M)])
    reg_term = regularization_term(c, M)
    return R_sum - lambda_penalty * reg_term


def possible_candidates(c_l, step_size=1):
    candidates = []
    if c_l - step_size >= 0:
        candidates.append(c_l - step_size)
    candidates.append(c_l)
    if c_l + step_size <= M:
        candidates.append(c_l + step_size)
    return candidates


def optimize_c(c, CFTf, lambda_penalty, min_freq=10, max_freq=15, max_iterations: int = 10):
    for iteration in range(max_iterations):
        freqmask = np.where(min_freq <= freqs <= max_freq)
        for l in freqs[freqmask]:
            best_c_l = c[l]
            best_obj_value = objective_function(c, CFTf, lambda_penalty)

            for candidate_c_l in possible_candidates(c[l]):
                c[l] = candidate_c_l
                obj_value = objective_function(c, CFTf, lambda_penalty)
                if obj_value > best_obj_value:
                    best_c_l = candidate_c_l
                    best_obj_value = obj_value

            c[l] = best_c_l
        # c = smooth_curve(c)  # Optional
    return c


def get_cons_int(arr: np.ndarray[int]):
    indexes = []
    start = 0
    for i in range(len(arr) - 1):
        if arr[i] + 1 != arr[i + 1]:
            if start == i:
                indexes.append((np.array([start]),))
            else:
                indexes.append((np.array([start, i]),))
            start = i + 1
    if start < len(arr) - 1:
        indexes.append((np.array([start, len(arr) - 1]),))
    elif start == len(arr) - 1:
        indexes.append((np.array([start]),))
    return indexes


if __name__ == '__main__':
    path = ''
    raw = mne.io.read_raw_fif(path, preload=True)
    K = raw.info['sfreq']
    f = f_gen(raw)

    CFTf = np.zeros((raw.n_times, M))
    for i in range(raw.n_times):
        for m in range(M):
            CFTf[i, m] = CFT(i, m, f)
    c = [0 for _ in range(N)]
    lambda_penalty = ...
    c_star = optimize_c(c, CFTf, lambda_penalty)

    amps_sigma = np.array([amp_sigma(i, CFTf) for i in range(N)])
    amps_avg, amps_std = np.average(amps_sigma), np.std(amps_sigma)

    norm_ps_sigma = np.array([norm_p_sigma(i, CFTf) for i in range(N)])

    delta = 1
    epsilon = .2
    T1 = np.where(amps_sigma > amps_avg + delta * amps_std)
    T2 = np.where(norm_ps_sigma >= epsilon)

    I = np.intersect1d(T1, T2, assume_unique=True)
