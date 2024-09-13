from collections.abc import Callable
from scipy.special import eval_hermite
import mne
import numpy as np
from functools import lru_cache
from tqdm import tqdm

e, pi = np.e, np.pi


def f_gen(raw_data: mne.io.BaseRaw):
    def f(start: int, end: int, c: int):
        return np.array(
            [0] * (max(-start, 0)) + list(raw_data.get_data()[c, max(start, 0):end]) + [0] * (
                max(end - raw_data.n_times, 0)))

    return f


def hermite_window(degree: int, N: int):  # N = 2 * K + 1
    x = np.linspace(-10, 10, N)
    H = eval_hermite(degree, x)

    exp_term = np.exp(-0.5 * x ** 2)
    H_derivative = eval_hermite(degree - 1, x) if degree > 0 else np.zeros_like(x)
    return np.exp(-0.5 * x ** 2) * H, exp_term * H_derivative - x * H * exp_term, x


@lru_cache(maxsize=None)
def g_gen(i, der) -> Callable[[int], complex]:
    @lru_cache(maxsize=None)
    def g(x: int) -> complex:
        return np.sum(zJ[i, :] * H_S[der][:, x - 1])

    return g


@lru_cache(maxsize=None)
def g_gen2(i, der) -> Callable[[int, int], complex]:
    @lru_cache(maxsize=None)
    def g(x_start: int, x_end: int) -> complex:
        return np.sum(zJ[i, :].reshape((-1, 1)) * H_S[der][:, x_start:x_end - 1], axis=0)

    return g


@lru_cache(maxsize=None)
def V_gen(i: int, der: int, f: Callable[[int, int, int], np.ndarray[float]]) -> Callable[
    [int, int], complex]:
    g = g_gen2(i, der)

    @lru_cache(maxsize=None)
    def V(n: int, m: int) -> complex:
        k_range = np.arange(2 * K + 1)
        pre_exp = np.exp(-pi * k_range * m * 2j / M)
        f_data = f(n - K, n + K + 1, 0)
        g_k = np.array(g(0, 2 * K + 2))
        return np.sum(pre_exp * f_data * g_k)

    return V

@lru_cache(maxsize=None)
def V_gen2(i: int, der: int, f: Callable[[int, int, int], np.ndarray[float]]) -> Callable[
    [int, int, int], complex]:
    g = g_gen2(i, der)

    @lru_cache(maxsize=None)
    def V(n: int, m_start: int, m_end: int) -> complex:
        k_range = np.arange(2 * K + 1).reshape(-1, 1)
        m_range = np.arange(m_start, m_end).reshape(1, -1)
        k_m = k_range.dot(m_range)
        pre_exp = np.exp(-pi * k_range * k_m * 2j / M)
        f_data = f(n - K, n + K + 1, 0)
        g_k = np.array(g(0, 2 * K + 2)).reshape((-1, 1))
        return np.sum(pre_exp * f_data.reshape((-1, 1)) * g_k)

    return V


@lru_cache(maxsize=None)
def Omega(n: int, m: int, i: int, f: Callable[[int, int, int], np.ndarray[float]]):
    Vi = V_gen(i, 0, f)
    Vip = V_gen(i, 1, f)
    return -np.imag(N * Vip(n, m) / (2 * pi * Vi(n, m)))


@lru_cache(maxsize=None)
def S_i(n: int, m: int, i: int, f: Callable[[int, int, int], np.ndarray[float]]):
    Vi = V_gen2(i, 0, f)
    return Vi(n, int(Omega(n, m, i, f) - nu), int(Omega(n, m, i, f) + nu) + 1)


def CFT(n: int, m: int, f: Callable[[int, int, int], np.ndarray[float]]):
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


def regularization_term(c: np.ndarray, n):
    return np.sum(np.diff(c)[:n - 1] ** 2)


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
    M = 4000  # ?
    Q = 30  # ?
    J = 3  # ?
    nu = 5  # ?
    freqs = np.linspace(0, 20, M)

    path = 'stw/102-10-21.fif'
    raw = mne.io.read_raw_fif(path, preload=True)
    K = int(raw.info['sfreq'])
    N = raw.n_times
    f = f_gen(raw)

    zJ = np.exp(1j * np.random.random((Q, J)) * 2 * np.pi)

    h1, h1p, x = hermite_window(1, 2 * K + 1)
    h2, h2p, _ = hermite_window(2, 2 * K + 1)
    h3, h3p, _ = hermite_window(3, 2 * K + 1)

    h_s = np.array([h1, h2, h3])
    hp_s = np.array([h1p, h2p, h3p])

    H_S = [h_s, hp_s]
    CFTf = np.zeros((N, M))
    print("Computing ConceFT...")
    for i in range(N):
        for m in range(M):
            CFTf[i, m] = CFT(i, m, f)
    c = np.full((N,), 0)
    lambda_penalty = .1 # ?

    print("Computing C*...")
    c_star = optimize_c(c, CFTf, lambda_penalty)

    print("Computing sigma amplitudes...")
    amps_sigma = np.array([amp_sigma(i, CFTf) for i in tqdm(range(N))])
    amps_avg, amps_std = np.average(amps_sigma), np.std(amps_sigma)

    norm_ps_sigma = np.array([norm_p_sigma(i, CFTf) for i in range(N)])

    delta = 1
    epsilon = .2
    T1 = np.where(amps_sigma > amps_avg + delta * amps_std)
    T2 = np.where(norm_ps_sigma >= epsilon)

    I = np.intersect1d(T1, T2, assume_unique=True)

    print(I)
