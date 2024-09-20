# Directly run the code.

from collections.abc import Callable, Generator
from scipy.special import eval_hermite
import mne
import numpy as np
from functools import lru_cache
from tqdm import tqdm

e, pi = np.e, np.pi

M = 4000  # The resolution of frequency axis
Q = 30  # The number of points sampled for multitaper in ConceFT
J = 3  # The number of Hermite windows for ConceFT
nu = 5  # ?
f = ...  # Defined at the bottom
chunk_size = 100


def hermite_window(degree: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # N = 2 * K + 1
    """
    Compute the hermite window and its derivative of a certain degree.
    :param degree: The degree of the hermite window.
    :param N: The resolution of x abscissa.
    :return: Hermite window, its derivative and the abscissa.
    """
    x = np.linspace(-10, 10, N)
    H = eval_hermite(degree, x)

    exp_term = np.exp(-0.5 * x ** 2)
    H_derivative = eval_hermite(degree - 1, x) if degree > 0 else np.zeros_like(x)
    return np.exp(-0.5 * x ** 2) * H, exp_term * H_derivative - x * H * exp_term, x


def g_gen2(i, der) -> Callable[[int, int], complex]:
    """
    Returns the `g` function described in the article for a certain i and hermitte window.
    :param i: The i.
    :param der: Whether to use the derivative of the hermite window.
    :return: The `g` function.
    """

    @lru_cache(maxsize=None)
    def g(x_start: int, x_end: int) -> complex:
        return np.sum(zJ[i, :].reshape((-1, 1)) * H_S[der][:, x_start:x_end - 1], axis=0)

    return g


def g_calc(g: Callable) -> np.ndarray:
    return np.array(g(0, 2 * K + 2)).reshape((-1, 1))


def V(i: int, der: int, m_start: int = 0, m_end: int = M) -> np.ndarray:
    k_range = np.arange(2 * K + 1).reshape(-1, 1)
    m_range = np.arange(m_start, m_end).reshape(1, -1)
    k_m = k_range.dot(m_range)
    pre_exp = np.exp(-pi * k_m * 2j / M)
    g_k = g_calc(g_gen2(i, der))

    # return np.sum(pre_exp * f * g_k, axis=1) # (257, 4000) * (3491456, 257, 1) * (257, 1) -> (3491456, 257, 4000)

    # Memory sparer alternative
    final = np.zeros((f.shape[0], pre_exp.shape[1]), dtype=np.complex128)

    for start in tqdm(range(0, f.shape[0], chunk_size)):
        end = min(start + chunk_size, f.shape[0])
        final[start:end] = np.sum(pre_exp * f[start:end] * g_k, axis=1)
    return final


def Omegas() -> Generator[np.ndarray]:
    # return np.fromiter((-np.imag(N * V(i, 1) / (2 * pi * V(i, 0))) for i in i_range), dtype=np.complex128,
    #                   count=Q)

    # Memory sparer alternative
    for i in np.arange(Q):
        yield -np.imag(N * V(i, 1) / (2 * pi * V(i, 0)))


def S() -> Generator[np.ndarray]:
    # return np.fromiter((V(i, 0, int(Om - nu), int(Om + nu) + 1) for i, Om in enumerate(omegas)), dtype=np.complex128,
    #                   count=Q)

    # Memory sparer alternative
    for i, Om in enumerate(Omegas()):
        yield np.abs(V(i, 0, int(Om - nu), int(Om + nu) + 1))


def CFT():
    # return np.mean(np.linalg.norm(S()))

    # Memory sparer alternative
    tot = 0
    count = 0
    for norm in S():
        tot += norm
        count += 1
    return tot / count


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
        # freqmask = (min_freq <= freqs) & (freqs <= max_freq)
        for l in range(len(c)):  # freqs[freqmask]:
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
    """
    Compute the indexes of consecutive int in an ordered int iterable.
    :param arr: The iterable.
    :return: List of tuples containing hte indexes.
    """
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
    freqs = np.linspace(0, 20, M)

    path = 'stw/102-10-21.fif'
    raw = mne.io.read_raw_fif(path, preload=True)
    K = int(raw.info['sfreq'])  # The length of the Hermite windows is 2K + 1
    N = raw.n_times

    channel = 0
    ext_f = np.array([0] * K + list(raw.get_data()[channel, :]) + [0] * K)

    f = np.array([ext_f[i: i + 2 * K + 1].reshape((-1, 1)) for i in range(N - K)])

    zJ = np.exp(1j * np.random.random((Q, J)) * 2 * np.pi)

    h1, h1p, x = hermite_window(1, 2 * K + 1)
    h2, h2p, _ = hermite_window(2, 2 * K + 1)
    h3, h3p, _ = hermite_window(3, 2 * K + 1)

    h_s = np.array([h1, h2, h3])
    hp_s = np.array([h1p, h2p, h3p])

    H_S = [h_s, hp_s]
    print("Computing ConceFT...")

    CFTf = CFT()
    c = np.full((N,), 0)
    lambda_penalty = .1  # ?

    print("Computing C*...")
    c_star = optimize_c(c, CFTf, lambda_penalty)

    print("Computing sigma amplitudes...")
    amps_sigma = np.array([amp_sigma(i, CFTf) for i in tqdm(range(N))])
    amps_avg, amps_std = np.average(amps_sigma), np.std(amps_sigma)

    norm_ps_sigma = np.array([norm_p_sigma(i, CFTf) for i in range(N)])

    delta = 1  # The parameter for the hard threshold of the sigma band
    epsilon = .2  # The threshold of the normalized sigma band power amplitude
    T1 = np.where(amps_sigma > amps_avg + delta * amps_std)
    T2 = np.where(norm_ps_sigma >= epsilon)

    I = np.intersect1d(T1, T2, assume_unique=True)

    print(I)
