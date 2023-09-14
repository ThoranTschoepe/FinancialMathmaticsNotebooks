import numpy as np

import scipy.stats

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
plt.style.use('ggplot')

def european_call_payoff(S, K):
    return np.maximum(S[-1] - K, 0)


def european_put_payoff(S, K):
    return np.maximum(K - S[-1], 0)


def asian_call_payoff(S, K):
    return np.maximum(np.mean(S, axis=0) - K, 0)


def asian_put_payoff(S, K):
    return np.maximum(K - np.mean(S, axis=0), 0)


def d1(t, x, K, r, sigma, T):
    return (np.log(x / K + 1e-10) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t) + 1e-10)


def d2(t, x, K, r, sigma, T):
    return (np.log(x / K + 1e-10) + (r - 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t) + 1e-10)


def BS_call_price(t, x, K, r, sigma, T):
    return x * scipy.stats.norm.cdf(d1(t, x, K, r, sigma, T)) - K * np.exp(-r * (T - t)) * scipy.stats.norm.cdf(
        d2(t, x, K, r, sigma, T))


def BS_put_price(t, x, K, r, sigma, T):
    return K * np.exp(-r * (T - t)) * scipy.stats.norm.cdf(-d2(t, x, K, r, sigma, T)) - x * scipy.stats.norm.cdf(
        -d1(t, x, K, r, sigma, T))


plot_bound = 0.01


def plot_mc(mc_prices, mc_intervals, label):
    plt.plot(mc_prices[int(len(mc_prices) * plot_bound):], label=label)
    plt.fill_between(np.arange(len(mc_prices[int(len(mc_prices) * plot_bound):])),
                     mc_prices[int(len(mc_prices) * plot_bound):] - mc_intervals[int(len(mc_prices) * plot_bound):],
                     mc_prices[int(len(mc_prices) * plot_bound):] + mc_intervals[int(len(mc_prices) * plot_bound):],
                     alpha=0.2)


def plot_true_price(true_price, len):
    plt.plot([true_price] * int(len * (1 - plot_bound)), label='true price', linewidth=2, color='black')


# here sigma is a matrix
def GBMpathsCov(x0, mu, sigma, T, eta):
    n = len(eta)
    m = len(eta[0])
    dt = T / n
    S = np.zeros((n + 1, m))
    S[0, :] = x0
    L = np.linalg.cholesky(sigma)
    for i in range(m):
        for j in range(1, n + 1):
            S[j, i] = S[j - 1, i] * np.exp(
                (mu[i] - 0.5 * np.sum(L[i, :] ** 4)) * dt + np.sum(L[i, :] ** 2 * np.sqrt(dt) * eta[j - 1, :]))

    return S


# here sigma is a vector
def GBMpaths(x0, mu, sigma, T, eta):
    n = len(eta)
    m = len(eta[0])
    dt = T / n
    S = np.zeros((n + 1, m))
    S[0, :] = x0 * np.ones(m)
    for i in range(n):
        S[i + 1, :] = S[i, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * eta[i, :])

    return S


def basic_mc(t, x, eta, payoff, r, sigma, T):
    assets = GBMpaths(x, r, sigma, T - t, eta)

    payoffs = np.exp(-r*(T-t)) * payoff(assets)

    means = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)

    #we should do this but this is way to slow in python if implemented like this
    #interval_range = 1.96 * ([np.std(payoffs[:n]) for n in range(1, len(payoffs) + 1)] / np.sqrt(np.arange(1, len(payoffs) + 1)))
    #so we do this instead
    interval_range = 1.96 * np.std(payoffs) / np.sqrt(np.arange(1, len(payoffs) + 1))

    return means, interval_range

def mc_importance_sampling(t, x, eta, payoff, is_drift, r = 0, sigma = 1, T = 1):
    if np.isscalar(is_drift):
        is_drift = np.array([is_drift])

    g_h = lambda x_, drift_: np.exp(-np.dot(drift_,x_) + 0.5*np.dot(drift_, drift_))

    assets = GBMpaths(x, r, sigma, T - t, eta + is_drift[:,np.newaxis])

    payoffs = np.exp(-r*(T-t)) * payoff(assets)*g_h(eta + is_drift[:,np.newaxis], is_drift)

    means = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)

    #we should do this but this is way to slow in python if implemented like this
    #interval_range = 1.96 * ([np.std(payoffs[:n]) for n in range(1, len(payoffs) + 1)] / np.sqrt(np.arange(1, len(payoffs) + 1)))
    #so we do this instead
    interval_range = 1.96 * np.std(payoffs) / np.sqrt(np.arange(1, len(payoffs) + 1))

    return means, interval_range