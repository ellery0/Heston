import time
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import simps, cumtrapz, romb
# matplotlib inline
import math



# Generalized Marsaglia
np.random.seed(2019)
def GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K, N, dt):
    start_time = time.time()
    Ntime = int(T / dt)
    sqrt_dt = sqrt(dt)
    S = np.ones(N) * s0
    v = np.ones(N) * v0
    vega = (4 * kappa * theta / (beta * beta))
    K1 = dt * (kappa * rho / beta - 0.5) / 2 - rho / beta
    K2 = dt * (kappa * rho / beta - 0.5) / 2 + rho / beta
    K3 = dt * (1 - rho * rho) / 2
    ss = K2 + K3 / 2
    yita = 4 * kappa * exp(-kappa * dt) / (beta * beta) / (1 - exp(-kappa * dt))
    sh = ss * exp(-kappa * dt) / yita




    for i in range(0, Ntime):
        Zs = np.random.randn(1, N)
        lamb = v * yita
        W = np.random.noncentral_chisquare(vega, lamb)
        v2 = W * exp(-kappa * dt) / yita
        K0 = -lamb * sh / (1 - 2 * sh) + 0.5 * vega * log(1 - 2 * sh) - (K1 + K3 / 2) * v
        #K0 = -(kappa * rho * theta) / beta
        S = S * exp(r * dt + K0 + K1 * v + K2 * v2 + np.multiply(sqrt(K3 * (v + v2)), Zs))
        #S = np.multiply((1 + r * dt + np.multiply(sqrt(v), Zs) * sqrt_dt), S)
        v = v2
        #print(v)
    payoff = np.maximum(S - K, 0) * exp(-r * T)
    std = np.std(payoff) / sqrt(N)
    price = np.mean(payoff)
    TIME = (time.time() - start_time)
    Result = collections.namedtuple('Result', ['price', 'std', 'time'])
    out = Result(price, std=std, time=TIME)
    return (out)


if __name__ == '__main__':

    #case1
    T, r, theta, kappa, beta, rho, s0, v0 = 15, 0, 0.04, 0.3, 0.9, -0.5, 100, 0.04
    #case2
    # T, r, theta, kappa, beta, rho, s0, v0 = 5, 0.05, 0.09, 1, 1, -0.3, 100, 0.09

    K = np.array([70,100,140])
    for i in range(0, 3):
        print(GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(), 10000, 1 / 32))
        # convert int32 to int
