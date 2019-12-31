import compute_time
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.integrate import simps, cumtrapz, romb
# matplotlib inline
import math


np.random.seed(2019)
def MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K, N, dt, is_explicit=True, is_weight=False):
    start_time = time.time()
    Ntime = int(T / dt)
    sqrt_dt = sqrt(dt)
    S = np.ones(N) * log(s0)
    #S = np.ones(N)*s0
    v = np.ones(N) * v0
    for i in range(0, Ntime):
        Zv = np.random.randn(1, N)
        Ztemp = np.random.randn(1, N)
        Zs = rho * Zv + sqrt(1 - (rho * rho)) * Ztemp
        # S=S*(1+r*dt+sqrt(v)*Zs*sqrt_dt)
        vreal = np.real(np.maximum(v, 0))
        #print(vreal)

        S = S - 0.5 * vreal * dt + np.multiply(sqrt(vreal), Zs) * sqrt_dt + r * dt
        #S = S * (1 + r * dt + sqrt(vreal) * Zs * sqrt_dt)

        # v=v+kappa*dt*(theta-v)+beta*sqrt(v)*Zv*sqrt_dt
        if is_explicit:
            v = v + kappa * dt * (theta - vreal) + beta * np.multiply(sqrt(vreal),
                                                                      Zv) * sqrt_dt + 1 / 4 * beta * beta * dt * (
                            np.power(Zv, 2) - 1)
        else:
            if is_weight:
                v = (v + kappa * dt * (theta-v/2) + beta * np.multiply(sqrt(vreal), Zv) * sqrt_dt +
                     1 / 4 * beta * beta * dt * (np.power(Zv, 2) - 1)) / (1 + kappa * dt/2)
            else:
                v = (v + kappa * dt * theta + beta * np.multiply(sqrt(vreal),Zv) * sqrt_dt +
                    1 / 4 * beta * beta * dt * (np.power(Zv, 2) - 1))/(1+kappa*dt)


    payoff = np.maximum(exp(S) - K, 0) * exp(-r * T)
    # payoff = np.maximum(S - K, 0) * exp(-r * T)

    std = np.std(payoff) / sqrt(N)
    price = np.mean(payoff)
    TIME = (time.time() - start_time)
    Result = collections.namedtuple('Result', ['price', 'std', 'time'])
    out = Result(price, std=std, time=TIME)
    return (out)


if __name__ == '__main__':

    #case1
    # T, r, theta, kappa, beta, rho, s0, v0 = 15, 0, 0.04, 0.3, 0.9, -0.5, 100, 0.04
    #case2
    T, r, theta, kappa, beta, rho, s0, v0 = 5, 0.05, 0.09, 1, 1, -0.3, 100, 0.09

    K = np.array([70,100,140])
    for i in range(0, 3):
        print(MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(), 10000, 1 / 32,False))
        # convert int32 to int
