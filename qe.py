import time
import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
import scipy.stats as stats
from scipy.integrate import simps, cumtrapz, romb
# matplotlib inline
import math




np.random.seed(2019)
def QEHeston(kappa, theta, beta, rho, v0, r, T, s0, K, N, dt):
    start_time = time.time()
    Ntime = int(T / dt)
    sqrt_dt = sqrt(dt)
    S = np.ones(N) * s0
    v = np.ones(N) * v0
    v2 = np.ones(N) * v0
    M = np.zeros(N)
    K0 = -(kappa*rho*theta)*dt/beta
    K1 = dt * (kappa * rho / beta - 0.5) / 2 - rho / beta
    K2 = dt * (kappa * rho / beta - 0.5) / 2 + rho / beta
    K3 = dt * (1 - rho * rho) / 2
    ind = np.array([i for i in range(N)])



    phi_c = np.zeros(N)+1.5


    for i in range(0, Ntime):

        E = np.exp(-(kappa * dt))
        m = theta + (v - theta) * E
        s2 = v * np.power(beta,2) * E * (1 - E) / kappa + theta * np.power(beta,2) * np.power(1 - E , 2) / (2 * kappa)
        #print(np.power(m,2)[[0,3]],s2[[0,3]])
        phi = s2 / np.power(m,2)

        cond = phi <= phi_c
        con = np.where(cond)

        ncon = np.where(~cond)
        #print(phi)

        Uv = np.random.uniform(0, 1, N)
        Zv = stats.norm.ppf(Uv)
        # Zv = np.random.normal(0,1,N)
        Zs = np.random.normal(0, 1, N)

        b = sqrt(2 / phi[con] - 1 + sqrt(2 / phi[con] * (2 / phi[con] - 1)))
        a = m[con] / (1 + np.power(b , 2))
        v2[con] = a * np.power(b + Zv[con], 2)
        A = K2+K3/2
        M[con] = np.exp(A*np.power(b,2)*a/1-2*A*a)/(sqrt(1-2*A*a))
        #print('b',b,'m',m[con],'a',a,'z',Zv[con],'v2con',v2[con])

        p = (phi[ncon] - 1) / (phi[ncon] + 1)
        sigma = (1 - p) / m[ncon]
        M[ncon] = p+(1-p)*sigma/(sigma-A)

        u_con = np.where(Uv[ncon] < p)
        u_ncon = np.where(Uv[ncon] >= p)

        ind_n_u = ind[ncon][u_con]
        ind_n_n = ind[ncon][u_ncon]

        #v2[ncon][u_con] = 0
        v2[ind_n_u]=0


        if len(u_ncon[0])>0:
            #v2[ncon][u_ncon] = 1 / sigma[u_ncon] * np.log((1 - p[u_ncon]) / (1 - Uv[ncon][u_ncon]))
            v2[ind_n_n] = 1 / sigma[u_ncon] * np.log((1 - p[u_ncon]) / (1 - Uv[ncon][u_ncon]))
            # assert v2[ncon][u_ncon].all() == np.array(1 / sigma[u_ncon] * np.log((1 - p[u_ncon]) / (1 - Uv[ncon][u_ncon]))).all()
            # print(v2[ncon][u_ncon].all() == np.array(1 / sigma[u_ncon] * np.log((1 - p[u_ncon]) / (1 - Uv[ncon][u_ncon]))).all())
            # print('cal',1 / sigma[u_ncon] * np.log((1 - p[u_ncon]) / (1 - Uv[ncon][u_ncon])))
        K0_star = -np.log(M)-(K1+K3/2)*v
        #print('k0',K0,'k0s',K0_star)
        #K0 = K0_star
        S = S * exp(r * dt + K0 + K1 * v + K2 * v2 + np.multiply(sqrt(K3 * (v + v2)), Zs))
        #S = S - 0.5 * v * dt + np.multiply(sqrt(v), Zs) * sqrt_dt + r * dt
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
        print(QEHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(), 10000, 1 / 32))
        # convert int32 to int


















