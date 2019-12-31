import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exact
import euler
import qe
import marsaglia
import milstein
from scipy.optimize import fmin
analytical=exact.call_price_exact
EulerHeston=euler.EulerHeston
MilsteinHeston=milstein.MilsteinHeston
QEHeston=qe.QEHeston
GMHeston=marsaglia.GMHeston
# case1
# T, r, theta, kappa, beta, rho, s0, v0 = 15, 0, 0.04, 0.3, 0.9, -0.5, 100, 0.04
# case2
T, r, theta, kappa, beta, rho, s0, v0 = 5, 0.05, 0.09, 1, 1, -0.3, 100, 0.09
K = np.array([70,100,140])
result=np.zeros((6,3))
result_std=np.zeros((5,3))
result_time=np.zeros((5,3))
N=100000
dt=1/32
for i in range(0,3):
    real=analytical(kappa, theta, beta, rho, v0, r, T, s0, K[i].item())
    eul=EulerHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    mil=MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    mil_in = MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(), N, dt,False)
    QE=QEHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    GM=GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K[i].item(),N,dt)
    result[:,i]=[real,eul[0],mil[0],mil_in[0],QE[0],GM[0]]
    result_std[:,i]=[eul[1],mil[1],mil_in[1],QE[1],GM[1]]
    result_time[:,i]=[eul[2],mil[2],mil_in[2],QE[2],GM[2]]

s = pd.Series(['Real','Eul','Mil','Mil-in','QE','GM'])
df = pd.DataFrame(result, index=s, columns=K)
print(df)
# df2 = pd.DataFrame(result_std, index=s.iloc[1:6], columns=K)
# print(df2)
df3 = pd.DataFrame(result_time, index=s.iloc[1:6], columns=K)
print(df3)