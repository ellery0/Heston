import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exact
import euler
import qe
import marsaglia
import milstein
from scipy import *
analytical=exact.call_price_exact
EulerHeston=euler.EulerHeston
MilsteinHeston=milstein.MilsteinHeston
QEHeston=qe.QEHeston
GMHeston=marsaglia.GMHeston
# case1
# T, r, theta, kappa, beta, rho, s0, v0 = 15, 0, 0.04, 0.3, 0.9, -0.5, 100, 0.04
# case2
T, r, theta, kappa, beta, rho, s0, v0 = 5, 0.05, 0.09, 1, 1, -0.3, 100, 0.09
K = np.arange(60, 180, 40)
N=100000
dt_range=np.arange(1, 1/32, -1/32)
result=np.zeros((5,len(dt_range)))
result_std=np.zeros((5,len(dt_range)))
real=analytical(kappa, theta, beta, rho, v0, r, T, s0, K)
for i in range(0,len(dt_range)):
    dt=dt_range[i].item()
    eul=EulerHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    mil = MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    mil_in=MilsteinHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt,False)
    QE=QEHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    GM=GMHeston(kappa, theta, beta, rho, v0, r, T, s0, K,N,dt)
    result[:,i]=[eul[0]-real,mil[0]-real,mil_in[0]-real,QE[0]-real,GM[0]-real]
    result_std[:,i]=[eul[1],mil[1],mil_in[1],QE[1],GM[1]]
    #convert int32 to int
s = pd.Series(['Eul','Mil','Mil-in','QE','GM'])
df = pd.DataFrame(result, index=s, columns=dt_range)
x=-log(dt_range)
for i in range(0,5):
    plt.plot(x,abs(result[i,:]),label=s[i])
plt.xlabel("-ln(dt)")
plt.ylabel("Error")
plt.legend()
plt.show()
# fig2 = plt.figure()
# for i in range(0,5):
#     plt.plot(x,abs(result_std[i,:]),label=s[i])
# plt.xlabel("-ln(dt)")
# plt.ylabel("Std")
# plt.legend()
# plt.show()