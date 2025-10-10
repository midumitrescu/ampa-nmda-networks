
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcx
from scipy.integrate import quad

from it_2_richardson import Experiment

'''
Vrest, tau, sigmaV, Vth, Vreset, tref
'''
def rate_LIF_whitenoise(experiment: Experiment):
    """
    tau*dV/dt=-V+mu+sigma*xi(t)
    Vreset, and Vth
    """
    Vrest = None
    Vreset = None
    Vth = None

    mu = (Vrest - Vreset) / (Vth - Vreset)
    s = np.sqrt(2) * sigmaV / (Vth - Vreset)
    a1 = (mu - 1) / s
    a2 = mu / s
    T, err = quad(func=erfcx, a=a1, b=a2)
    T = T * np.sqrt(np.pi)
    return 1. / (T * tau + tref)


L = 1001  # #datapoints
mu = np.linspace(0, 30, L)
sigmaV = [0.5, 1., 2., 4., 6.]
rate = np.zeros((len(sigmaV), L))

taum = 0.02  # seconds
Vth = 15.0  # mV
Vreset = 0.  # mV
tref = 0.002  # absolute refractory period in s

for i in range(len(sigmaV)):
    print(sigmaV[i])
    for j in range(L):
        rate[i, j] = rate_LIF_whitenoise(mu[j], taum, sigmaV[i], Vth, Vreset, tref)

# firing rate for sigma=0 (no noise)
rate_determ = np.zeros(L)
for j in range(L):
    if mu[j] > Vth:
        T = taum * np.log((mu[j] - Vreset) / (mu[j] - Vth))
        rate_determ[j] = 1. / (T + tref)

plt.figure(1)
plt.clf()
plt.plot(mu, rate_determ,ls='--', color='k', label=r'$\sigma_V=0$mV')
for i in range(len(sigmaV)):
    plt.plot(mu,rate[i],label=r'$\sigma_V=%g$mV'%(sigmaV[i],))
plt.xlabel(r'input $\mu$ [mV]')
plt.ylabel('firing rate [Hz]')
plt.legend(loc=0)

plt.show()