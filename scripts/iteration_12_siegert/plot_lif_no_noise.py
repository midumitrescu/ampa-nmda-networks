import numpy as np
import matplotlib.pyplot as plt

# LIF parameters
taum = 20e-3       # 20 ms
tref = 2e-3        # 2 ms
Vth = -40e-3       # threshold (V)
plt.figure(figsize=(6,4))
for Vreset in np.array([-65, -50, -45, -42, -41]) * 1E-3:

    # mu values slightly above threshold
    mu = np.linspace(Vth + 1e-5, Vth + 5e-3, 200)  # V
    T = taum * np.log((mu - Vreset) / (mu - Vth))
    rate = 1. / (T + tref)  # firing rate in Hz


    plt.plot(mu*1e3, rate, label=r"$V_{reset}$=" + f"{Vreset * 1E3:.0f} mV")

plt.xlabel(r'$\mu$ (mV)')
plt.ylabel('Firing rate (Hz)')
plt.title('Deterministic LIF firing rate near threshold')
plt.legend()
plt.grid(True)
plt.show()