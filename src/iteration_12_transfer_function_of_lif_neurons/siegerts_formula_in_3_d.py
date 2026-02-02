import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcx
from scipy.integrate import quad

def grad_mag(x, sigma, rate):
    dR_dsigma, dR_dmu = np.gradient(
        rate,
        sigma, x,
        edge_order=2
    )

    return np.sqrt(dR_dmu**2 + dR_dsigma**2)


def rate_LIF_whitenoise(V_mean, tau, sigmaV, Vth, Vreset, tref):
    """
    tau*dV/dt=-V+mu+sigma*xi(t)
    Vreset, and Vth
    """
    # [V Mean, V Reset, V Th, V reset] = mV
    mu = (V_mean - Vreset) / (Vth - Vreset)
    s = np.sqrt(2) * sigmaV / (Vth - Vreset)
    a1 = (mu - 1) / s
    a2 = mu / s

    T, err = quad(func=erfcx, a=a1, b=a2, epsabs=1e-13, epsrel=1e-13)
    T = T * np.sqrt(np.pi)
    return 1. / (T * tau + tref)


def plot_3d_meshgrid():
    # axes
    Lx = 1001
    Ls = 100

    x = np.linspace(7.5, 20, Lx)          # μ
    sigma = np.linspace(0.01, 5.0, Ls)  # σV (avoid 0)

    # mesh
    X, SIGMA = np.meshgrid(x, sigma)    # (Ls, Lx)
    RATE = np.zeros_like(X)

    # parameters
    taum = 0.02
    Vth = 15.0
    Vreset = 0.0
    tref = 0.002

    # evaluate on mesh
    for i in range(SIGMA.shape[0]):
        for j in range(X.shape[1]):
            RATE[i, j] = rate_LIF_whitenoise(
                X[i, j], taum, SIGMA[i, j], Vth, Vreset, tref
            )

    # plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, SIGMA, RATE,
        cmap='viridis',
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel(r'input $\mu$ [mV]')
    ax.set_ylabel(r'noise $\sigma_V$ [mV]')
    ax.set_zlabel('firing rate [Hz]')

    fig.colorbar(surf, label='rate [Hz]', shrink=0.6)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6, 5))
    cs = plt.contour(
        X, SIGMA, RATE,
        levels=15,
        colors='k'
    )
    plt.clabel(cs, inline=True, fontsize=8)

    plt.contourf(
        X, SIGMA, RATE,
        levels=50,
        cmap='viridis'
    )
    plt.colorbar(label='firing rate [Hz]')

    plt.xlabel(r'input $\mu$ [mV]')
    plt.ylabel(r'noise $\sigma_V$ [mV]')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        X, SIGMA, grad_mag(X, SIGMA, RATE),
        shading='auto',
        cmap='magma'
    )
    plt.colorbar(label=r'$|\nabla F|$')
    plt.xlabel(r'input $\mu$ [mV]')
    plt.ylabel(r'noise $\sigma_V$ [mV]')
    plt.tight_layout()
    plt.show()




#plot_3d_meshgrid()

if __name__ == '__main__':
    #plot_3d_meshgrid()
    #plt.show()
    pass