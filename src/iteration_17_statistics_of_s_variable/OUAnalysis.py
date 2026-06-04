from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate


@dataclass
class OUFitResult:

    fitted_tau: float
    fitted_variance: float
    mse_covariance: float
    r_squared: float
    lags: np.ndarray

    c_emp: np.ndarray
    c_theory: np.ndarray

def empirical_autocovariance(
    x: np.ndarray,
    max_lag: int | None = None):
    """
    Returns:

        lags      : integer lag indices
        c_emp     : empirical autocovariance

    Uses FFT-based scipy correlation.
    """

    x = np.asarray(x)

    x0 = x - np.mean(x)

    c = correlate(
        x0,
        x0,
        mode="full",
        method="fft",
    )

    center = len(c) // 2

    c = c[center:]

    # unbiased normalization
    n = len(x)

    c /= np.arange(n, 0, -1)

    if max_lag is not None:
        c = c[:max_lag]

    lags = np.arange(len(c))

    return lags, c

def theoretical_covariance(
    lags: np.ndarray,
    dt: float,
    tau_rise: float,
    r_n: float,
    w_x: float):

    tau = lags * dt

    variance = (
        r_n
        * w_x**2
        * tau_rise
        / 2.0
    )

    c_theory = (
        variance
        * np.exp(-tau / tau_rise)
    )

    return c_theory

def covariance_fit_metrics(
    c_emp: np.ndarray,
    c_theory: np.ndarray):
    residuals = c_emp - c_theory

    mse = np.mean(residuals**2)

    ss_res = np.sum(residuals**2)

    ss_tot = np.sum(
        (c_emp - np.mean(c_emp))**2
    )

    r_squared = 1.0 - ss_res / ss_tot

    return mse, r_squared

def analyze_ou_match(
    x: np.ndarray,
    dt: float,
    tau_rise: float,
    r_n: float,
    w_x: float,
    max_lag: int = 5000):
    """
    Compare empirical covariance
    against OU theory.
    """

    lags, c_emp = empirical_autocovariance(
        x,
        max_lag=max_lag,
    )

    c_theory = theoretical_covariance(
        lags=lags,
        dt=dt,
        tau_rise=tau_rise,
        r_n=r_n,
        w_x=w_x,
    )

    mse, r_squared = covariance_fit_metrics(
        c_emp,
        c_theory,
    )

    return OUFitResult(
        fitted_tau=tau_rise,
        fitted_variance=c_emp[0],
        mse_covariance=mse,
        r_squared=r_squared,
        lags=lags,
        c_emp=c_emp,
        c_theory=c_theory,
    )