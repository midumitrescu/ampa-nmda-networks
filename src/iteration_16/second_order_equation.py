import numpy as np
from brian2 import Quantity

from iteration_16.model import ConductanceDiffusionSimulationConfig

def compute_A_B_C(conductance_diffusion_config: ConductanceDiffusionSimulationConfig, mu_v: Quantity, sigma_v: Quantity, verbose=True):
    gL = conductance_diffusion_config.g_L

    Ee = conductance_diffusion_config.e_ampa
    Ei = conductance_diffusion_config.e_gaba
    EL = conductance_diffusion_config.e_L

    # --- shifted potentials ---
    bar_EL = EL - mu_v
    bar_Ee = Ee - mu_v
    bar_Ei = Ei - mu_v

    if verbose:
        print("\n=== SHIFTED POTENTIALS ===")
        print("bar_E_L =", bar_EL)
        print("bar_E_e =", bar_Ee)
        print("bar_E_i =", bar_Ei)

    # --- useful factors ---
    a = (1 - bar_EL / bar_Ei)
    b = (1 - bar_Ee / bar_Ei)

    if verbose:
        print("\n=== NORMALIZED FACTORS ===")
        print("a = (1 - bar_E_L / bar_E_i) =", a)
        print("b = (1 - bar_E_e / bar_E_i) =", b)

    # --- quadratic coefficients A g^2 + B g + C = 0 ---

    A = (
            4 * sigma_v ** 2 * b ** 2
            - 2 * bar_Ee ** 2
    )

    B = (
            8 * sigma_v ** 2 * gL * a * b
            - 2 * gL * bar_EL * bar_Ee
    )

    C = (
            4 * sigma_v ** 2 * gL ** 2 * a ** 2
            - gL ** 2 * bar_EL ** 2
    )
    return A, B, C

def solve_ge0_from_moments(conductance_diffusion_config: ConductanceDiffusionSimulationConfig, mu_v: Quantity, sigma_v: Quantity, verbose=True):
    """
    Solves for g_{e,0} using the quadratic approximation derived from:
        4 σ_v^2 g0^2 = g_e^2 E_e^2 + g_i^2 E_i^2

    Returns both quadratic roots.
    """

    # --- extract parameters ---
    A, B, C = compute_A_B_C(conductance_diffusion_config, mu_v, sigma_v, verbose=verbose)

    if verbose:
        print("\n=== QUADRATIC COEFFICIENTS ===")
        print("A =", A)
        print("B =", B)
        print("C =", C)

    # --- discriminant ---
    D = B**2 - 4 * A * C

    if verbose:
        print("\n=== DISCRIMINANT ===")
        print("D =", D)

    if D < 0:
        print("\nWARNING: negative discriminant → no real solution")
        return None

    sqrtD = np.sqrt(D)

    # --- solutions ---
    g_e_plus = (-B + sqrtD) / (2 * A)
    g_e_minus = (-B - sqrtD) / (2 * A)

    if verbose:
        print("\n=== SOLUTIONS ===")
        print("g_e0 (+) =", g_e_plus)
        print("g_e0 (-) =", g_e_minus)

    return g_e_plus, g_e_minus