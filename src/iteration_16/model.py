from dataclasses import dataclass, field, replace

import numpy as np
from brian2 import (
    ms,
    mV,
    nS,
    Hz,
    Quantity, Mohm, kHz, nF,
)

WANG_MODEL = """
dv/dt = 1/C * (-g_L * (v - E_L) -g_ampa * (v - e_ampa) -g_gaba * (v - e_gaba) - g_nmda_max * s_nmda * sigma_of_v * (v - e_nmda)): volt

dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens

dx_nmda/dt = -x_nmda / tau_nmda_rise : 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_nmda): 1

sigma_of_v = 1 / (1 + exp(-0.062 * v / mV) * (mg_concentration / 3.57)) : 1
"""

WANG_MODEL_FOR_FULL_NMDA_INPUT = """
dv/dt = 1/C * (-g_L * (v - E_L) -g_ampa * (v - e_ampa) -g_gaba * (v - e_gaba) - g_nmda_max * s_nmda * (v - e_nmda)): volt

dg_ampa/dt = -g_ampa / tau_ampa : siemens
dg_gaba/dt = -g_gaba / tau_gaba : siemens

dx_nmda/dt = -x_nmda / tau_nmda_rise : 1
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha_nmda * x_nmda * (1 - s_nmda): 1
"""

def copy_of(arr: np.ndarray):
    return  None if arr is None else arr.copy()

class Chapter1Results:
    mu_v = - 47.61595645 * mV
    sigma_v = 1.9 * mV

@dataclass(frozen=True)
class ConductanceDiffusionSimulationConfig:
    # simulation
    simulation_time: Quantity = field(default_factory=lambda: 500 * ms)
    dt: Quantity = field(default_factory=lambda: 0.05 * ms)

    model: str = WANG_MODEL

    # membrane
    membrane_resistance: Quantity = field(default_factory=lambda: 50 * Mohm)  # Ohm
    membrane_capacitance: Quantity = field(default_factory=lambda: 0.5 * nF) # value from Wang is 0.5 * nF

    e_L: Quantity = field(default_factory=lambda: -65 * mV)
    resting_voltage: Quantity = field(default_factory=lambda: -70 * mV)

    # synaptic reversals
    e_ampa: Quantity = field(default_factory=lambda: 0 * mV)
    e_nmda: Quantity = field(default_factory=lambda: 0 * mV)
    e_gaba: Quantity = field(default_factory=lambda: -80 * mV)

    # conductance decay
    tau_ampa: Quantity = field(default_factory=lambda: 2 * ms)
    tau_gaba: Quantity = field(default_factory=lambda: 5 * ms)

    tau_nmda_rise: Quantity = field(default_factory=lambda: 2 * ms)
    tau_nmda_decay: Quantity = field(default_factory=lambda: 100 * ms)

    alpha_nmda: Quantity = field(default_factory=lambda: 0.5 * kHz)

    N: int | None = None
    # in our manuscript, this is actually k
    k: float = 1

    N_E: int | None = None
    N_I: int | None = None

    r_e: Quantity = field(default_factory=lambda: 0 * Hz)
    r_i: Quantity = field(default_factory=lambda: 0 * Hz)
    r_n: Quantity = field(default_factory=lambda: 0 * Hz)

    ampa_spike_times: np.ndarray | None = None
    gaba_spike_times: np.ndarray | None = None
    nmda_spike_times: np.ndarray | None = None

    g_L: Quantity = field(default_factory=lambda: 20 * nS)
    w_ampa: Quantity = field(default_factory=lambda: 0.5 * nS)
    w_gaba: Quantity = field(default_factory=lambda: 0.5 * nS)
    w_x: Quantity = field(default_factory=lambda: 1)

    g_nmda_max: Quantity = field(default_factory=lambda: 1 * nS)

    magnesium_concentration: float = 1.0

    seed: int | None = None

    # from script \frac{g_{i, 0}}{g_{e, 0}} =  \gamma \\
    def g(self):
        return self.tau_ampa * self.w_ampa * self.N_E

    def with_property(self, **changes):

        if "k" in changes and ("N_E" or "N_I") in changes:
            raise ValueError("Either (N, k) or (N E, N I)")

        if changes.__contains__("N") or changes.__contains__("k"):
            changes["N_E"] = None,
            changes["N_I"] = None

        elif changes.__contains__("N_E"):
            changes["N"] = None

        return replace(
            self,
            ampa_spike_times=copy_of(self.ampa_spike_times),
            gaba_spike_times=copy_of(self.gaba_spike_times),
            nmda_spike_times=copy_of(self.nmda_spike_times),
            **changes,
        )

    def __post_init__(self):

        if self.N_E is not None and self.N_I is not None:
            object.__setattr__(self, "N", self.N_E + self.N_I)
            object.__setattr__(self, "k", self.N_E / self.N_I)
        elif self.N is not None:
            N_E = int(self.N / (1 + self.k))
            N_I = self.N - N_E

            object.__setattr__(self, "N_E", N_E)
            object.__setattr__(self, "N_I", N_I)

        else:
            raise ValueError("Must provide either (N, gamma) or (N_E, N_I)")

calibrated_configuration = ConductanceDiffusionSimulationConfig(
    w_ampa=1.84 * nS,
    w_gaba=8.17 * nS,
    g_nmda_max= 3.551 * nS,
    N_E=1,
    N_I=1
)

''' for pyramidal cells, g ext,AMPA = 2.1, g rec,AMPA = 0.05, gNMDA = 0.165, and g GABA = 1.3 '''
''' What are mean g_s in Wang? g_ampa = g ext,AMPA_0 * 2.4 kHz + g rec,AMPA_0'''

@dataclass
class WangSimulationResult:
    time_ms: np.ndarray

    membrane_voltage_mV: np.ndarray

    g_ampa_nS: np.ndarray
    g_gaba_nS: np.ndarray

    s_nmda: np.ndarray
    x_nmda: np.ndarray

    ampa_presyn_spikes: np.ndarray
    gaba_presyn_spikes: np.ndarray
    nmda_presyn_spikes: np.ndarray

    @classmethod
    def from_monitors(
            cls,
            state_monitor,
            ampa_spike_monitor,
            gaba_spike_monitor,
            nmda_spike_monitor):
        return cls(
            time_ms=state_monitor.t[:] / ms, membrane_voltage_mV=state_monitor.v[0] / mV,
            g_ampa_nS=state_monitor.g_ampa[0] / nS, g_gaba_nS=state_monitor.g_gaba[0] / nS,
            s_nmda=state_monitor.s_nmda[0], x_nmda=state_monitor.x_nmda[0],
            ampa_presyn_spikes=np.asarray(ampa_spike_monitor.t[:] / ms),
            gaba_presyn_spikes=np.asarray(gaba_spike_monitor.t[:] / ms),
            nmda_presyn_spikes=np.asarray(nmda_spike_monitor.t[:] / ms),
        )

    def ampa_spike_delta_v(self, spike_index: int = 0, window_ms: float = 10.0) -> float:
        return self._spike_delta_v(self.ampa_presyn_spikes, spike_index, window_ms)

    def gaba_spike_delta_v(self, spike_index: int = 0, window_ms: float = 30.0) -> float:
        return self._spike_delta_v(self.gaba_presyn_spikes, spike_index, window_ms)

    def nmda_spike_delta_v(self, spike_index: int = 0, window_ms: float = 300.0) -> float:
        return self._spike_delta_v( self.nmda_presyn_spikes, spike_index, window_ms)

    def _spike_delta_v( self, spike_times_ms: np.ndarray, spike_index: int, window_ms: float) -> float:
        """
        Estimate the ΔV produced by a single presynaptic spike.

        Returns:
            V_peak_after_spike - V_before_spike
        """

        if len(spike_times_ms) - 1  < spike_index:
            return

        dt_ms = self.time_ms[1] - self.time_ms[0]
        spike_time_ms = spike_times_ms[spike_index]

        # first sample at or after the spike
        index_of_spike  = int(spike_time_ms / dt_ms) - 1

        if index_of_spike < 0:
            raise ValueError("Spike occurs before first voltage sample.")

        v_before = self.membrane_voltage_mV[index_of_spike]

        n_window = max(1, int(round(window_ms / dt_ms)))
        index_end_of_spike = min(len(self.time_ms), index_of_spike + n_window)

        v_segment = self.membrane_voltage_mV[index_of_spike:index_end_of_spike]

        dv_max = np.max(v_segment) - v_before
        dv_min = np.min(v_segment) - v_before

        # choose whichever excursion is larger
        return float(
            dv_max if abs(dv_max) >= abs(dv_min) else dv_min
        )
