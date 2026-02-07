from brian2 import Hz, nS, PopulationRateMonitor, ms, SpikeMonitor, StateMonitor
import numpy as np
from brian2.units.allunits import nampere, pampere

from utils import ExtendedDict


class AnExampleExperiment:

    def __init__(self, G_EE_AMPA, G_EE_NMDA, G_EI, G_IE, G_II, NE, NI, label,
                 gext_E=3.1, gext_I=2.38, nu_ext=1800, N_ext=1000, NE_ref=2048, NI_ref=512, seed=None):
        self.G_EE_AMPA = G_EE_AMPA * (NE_ref / NE) * nS
        self.G_EE_NMDA = G_EE_NMDA * (NE_ref / NE) * nS
        self.GEI = G_EI * (NE_ref / NE) * nS
        self.GIE = G_IE * (NI_ref / NI) * nS
        self.GII = G_II * (NI_ref / NI) * nS

        self.gext_E = gext_E * nS
        self.gext_I = gext_I * nS

        self.NE = NE
        self.NI = NI

        self.label = label

        self.in_testing = seed is not None
        self.seed = seed

        self.nu_ext_total = nu_ext * Hz
        self.N_ext = N_ext
        self.nu_ext = self.nu_ext_total / self.N_ext

    def summary(self):
        return {
            "G_EE_AMPA": self.G_EE_AMPA / nS,
            "G_EE_NDMA": self.G_EE_NMDA / nS,
            "GEI": self.GEI / nS,
            "GIE": self.GIE / nS,
            "GII": self.GII / nS
        }


class CompteResults:

    def __init__(self, population_rate_monitor: PopulationRateMonitor, spikes_monitor: SpikeMonitor,
                 currents_monitor: StateMonitor, example: AnExampleExperiment, sim_time: int, dt: float):
        self.example = example
        self.sim_time = sim_time # ms
        self.dt = dt # ms

        self.population_rate_monitor = self.__extract_rates__(population_rate_monitor)
        self.spikes_monitor = self.__extract_spikes__(spikes_monitor, sim_time=sim_time)
        self.currents_monitor = self.__extract_currents__(currents_monitor)

    def extract_end_rates(self, last_percent: float = 0.2):
        rates = self.population_rate_monitor.population_rate
        region_of_interest = int((1 - last_percent) * len(rates))
        return np.mean(rates[region_of_interest:])

    def __extract_rates__(self, rate_monitor: PopulationRateMonitor):
        if rate_monitor is None:
            return None
        rates_to_extract = rate_monitor.smooth_rate(width=10 * ms) / Hz
        return ExtendedDict({
            "t": np.array(rate_monitor.t / ms),
            "population_rate": np.array(rates_to_extract)
        })

    @staticmethod
    def __extract_spikes__(spike_monitor: SpikeMonitor, sim_time):
        if spike_monitor is None:
            return ExtendedDict({})
        return ExtendedDict({
            "t": np.array(spike_monitor.t / ms),
            "i": np.array(spike_monitor.i),
            "all_values": spike_monitor.all_values()['t'],
            "num_spikes": spike_monitor.num_spikes,
            "mean_rate": spike_monitor.num_spikes / sim_time,
        })

    def __extract_currents__(self, currents_monitor: StateMonitor):
        if currents_monitor is None:
            return ExtendedDict({})
        values = {
            "t": np.array(currents_monitor.t / ms),
            "dt": self.dt,
            "recorded": np.array(currents_monitor.needed_variables),
        }
        dt = self.dt
        for current in currents_monitor.needed_variables:
            current_array = np.array(currents_monitor.__getattr__(current) / pampere)
            values[current] = current_array
            q_name = current.replace("I", "q")
            values[q_name] = np.sum(current_array * dt, axis=1)
        return ExtendedDict(values)

    def stats(self):
        result = self.example.summary()
        result['end_rate'] = self.extract_end_rates()
        return result
