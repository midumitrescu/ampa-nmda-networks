from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

from brian2 import (
    ms,
    mV,
    nS,
    pF,
    Hz,
    second,
    start_scope,
    NeuronGroup,
    StateMonitor,
    run,
    defaultclock,
    seed, Synapses, PoissonGroup, Quantity, SpikeGeneratorGroup, SpikeMonitor, device,
)

from Plotting import show_plots_non_blocking
from iteration_16.model import ConductanceDiffusionSimulationConfig, WANG_MODEL, WangSimulationResult


def create_spike_source(spike_times_ms: np.ndarray | None, poisson_rate: Quantity, N=1):
    if spike_times_ms is None:
        return PoissonGroup(
            N=N,
            rates=poisson_rate
        )

    return SpikeGeneratorGroup(
        N=1,
        indices=np.zeros(len(spike_times_ms), dtype=int),
        times=spike_times_ms * ms,
    )

class WangSimulation:

    @staticmethod
    def run(config: ConductanceDiffusionSimulationConfig):
        start_scope()

        if config.seed is not None:
            device.reinit()
            np.random.seed(config.seed)
            seed(config.seed)
        neuron = NeuronGroup(
            1,
            config.model,
            method="euler",
            namespace={
                "C": config.membrane_capacitance,
                "mg_concentration": config.magnesium_concentration,

                "tau_ampa": config.tau_ampa,
                "tau_gaba": config.tau_gaba,
                "tau_nmda_rise": config.tau_nmda_rise,
                "tau_nmda_decay": config.tau_nmda_decay,
                "alpha_nmda": config.alpha_nmda,

                "E_L": config.e_L,
                "e_ampa": config.e_ampa,
                "e_gaba": config.e_gaba,
                "e_nmda": config.e_nmda,

                "g_L": config.g_L,
                "g_nmda_max": config.g_nmda_max
            },
        )

        defaultclock.dt = config.dt


        neuron.v = config.resting_voltage

        neuron.g_ampa = 0 * nS
        neuron.g_gaba = 0 * nS

        neuron.s_nmda = 0
        neuron.x_nmda = 0

        ampa_source = create_spike_source(
            config.ampa_spike_times,
            config.r_e,
        )

        gaba_source = create_spike_source(
            config.gaba_spike_times,
            config.r_i,
        )

        nmda_source = create_spike_source(
            config.nmda_spike_times,
            config.r_n,
        )

        excitatory_synapses = Synapses(
            ampa_source,
            neuron,
            on_pre="""
            g_ampa += w_ampa
            """,
            namespace={
                "w_ampa": config.w_ampa
            }
        )

        inhibitory_synapse = Synapses(
            gaba_source,
            neuron,
            on_pre="""
            g_gaba += w_gaba
            """,
            namespace={
                "w_gaba": config.w_gaba
            }
        )
        nmda_synapse = Synapses(
            nmda_source,
            neuron,
            on_pre="""
            x_nmda += w_x
            """,
            namespace={
                "w_x": config.w_x
            }
        )

        excitatory_synapses.connect()
        inhibitory_synapse.connect()
        nmda_synapse.connect()

        monitor = StateMonitor(
            neuron,
            [
                "v",
                "g_ampa",
                "g_gaba",
                "s_nmda",
                "x_nmda",
            ],
            record=True,
        )

        ampa_spike_monitor = SpikeMonitor(ampa_source)
        gaba_spike_monitor = SpikeMonitor(gaba_source)
        nmda_spike_monitor = SpikeMonitor(nmda_source)

        run(config.simulation_time)
        result = WangSimulationResult.from_monitors(
            state_monitor=monitor,
            ampa_spike_monitor=ampa_spike_monitor,
            gaba_spike_monitor=gaba_spike_monitor,
            nmda_spike_monitor=nmda_spike_monitor,
        )

        return result

    @staticmethod
    def run_and_plot(config: ConductanceDiffusionSimulationConfig):
        result = WangSimulation.run(config)
        plot(result)
        return result


def plot(result: WangSimulationResult):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10, 10),
        sharex=True
    )

    axes[0].plot(
        result.time_ms,
        result.membrane_voltage_mV
    )
    axes[0].set_ylabel("V (mV)")

    axes[1].plot(
        result.time_ms,
        result.g_ampa_nS
    )
    axes[1].set_ylabel("gAMPA (nS)")

    axes[2].plot(
        result.time_ms,
        result.g_gaba_nS
    )
    axes[2].set_ylabel("gGABA (nS)")

    axes[3].plot(
        result.time_ms,
        result.s_nmda
    )
    axes[3].set_ylabel("gNMDA (nS)")

    axes[4].plot(
        result.time_ms,
        result.x_nmda
    )
    axes[4].set_ylabel("xNMDA (nS)")
    axes[4].set_xlabel("Time (ms)")

    plt.tight_layout()
    show_plots_non_blocking()


if __name__ == '__main__':
    config = ConductanceDiffusionSimulationConfig(
        simulation_time=200 * ms,
        seed=123,
    )

    result = WangSimulation.run(config)

    plot(result)