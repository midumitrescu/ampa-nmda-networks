import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms, siemens, cm, second, Hz

from BinarySeach import binary_search_for_target_value
from Configuration import Experiment
from iteration_4_conductance_based_model.conductance_based_model import sim_and_plot, sim


def compute_rate_for_nu_ext_over_nu_thr(nu_ext_over_nu_thr, wait_for=4_000):
    conductance_based_simulation = {

        "sim_time": 5_000,
        "sim_clock": 0.1 * ms,
        "g": 15,
        "g_ampa": 2.518667367869784e-06,
        "g_gaba": 2.518667367869784e-06,
        "nu_ext_over_nu_thr": nu_ext_over_nu_thr,
        "epsilon": 0.1,
        "C_ext": 1000,

        "g_L": 0.00004,

        "panel": f"Testing conductance based model",
        "t_range": [[100, 120], [100, 300], [0, 1000], [1000, 2000]],
        "voltage_range": [-70, -30],
        "smoothened_rate_width": 3 * ms
    }

    experiment = Experiment(conductance_based_simulation)

    skip_iterations = int(wait_for / (experiment.sim_clock / ms))
    rate_monitor, _, _, _ = sim(experiment)

    mean_unsmoothened = np.mean(rate_monitor.rate[skip_iterations:])
    mean_smothened = np.mean(rate_monitor.smooth_rate(width=experiment.plot_params.smoothened_rate_width)[skip_iterations:])
    print(f"For {nu_ext_over_nu_thr : .5f}, we get {mean_smothened}, {mean_unsmoothened}")
    print(f"For {nu_ext_over_nu_thr : .5f}, we get without units {mean_smothened / Hz}, {mean_unsmoothened / Hz}")
    return mean_unsmoothened / Hz, mean_smothened / Hz

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

class MyTestCase(unittest.TestCase):

    def test_new_configuration_is_parsed(self):
        new_config = {

            "g": 1,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_ampa": 0.002,
            "g_gaba": 0.002,

            "tau_ampa_ms": 5,
            "tau_gaba_ms": 5,
        }

        object_under_test = Experiment(new_config)
        self.assertEqual(0.002 * siemens / cm ** 2, object_under_test.synaptic_params.g_ampa)
        self.assertEqual(0.002 * siemens / cm ** 2, object_under_test.synaptic_params.g_gaba)

        self.assertEqual(5 * ms, object_under_test.synaptic_params.tau_ampa)
        self.assertEqual(5 * ms, object_under_test.synaptic_params.tau_gaba)

        self.assertEqual(1, object_under_test.network_params.g)

    def test_model_runs_with_default_eq(self):
        conductance_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 0,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "panel": f"Testing conductance based model",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }

        experiment = Experiment(conductance_based_simulation)
        sim_and_plot(experiment)
        plt.show()

    def test_model_runs_only_with_exc_current(self):
        conductance_based_simulation = {

            "sim_time": 100,
            "sim_clock": 0.05 * ms,

            "g": 1,
            "nu_ext_over_nu_thr": 2,
            "epsilon": 0.1,
            "C_ext": 1000,

            "g_ampa": 0.002,
            "g_gaba": 0.002,

            "panel": f"Testing conductance based model only with excitatory current",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }

        experiment = Experiment(conductance_based_simulation)
        sim_and_plot(experiment, eq="""        
                            dv/dt = - 1/C * g_e * (v-E_ampa) : volt
                            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        """)
        plt.show()

    def test_model_scan_nu_ext_for_excitation_balance(self):

        for nu_ext_over_nu_thr in np.linspace(start=1, stop=3, num=3):
            conductance_based_simulation = {

                "sim_time": 1000,
                "sim_clock": 0.05 * ms,

                "g": 1,
                "nu_ext_over_nu_thr": nu_ext_over_nu_thr,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_ampa": 0.002,
                "g_gaba": 0.002,

                "panel": f"Scanning ",
                "t_range": [100, 120],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 0.5 * ms
            }
            experiment = Experiment(conductance_based_simulation)
            sim_and_plot(experiment, eq="""        
                                        dv/dt = - 1/C * g_e * (v-E_ampa) : volt
                                        dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                                        dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                                    """)
            plt.show()

    def test_model_runs_only_with_inhibitory_current(self):
        conductance_based_simulation = {

            "sim_time": 1500,
            "sim_clock": 0.05 * ms,

            "g": 1,
            "nu_ext_over_nu_thr": 1,
            "epsilon": 0.1,
            "C_ext": 1000,

            "panel": f"Testing conductance based model",
            "t_range": [100, 120],
            "voltage_range": [-70, -30],
            "smoothened_rate_width": 0.5 * ms
        }


        experiment = Experiment(conductance_based_simulation)

        self.assertEqual(1, experiment.network_params.g)

        sim_and_plot(experiment, eq="""        
                            dv/dt = - 1/C * g_i * (v-E_gaba) : volt
                            dg_e/dt = -g_e / tau_ampa : siemens / meter**2
                            dg_i/dt = -g_i / tau_gaba  : siemens / meter**2
                        """)
        plt.show()

    def test_default_model_works(self):
        for current_nu_ext_over_nu_thr, current_g in itertools.product(np.linspace(start=0.01, stop=0.2, num=20), np.linspace(start=10, stop=20, num=10)):
            conductance_based_simulation = {

                "sim_time": 5000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                "g_ampa": 2.518667367869784e-06,
                "g_gaba": 2.518667367869784e-06,
                "nu_ext_over_nu_thr": 10,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Testing conductance based model",
                "t_range": [[100, 120], [100, 300], [0, 1500], [2500, 3000], [4500, 5000]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 3 * ms
            }


            experiment = Experiment(conductance_based_simulation)

            sim_and_plot(experiment)
            plt.show()

    def test_understand_why_most_firing_is_external(self):
        for current_g in np.linspace(start=5, stop=50, num=5):
            conductance_based_simulation = {

                "sim_time": 2000,
                "sim_clock": 0.1 * ms,
                "g": current_g,
                "g_ampa": 1.518667367869784e-06,
                "g_gaba": 1.518667367869784e-06,
                "nu_ext_over_nu_thr": 10,
                "epsilon": 0.1,
                "C_ext": 1000,

                "g_L": 0.00004,

                "panel": f"Testing conductance based model",
                "t_range": [[0, 100], [100, 300], [300, 500], [500, 1000], [1000, 2000], [1900, 1950]],
                "voltage_range": [-70, -30],
                "smoothened_rate_width": 10 * ms
            }

            experiment = Experiment(conductance_based_simulation)

            rate_monitor, spike_monitor, _, _, = sim_and_plot(experiment)
            plt.show()

            sampling_rate = 1000  # Adjust this based on your data's time step

            # Perform the Fast Fourier Transform
            n = len(rate_monitor.t)
            fft_result = np.fft.fft(rate_monitor.rate)
            fft_freq = np.fft.fftfreq(n, d=1 / sampling_rate)
            fft_magnitude = np.abs(fft_result)

            # Only keep the positive half of the spectrum
            positive_frequencies = fft_freq[:n // 2]
            positive_magnitude = fft_magnitude[:n // 2]

            smoothened_magnitude = moving_average(positive_magnitude, window_size=250)


            # Plot the frequency spectrum
            fig, (ax_fft, ax_cvs) = plt.subplots(1, 2, figsize=(14, 6))

            ax_fft.plot(positive_frequencies[2_000:], smoothened_magnitude[2_000:])
            ax_fft.set_title("Frequency Spectrum")
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Magnitude")

            # To identify dominant frequencies
            dominant_frequencies = positive_frequencies[np.argsort(positive_magnitude)[-5:]]  # top 5 frequencies
            print("Dominant frequencies:", dominant_frequencies)

            cvs = self.compute_cvs(spike_monitor)

            ax_cvs.set_title("CVs")
            ax_cvs.set_xlabel("CV")
            ax_cvs.set_ylabel("Density")
            ax_cvs.hist(cvs, bins=50, density=True)

            fig.show()


            print(f"Information regarding CVs: min={np.min(cvs)}, max={np.max(cvs)}, averaage={np.average(cvs)}")
            counts, bins = np.histogram(cvs, bins=50, density=True)
            bin_widths = np.diff(bins)
            area = np.sum(counts * bin_widths)

            print(f"Estimated area under the histogram: {area}")

    def compute_cvs(self, spike_monitor):

        result = np.zeros(len(spike_monitor.spike_trains()))

        for index, spike_train in spike_monitor.spike_trains().items():
            isis_s = np.diff(spike_train)
            result[index] = np.std(isis_s) / np.mean(isis_s)

            if np.mean(isis_s) == 0 or np.std(isis_s) / ms > 1000:
                print(f"Detected mean == 0 at {index}, std={np.std(isis_s)}")

        args_with_nan = np.argwhere(np.isnan(result))
        return result[~np.isnan(result)]

    def test_find_nu_ext_over_nu_thr_binary_search(self):

        def look_for_rate_of_input_value(value):
            return compute_rate_for_nu_ext_over_nu_thr(value)[1]
        #(0.007029794622212648, 0.007029795087873936)
        # (2.3248291021445766, 2.324829102435615)+
        print(binary_search_for_target_value(lower_value=0, upper_value=10, func=look_for_rate_of_input_value, target_result=1))





if __name__ == '__main__':
    unittest.main()
