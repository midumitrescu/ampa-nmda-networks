import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from brian2 import ufarad, cm, siemens, mV, ms

# Default directory for test/script-generated figures (clear provenance from filename).
PLOT_OUTPUT_DIR = os.path.join(os.getcwd(), "plot_output")


class SynapticParams:
    KEY_SYNAPTIC_STRENGTH = "J"
    KEY_SYNAPTIC_DELAY = "D"

    def __init__(self, params: dict):
        self.J = params.get(SynapticParams.KEY_SYNAPTIC_STRENGTH, 0.5 * mV)
        self.D = params.get(SynapticParams.KEY_SYNAPTIC_DELAY, 1.5 * ms)

    def __str__(self):
        return f"{self.__class__}(J={self.J}, D={self.D})"


class NetworkParams:
    KEY_G = "g"
    KEY_NU_THR = "nu_thr"
    KEY_NU_E_OVER_NU_THR = "nu_ext_over_nu_thr"

    KEY_N = "N"
    KEY_N_E = "N_E"
    KEY_N_I = "N_I"

    KEY_C_EXT = "C_ext"

    KEY_GAMMA = "GAMMA"
    KEY_EPSILON = "epsilon"

    def __init__(self, params: dict):
        self.synaptic_params = SynapticParams(params)

        self.g = params.get(NetworkParams.KEY_G, 0)
        self.N_E = params.get(NetworkParams.KEY_N_E, 10_000)
        self.gamma = params.get(NetworkParams.KEY_GAMMA, 0.25)
        self.epsilon = params.get(NetworkParams.KEY_EPSILON, 0.1)

        self.N_I = round(self.gamma * self.N_E)
        self.N = self.N_E + self.N_I

        self.C_E = int(self.epsilon * self.N_E)

        self.C_ext = params.get(NetworkParams.KEY_C_EXT, self.C_E)

    def __str__(self):
        return f"{self.__class__}({self.KEY_G}={self.g}, {self.KEY_GAMMA}={self.gamma}, {self.KEY_EPSILON}={self.epsilon}, {self.KEY_N_E}={self.N_E}, {self.KEY_N_I}={self.N_I}, \
                {self.KEY_N}={self.N}, C_E={self.C_E}, {self.KEY_C_EXT}={self.C_ext})"


class NeuronModelParams:
    KEY_NEURON_C = "C"
    KEY_NEURON_G_L = "g_L"
    KEY_NEURON_THRESHOLD = "theta"
    KEY_NEURON_V_R = "v_reset"
    KEY_NEURON_E_L = "E_leak"
    KEY_TAU_REF = "tau_ref"

    def __init__(self, params: dict, network_params: NetworkParams = NetworkParams):
        self.synaptic_params = SynapticParams(params)

        self.C = params.get(NeuronModelParams.KEY_NEURON_C, 1 * ufarad * (cm ** -2))
        self.g_L = params.get(NeuronModelParams.KEY_NEURON_G_L, 0.004 * siemens * (cm ** -2))
        self.theta = params.get(NeuronModelParams.KEY_NEURON_THRESHOLD, -40 * mV)
        self.V_r = params.get(NeuronModelParams.KEY_NEURON_V_R, -65 * mV)
        self.E_leak = params.get(NeuronModelParams.KEY_NEURON_E_L, -65 * mV)
        self.tau_rp = params.get(NeuronModelParams.KEY_TAU_REF, 2 * ms)

        self.tau = self.C / self.g_L
        self.nu_thr = (self.theta - self.E_leak) / (self.synaptic_params.J * network_params.C_E * self.tau)

        logger.info("Computed tau membrane = {}, nu threshold = {}", self.tau, self.nu_thr)

    def __str__(self):
        return (
            f"{self.__class__}({NeuronModelParams.KEY_NEURON_C}={self.C}, {NeuronModelParams.KEY_NEURON_G_L}={self.g_L}, \
                {NeuronModelParams.KEY_NEURON_THRESHOLD}={self.theta}, {NeuronModelParams.KEY_NEURON_V_R}={self.V_r}, \
                {NeuronModelParams.KEY_NEURON_E_L}={self.E_leak}, {NeuronModelParams.KEY_TAU_REF}={self.tau_rp}, nu_thr(computed)={self.nu_thr},\
                tau membrane(computed)={self.tau})")


class PlotParams:
    KEY_PANEL = "panel"

    KEY_T_RANGE = "t_range"
    KEY_RATE_RANGE = "rate_range"
    KEY_VOLTAGE_RANGE = "voltage_range"
    KEY_RATE_TICK_STEP = "rate_tick_step"

    KEY_PLOT_SMOOTH_WIDTH = "smoothened_rate_width"

    def __init__(self, params):
        self.panel = params.get(PlotParams.KEY_PANEL, "")
        self.t_range = params.get(PlotParams.KEY_T_RANGE, [0, 100])
        self.rate_range = params.get(PlotParams.KEY_RATE_RANGE, [0, 150])
        self.voltage_range = params.get(PlotParams.KEY_VOLTAGE_RANGE, None)

        self.rate_tick_step = params.get(PlotParams.KEY_RATE_TICK_STEP, 30)
        self.smoothened_rate_width = params.get(self.KEY_PLOT_SMOOTH_WIDTH, None)
        self.plot_smoothened_rate = self.smoothened_rate_width is not None


class Experiment:
    KEY_SIM_TIME = "sim_time"
    KEY_SIMULATION_CLOCK = "simulation_clock"

    def __init__(self, params: dict):
        self.sim_time = params.get(Experiment.KEY_SIM_TIME, params.get(PlotParams.KEY_T_RANGE, (0, 200))[1]) * ms
        self.network_params = NetworkParams(params)
        self.synaptic_params = SynapticParams(params)
        self.neuron_params = NeuronModelParams(params=params, network_params=self.network_params)
        self.plot_params = PlotParams(params)

        self.nu_ext_over_nu_thr = params.get(NetworkParams.KEY_NU_E_OVER_NU_THR, 1)
        self.nu_thr = self.neuron_params.nu_thr
        self.nu_ext = self.nu_ext_over_nu_thr * self.nu_thr

        self.mean_excitatory_input = self.synaptic_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext
        self.mean_inhibitory_input = - self.network_params.g * self.synaptic_params.J * self.neuron_params.tau * self.network_params.C_E * self.nu_ext

        self.sim_clock = params.get(Experiment.KEY_SIMULATION_CLOCK, 0.05 * ms)


def prepare_bigger_fonts(zoom=0):
    if zoom == 0:
        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "figure.titlesize": 20
        })
    elif zoom == 1:
        plt.rcParams.update({
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "legend.fontsize": 18,
            "figure.titlesize": 24
        })
    elif zoom == 2:
        plt.rcParams.update({
            "font.size": 24,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "legend.fontsize": 20,
            "figure.titlesize": 28
        })


def _safe_filename_part(s):
    """Replace anything that's not alphanumeric or underscore with underscore."""
    return re.sub(r"[^\w]", "_", str(s))


def build_figure_basename(caller_test_case=None, script_file=None, descriptor=None):
    """
    Build a filename stem that identifies exactly which test or script produced the figure.
    - From unittest: use caller_test_case (TestCase instance) → {module_file}_{class_name}_{method_name}[_{descriptor}]
    - From script: use script_file (__file__) and optional descriptor → {script_basename}[_{descriptor}]
    """
    if caller_test_case is not None:
        mod = getattr(caller_test_case.__class__, "__module__", "")
        module_part = mod.split(".")[-1] if mod else "unknown"
        class_name = caller_test_case.__class__.__name__
        method_name = getattr(caller_test_case, "_testMethodName", "unknown")
        base = f"{_safe_filename_part(module_part)}_{_safe_filename_part(class_name)}_{_safe_filename_part(method_name)}"
        if descriptor:
            base = f"{base}_{_safe_filename_part(descriptor)}"
        return base
    if script_file is not None:
        base = _safe_filename_part(os.path.splitext(os.path.basename(script_file))[0])
        if descriptor:
            return f"{base}_{_safe_filename_part(descriptor)}"
        return base
    return "figure"


def get_or_create_file_base_dir(
    save_name=None,
    out_dir=None,
    caller_test_case=None,
    script_file=None,
    descriptor=None,
):
    if out_dir is None:
        out_dir = PLOT_OUTPUT_DIR

    if save_name is None and (
        caller_test_case is not None or script_file is not None
    ):
        save_name = build_figure_basename(
            caller_test_case=caller_test_case,
            script_file=script_file,
            descriptor=descriptor,
        )

    if save_name is None:
        return None

    base_dir = Path(out_dir) / save_name

    base_dir.mkdir(parents=True, exist_ok=True)

    return Path(base_dir)


def save_current_figure(
        save_name=None,
        out_dir=None,
        caller_test_case=None,
        script_file=None,
        descriptor=None,
):
    """
    Save the current matplotlib figure to a PNG file.

    Builds the filename from save_name, or from caller_test_case / script_file when save_name
    is not provided. Creates out_dir if needed. Returns the absolute path of the saved file,
    or None if no save was performed (no save_name and no caller/script to build one).

    - save_name: exact stem for the file (e.g. "my_figure")
    - out_dir: directory to write to (default: PLOT_OUTPUT_DIR)
    - caller_test_case: unittest.TestCase instance → filename from module, class, method
    - script_file: __file__ and optional descriptor → filename from script basename
    """
    if out_dir is None:
        out_dir = PLOT_OUTPUT_DIR
    if save_name is None and (caller_test_case is not None or script_file is not None):
        save_name = build_figure_basename(
            caller_test_case=caller_test_case, script_file=script_file, descriptor=descriptor
        )
    if save_name is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{save_name}.png")
    fig = plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure: {}", path)
    return os.path.abspath(path)


def add_panel_info(ax_iterable, panel_labels=None):
    if panel_labels is None:
        panel_labels = [f"{chr(ord("A") + index)}" for index in range(0, len(ax_iterable))]
    for ax, label in zip(ax_iterable, panel_labels):
        ax.text(
            0.02, 1.2, f"({label})",
            transform=ax.transAxes,
            fontsize=20,
            fontweight=1000,
            va="top",
            ha="left"
        )


def show_plots_non_blocking(
        show=True,
        save_name=None,
        caller_test_case=None,
        script_file=None,
        descriptor=None,
        out_dir=None,
):
    """
    Apply bigger fonts, optionally save current figure to a file with a clear test/script name, then show non-blocking and close.

    For saving, provide either:
    - save_name: exact stem for the file (e.g. "SiegerGradientDescentTestCases_GradientDescentTestCases_test_foo")
    - caller_test_case: unittest.TestCase instance (e.g. self) → filename from module, class, method
    - script_file: __file__ and optional descriptor → filename from script basename

    File is written to out_dir (default: plot_output under cwd) as {save_name}.png.
    """
    save_current_figure(
        save_name=save_name,
        out_dir=out_dir,
        caller_test_case=caller_test_case,
        script_file=script_file,
        descriptor=descriptor,
    )
    if show:
        plt.show(block=False)
        plt.close("all")
