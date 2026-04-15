from brian2 import mV, ms

from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import NeuronModelParams, Experiment


class DiffusionLIFConfig:

    KEY_V_R = NeuronModelParams.KEY_NEURON_V_R
    KEY_TAU_REF = NeuronModelParams.KEY_TAU_REF
    KEY_NEURON_THRESHOLD = NeuronModelParams.KEY_NEURON_THRESHOLD

    KEY_TAU_MEMBRANE = "tau_membrane"
    KEY_MU_V = "mu_v"
    KEY_SIGMA_V = "sigma_v"

    KEY_LABEL = "label"

    def __init__(self, params):
        self.params = params

        self.V_r = params.get(DiffusionLIFConfig.KEY_V_R, -65) * mV
        self.tau_rp = params.get(DiffusionLIFConfig.KEY_TAU_REF, 2) * ms
        self.theta = params.get(DiffusionLIFConfig.KEY_NEURON_THRESHOLD, -40) * mV

        self.tau_m = params.get(DiffusionLIFConfig.KEY_TAU_MEMBRANE, 10) * ms

        self.mu_v = params.get(DiffusionLIFConfig.KEY_TAU_MEMBRANE, -55) * mV
        self.sigma_v = params.get(DiffusionLIFConfig.KEY_SIGMA_V, -0) * mV

        self.label = params.get(DiffusionLIFConfig.KEY_LABEL, "")

    @staticmethod
    def from_experiment(experiment: Experiment):
        return DiffusionLIFConfig({
            DiffusionLIFConfig.KEY_TAU_MEMBRANE: experiment.effective_time_constant_up_state.tau_eff() / ms,
            DiffusionLIFConfig.KEY_TAU_REF: experiment.neuron_params.tau_rp / ms,
            DiffusionLIFConfig.KEY_NEURON_THRESHOLD: experiment.neuron_params.theta / mV,
            DiffusionLIFConfig.KEY_V_R: experiment.neuron_params.V_r / mV,
            DiffusionLIFConfig.KEY_LABEL: experiment.plot_params.panel
        })

    def with_label(self, label: str):
        return self.with_property(DiffusionLIFConfig.KEY_LABEL, label)

    def with_property(self, key: str, value):
        new_params = self.params.copy()
        new_params[key] = value
        return DiffusionLIFConfig(new_params)


default_diffusion_lif_config = DiffusionLIFConfig(params={})