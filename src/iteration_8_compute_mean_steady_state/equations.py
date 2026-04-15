import numpy as np
from brian2 import mmole, mV


def sigmoid_v(experiment, v):
    MG_C = experiment.synaptic_params.MG_C
    return 1 / (1 + (MG_C / mmole) / 3.57 * np.exp(-0.062 * (v / mV)))