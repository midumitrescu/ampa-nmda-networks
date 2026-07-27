import math
from loguru import logger

from dataclasses import dataclass, field, replace

import numpy as np
from brian2 import (
    ms,
    mV,
    nS,
    Hz,
    Quantity, Mohm, kHz, nF, is_dimensionless,
)

from iteration_16.model import ConductanceDiffusionSimulationConfig, config_with_intermediate_synapses, \
    high_shunt_config


@dataclass(frozen=True)
class Stimulus:
    baseline: ConductanceDiffusionSimulationConfig
    label: int
    t_onset: Quantity = field(default_factory=lambda: 100 * ms)
    t_offset: Quantity = field(default_factory=lambda: 400 * ms)

    delta_e_rate: Quantity = field(default_factory=lambda: 20 * Hz)

    to_compartment = 1

    def with_property(self, **changes):
        return replace(
            self,
            **changes,
        )

a_stimulus = Stimulus(baseline = high_shunt_config, label=0)

@dataclass(frozen=True)
class Stimuli:

    baseline: ConductanceDiffusionSimulationConfig
    stimuli: list[Stimulus]

    def with_property(self, **changes):
        return replace(
            self,
            **changes,
        )

    @staticmethod
    def of(stimuli: list[Stimulus]):
        return Stimuli(stimuli=stimuli, baseline=stimuli[0].baseline)
