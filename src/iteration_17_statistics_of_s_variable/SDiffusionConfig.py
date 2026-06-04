from dataclasses import dataclass, field

from brian2 import ms, start_scope, PoissonGroup, NeuronGroup, Synapses, StateMonitor, run, seed, kHz, defaultclock
from brian2 import Hz
from brian2 import second
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import welch, correlate

default_x_and_s_model = """
dx/dt = -x/tau_rise : 1
ds/dt = -s/tau_decay + alpha*x*(1-s): 1
"""


@dataclass(frozen=True)
class SDiffusionConfig:
    T: object = field(default_factory=lambda: 1 * second)
    dt: object = field(default_factory=lambda: 0.05 * ms)

    tau_rise: object = field(default_factory=lambda: 2 * ms)
    tau_decay: object = field(default_factory=lambda: 100 * ms)
    r_n: object = field(default_factory=lambda: 100 * Hz)
    alpha: object = field(default_factory=lambda: 0.5 * kHz)

    w_x: float = 1.0

    seed: int | None = None

    in_testing: bool = False

    experiment_label: str = "default"

    model: str = default_x_and_s_model

    def get_seed(self):
        if self.in_testing:
            return 123456

        return self.seed

def autocov(array: np.ndarray) -> np.ndarray:
    array_centered = array - np.mean(array)

    acov = correlate(
        array_centered,
        array_centered,
        mode="full"
    )

    acov /= len(array_centered)
    return acov[len(acov)//2:]

@dataclass
class SDiffusionResult:
    t: np.ndarray

    x: np.ndarray

    s: np.ndarray

    config: SDiffusionConfig

    def x_power_spectrum(self):
        return welch(
            self.x,
            fs=1 / self.config.dt,
            nperseg=8192
    )

    def s_power_spectrum(self):
        return welch(
            self.s,
            fs=1 / self.config.dt,
            nperseg=8192
    )

    def x_autocovariance(self):
        return autocov(self.x)

    def s_autocovariance(self):
        return autocov(self.s)


class SDiffusionSimulation:

    @staticmethod
    def run(x_and_s_config: SDiffusionConfig):
        start_scope()

        if x_and_s_config.get_seed() is not None:
            seed(x_and_s_config.get_seed())

        defaultclock.dt = x_and_s_config.dt

        source = PoissonGroup(
            1,
            rates=x_and_s_config.r_n
        )

        target = NeuronGroup(
            1,
            x_and_s_config.model,
            method="euler",
            namespace={
                "tau_rise": x_and_s_config.tau_rise,
                "tau_decay": x_and_s_config.tau_decay,
                "alpha": x_and_s_config.alpha
            }
        )

        syn = Synapses(
            source,
            target,
            on_pre="x += w_x" ,
            namespace= {
                "w_x": x_and_s_config.w_x,
            }
        )

        syn.connect()

        mon = StateMonitor(
            target,
            ["x", "s"],
            record=True
        )

        run(x_and_s_config.T)

        return SDiffusionResult(
            t=mon.t[:] / ms,
            x=mon.x[0][:],
            s=mon.s[0][:],
            config=x_and_s_config
        )


def run_single(x_and_s_config):
    return SDiffusionSimulation.run(x_and_s_config)


def run_many_parallel(x_and_s_configs: list[SDiffusionConfig]):
    return (Parallel(
        n_jobs=-1,
        backend="loky")
            (delayed(run_single)(one_config) for one_config in x_and_s_configs))
