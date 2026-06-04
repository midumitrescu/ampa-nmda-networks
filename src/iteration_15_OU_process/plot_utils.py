import numpy as np
from brian2 import ms
from scipy.ndimage import gaussian_filter1d


def exp_label_to_folder_name(exp_label: str) -> str:
    return exp_label.replace(".", "_").replace(" ", "_").replace(",", "_").lower()


def filter_spikes_in_time_window(spike_times, start, end):
    if spike_times.size == 0:
        return spike_times

    left = np.searchsorted(spike_times, start, side="left")
    right = np.searchsorted(spike_times, end, side="right")

    return spike_times[left:right]


def smoothen_v(v, smooth_width=30, dt=0.1 * ms):
    sigma_ms = smooth_width
    kernel_size = sigma_ms / (dt / ms)

    return gaussian_filter1d(v, sigma=kernel_size)

