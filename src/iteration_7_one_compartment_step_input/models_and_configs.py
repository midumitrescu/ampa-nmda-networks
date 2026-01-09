import brian2.devices.device
import matplotlib.pyplot as plt
import numpy as np
from brian2 import PopulationRateMonitor, SpikeMonitor, StateMonitor, Hz, ms, nsiemens, uamp, seed, mpl, start_scope, \
    defaultclock, kHz, mmole, NeuronGroup, PoissonGroup, Synapses, network_operation, second, mV, run
from loguru import logger
from matplotlib import gridspec
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.axes_grid1.mpl_axes import Axes

from Plotting import show_plots_non_blocking
from iteration_7_one_compartment_step_input.Configuration_with_Up_Down_States import Experiment
from utils import ExtendedDict


main_equations = '''
dv/dt = 1/C * (- I_L - I_ampa - I_gaba - I_nmda): volt (unless refractory)

I_L = g_L * (v-E_leak): amp

I_ampa = g_e * (v - E_ampa): amp
I_gaba = g_i * (v - E_gaba): amp
I_nmda = g_nmda * (v - E_nmda): amp

dg_e/dt = -g_e / tau_ampa : siemens
dg_i/dt = -g_i / tau_gaba  : siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens
ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1
'''

single_compartment_with_nmda = f'''
{main_equations}

sigmoid_v = 1/(1 + (MG_C/mmole)/3.57 * exp(-0.062*(v/mvolt))): 1
'''

single_compartment_without_nmda_deactivation = f'''
{main_equations}

sigmoid_v = 1: 1
'''

logged_variables = '''
one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
v_minus_e_gaba = v-E_gaba : volt
I_fast = I_ampa + I_gaba : amp
'''

single_compartment_with_nmda_and_logged_variables = f'''
{single_compartment_with_nmda}
{logged_variables}
'''

single_compartment_without_nmda_deactivation_and_logged_variables = f'''
{single_compartment_without_nmda_deactivation}
{logged_variables}
'''
