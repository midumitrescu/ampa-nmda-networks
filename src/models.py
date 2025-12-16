import enum

from utils import ExtendedDict

class NMDAModels(enum.Enum):
    model_with_detailed_hidden_variables = ExtendedDict({
        "eq": """
dv/dt = 1/C * (- I_L - I_syn - I_Exc - I_Inh - I_nmda): volt (unless refractory)

I_L = g_L * (v-E_leak): amp
I_syn = g_e_syn * (v-E_ampa): amp
I_Exc = g_e * (v-E_ampa): amp
I_Inh = g_i * (v-E_gaba): amp
I_nmda = g_nmda * (v - E_ampa): amp

dg_e_syn/dt = -g_e_syn / tau_ampa  : siemens
dg_e/dt = -g_e / tau_ampa : siemens
dg_i/dt = -g_i / tau_gaba  : siemens

g_nmda = g_nmda_max * sigmoid_v * s_nmda: siemens

ds_nmda/dt = -s_nmda / tau_nmda_decay + alpha * x_nmda * (1 - s_nmda) : 1
x_nmda_not_cliped : 1
dx_nmda/dt = - x_nmda / tau_nmda_rise : 1

sigmoid_v = 1/(1 + exp(-0.062 * v/mvolt) * (MG_C/mmole / 3.57)) : 1
one_minus_s_nmda = 1 - s_nmda : 1
alpha_x_t = alpha * x_nmda: Hz
s_drive = alpha * x_nmda * (1 - s_nmda) : Hz
""",
        "documentation": "Model with detailed hidden variables and the possibility of setting a g max (g bar). This is "
                         "the refactored NMDA model, after correcting the modelling issue. Both x and s are unitless",
    })