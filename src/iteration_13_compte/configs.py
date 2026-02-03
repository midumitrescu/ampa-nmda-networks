from brian2 import Hz, nS


class AnExampleExperiment:

    def __init__(self, G_EE_AMPA, G_EE_NMDA, G_EI, G_IE, G_II, NE, NI, label,
                 gext_E=3.1, gext_I=2.38, nu_ext = 1800, N_ext = 1000, NE_ref=2048, NI_ref=512, seed=None):
        self.G_EE_AMPA = G_EE_AMPA * (NE_ref / NE) * nS
        self.G_EE_NDMA = G_EE_NMDA * (NE_ref / NE) * nS
        self.GEI = G_EI * (NE_ref / NE) * nS
        self.GIE = G_IE * (NI_ref / NI) * nS
        self.GII = G_II * (NI_ref / NI) * nS

        self.gext_E = gext_E * nS
        self.gext_I = gext_I * nS

        self.NE = NE
        self.NI = NI

        self.label = label

        self.in_testing = seed is not None
        self.seed = seed

        self.nu_ext_total = nu_ext * Hz
        self.N_ext = N_ext
        self.nu_ext = self.nu_ext_total / self.N_ext
