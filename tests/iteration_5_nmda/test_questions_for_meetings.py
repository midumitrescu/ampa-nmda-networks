import unittest

from matplotlib import pyplot as plt

from iteration_4_conductance_based_model.Questions import question_q_network_instability_for_g_0


class TestsForMeeting(unittest.TestCase):

    @unittest.skip(reason="very long running. Run by hand")
    def test_q_0(self):
        question_q_network_instability_for_g_0()
        plt.close()

    @unittest.skip(reason="very long running. Run by hand")
    def test_q_0_for_other_g_values(self):
        question_q_network_instability_for_g_0(g=1)
        plt.close()
        question_q_network_instability_for_g_0(g=2)
        plt.close()

    @unittest.skip(reason="very long running. Run by hand")
    def test_in_development_delete_this(self):
        question_q_network_instability_for_g_0(g=5, g_ampa_to_take=2.625e-6, nu_ext_over_nu_thr=1.8)
        plt.close()




if __name__ == '__main__':
    unittest.main()
