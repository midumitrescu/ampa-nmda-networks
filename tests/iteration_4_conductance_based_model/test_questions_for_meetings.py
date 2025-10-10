import unittest

from matplotlib import pyplot as plt

from iteration_4_conductance_based_model.Questions import question_q_network_instability_for_g_0


class TestsForMeeting(unittest.TestCase):

    @unittest.skip(reason="very long running. Run by hand")
    def test_q_0(self):
        question_q_network_instability_for_g_0()
        plt.close()


if __name__ == '__main__':
    unittest.main()
