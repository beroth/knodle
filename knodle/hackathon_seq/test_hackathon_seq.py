import unittest
import numpy as np
from knodle.hackathon_seq import utils

class TestSeqMethods(unittest.TestCase):
    def test_majority_vote(self):
        no_lf = [0, 0, 0, 0]
        lf_1a = [1, 0, 0, 0]
        #lf_1b = [0, 1, 0, 0]
        lf_1ab = [1, 1, 0, 0]
        lf_2a = [0, 0, 1, 0]
        lf_2b = [0, 0, 0, 1]
        all_lfs = [1, 1, 1, 1]
        other_label = [1, 0, 0]
        label_1 = [0, 1, 0]
        label_2 = [0, 0, 1]
        label_1_2 = [0, 0.5, 0.5]

        lf_matches = np.array([[lf_1a, lf_2a, all_lfs],
                               [lf_2b, lf_1ab, no_lf]])
        expected_labels = np.array([[label_1, label_2, label_1_2],
                                    [label_2, label_1, other_label]])
        lf_to_label = np.array([label_1,
                                label_1,
                                label_2,
                                label_2])
        voted_labels = utils.majority_vote(lf_matches=lf_matches,
                                           t_matrix=lf_to_label, OTHER_ID=0)
        self.assertSequenceEqual(voted_labels.tolist(),
                                 expected_labels.tolist())

if __name__ == '__main__':
    unittest.main()
