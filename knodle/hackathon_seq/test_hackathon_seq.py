import unittest
import numpy as np
import torch
import utils
import models
import seq_trainer

class TestSeqMethods(unittest.TestCase):
    def test_01_majority_vote(self):
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



    def test_02_train_evaluate_atis(self):
        train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, \
        t_matrix, id_to_word, id_to_lf, id_to_label = utils.read_atis()

        model = models.LstmModel(vocab_size=len(id_to_word), tagset_size=len(id_to_label), embedding_dim=15, hidden_dim=15)

        predicted_labels = model(torch.tensor(dev_sents_padded)).argmax(axis=2)
        majority_acc = float(utils.accuracy_padded(torch.tensor(np.zeros_like(dev_labels_padded)),
                                                   torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)))

        self.assertAlmostEqual(majority_acc, 0.6341, places=2)
        untrained_acc = float(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded),
                                                    mask=torch.tensor(dev_sents_padded != 0)))
        self.assertLessEqual(untrained_acc, majority_acc + 0.05)

        trainer = seq_trainer.MajorityVoteSeqTrainer(model=model, mapping_rules_labels_t=t_matrix)
        trainer.train(model_input_x=train_sents_padded,
                      rule_matches_z=train_lfs_padded,
                      dev_model_input_x=dev_sents_padded,
                      dev_gold_labels_y=dev_labels_padded,
                      epochs=10)

        predicted_labels = trainer.model(torch.tensor(dev_sents_padded)).argmax(axis=2)
        trained_acc = float(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded),
                                              mask=torch.tensor(dev_sents_padded != 0)))

        self.assertGreaterEqual(trained_acc, 0.85)

if __name__ == '__main__':
    unittest.main()
