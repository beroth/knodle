import torch
import utils
import models
import seq_trainer

train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, \
    t_matrix, id_to_word, id_to_lf, id_to_label = utils.read_atis()

model = models.LstmModel(vocab_size=len(id_to_word), tagset_size=len(id_to_label), embedding_dim=15, hidden_dim=15)

trainer = seq_trainer.MajorityVoteSeqTrainer(model=model, mapping_rules_labels_t=t_matrix)

trainer.train(model_input_x=train_sents_padded,
              rule_matches_z=train_lfs_padded,
              dev_model_input_x=dev_sents_padded,
              dev_gold_labels_y=dev_labels_padded,
              epochs=10)

predicted_labels = trainer.model(torch.tensor(dev_sents_padded)).argmax(axis=2)

print([(id_to_word[wid], id_to_label[lid]) for (wid, lid) in zip(dev_sents_padded[0], predicted_labels[0].tolist())])


import numpy as np

#predicted_labels = model(torch.tensor(dev_sents_padded)).argmax(axis=2)
#print(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)))

dev_acc = float(utils.accuracy_padded(predicted_labels, torch.tensor(dev_labels_padded),
                                      mask=torch.tensor(dev_sents_padded != 0)))

majority_acc = float(utils.accuracy_padded(torch.tensor(np.zeros_like(dev_labels_padded)),
                                           torch.tensor(dev_labels_padded), mask=torch.tensor(dev_sents_padded != 0)))

print(dev_acc)

print(majority_acc)
