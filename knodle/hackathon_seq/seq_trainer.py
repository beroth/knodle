import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from knodle.hackathon_seq import utils

class MajorityVoteSeqTrainer:
    def __init__(self, model, mapping_rules_labels_t):
        self.model = model
        self.mapping_rules_labels_t = mapping_rules_labels_t

    def train(
            self,
            model_input_x: np.ndarray = None, rule_matches_z: np.ndarray = None,
            dev_model_input_x: np.ndarray = None, dev_gold_labels_y: np.ndarray = None,
            epochs = 20
    ):
        # Training parameters
        start_epoch = 0  # start at this epoch
        batch_size = 10  # batch size
        lr = 0.015  # learning rate
        lr_decay = 0.05  # decay learning rate by this amount
        momentum = 0.9  # momentum
        workers = 1  # number of workers for loading data in the DataLoader
        grad_clip = 5.  # clip gradients at this value
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad,
                                                  self.model.parameters()), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss().to(device)
        train_mapped_labels = utils.majority_vote(rule_matches_z, self.mapping_rules_labels_t).argmax(axis=2)
        train_loader = torch.utils.data.DataLoader(utils.SeqDataset(model_input_x, train_mapped_labels), batch_size=batch_size, shuffle=True,
                                                   num_workers=workers, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(utils.SeqDataset(dev_model_input_x, dev_gold_labels_y), batch_size=batch_size, shuffle=True,
                                                 num_workers=workers, pin_memory=False)
        print_val = True
        for epoch in range(start_epoch, epochs):
            self.model.train()
            if print_val:
                avg_acc = 0
                for i, (tokens, labels) in enumerate(val_loader):
                    preds = self.model(tokens).argmax(axis=2)
                    mask = (tokens != 0)
                    acc = utils.accuracy_padded(predicted=preds, gold=labels, mask=mask)
                    avg_acc = (avg_acc * i + acc) / (i + 1)
                print(avg_acc)
            for i, (tokens, labels) in enumerate(train_loader):
                logits = self.model(tokens).permute(0, 2, 1)
                loss = criterion(logits, labels)
                # Back prop.
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    utils.clip_gradient(optimizer, grad_clip)
                optimizer.step()
            utils.adjust_learning_rate(optimizer, lr / (1 + (epoch + 1) * lr_decay))
