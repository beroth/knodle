import numpy as np
import json, math
import torch
from torch.utils.data import Dataset

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def majority_vote(lf_matches, t_matrix, OTHER_ID = 0):
    """
    Majority voting, from labeling function (lf) matches (multi-hot) to labels (normalized multi-hot).
    Works also for sequences, where each position has its own vector (the last dimension of the array)
    counting labeling function matches.
    You can specify a OTHER_ID, positions with no matches are mapped to this output label.
    If OTHER_ID is set to None, it is required that every position has at least one matching labeling function.
    (This can be ensured by, e.g., filtering out those positions in a pre-processing step).

    :param lf_matches: Multi-dimensional numpy array, the last dimension is a vector with one dimension per labeling
    function.
    :param t_matrix: 2-dimensional numpy array, required to have size num_lfs x num_labels. A cell in row r and column c
    contains the value 1 if the lf with lf_id==r corresponds to the label with label_id==c (other cells are 0).
    :param OTHER_ID: Label id to which instances / positions with no lf are mapped to.
    :return: Multi-dimensional numpy array, the last dimension is a vector with a probability distribution over all
    labels.
    """
    ws_labels = lf_matches.dot(t_matrix)
    num_labels = t_matrix.shape[1]

    # Count all lf matches
    mv_vector = np.ones(num_labels)

    if OTHER_ID == None:
        lf_match = ws_labels.dot(mv_vector)
        assert((lf_match >= 1).all())
    else:
        assert(OTHER_ID >= 0)
        assert(OTHER_ID < num_labels)
        mv_vector[OTHER_ID] = 0
        lf_match = ws_labels.dot(mv_vector)
        # If no lf-match, assign OTHER class
        ws_labels[lf_match==0, OTHER_ID] = 1

    # Divide by total count (last axis)
    ws_labels = ws_labels / ws_labels.sum(axis=-1)[..., None]
    return ws_labels

def read_atis(atis_fn = "atis.json"):
    with open(atis_fn, "rb") as input_file:
        data = json.load(input_file, encoding="UTF8")
    train_dev_sents = data["train_sents"] # list of lists
    train_dev_lfs = data["train_labels"] # list of lists
    num_train = math.floor(0.8 * len(train_dev_sents))
    train_sents = train_dev_sents[:num_train]
    train_lfs = train_dev_lfs[:num_train]
    dev_sents = train_dev_sents[num_train:]
    dev_lfs = train_dev_lfs[num_train:]
    test_sents = data["test_sents"]
    test_lfs = data["test_labels"]
    word_to_id = data["vocab"]
    lf_to_id = data["label_dict"]
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    NUM_LFs = len(lf_to_id)
    MAX_LENGTH = 20

    #  We reorganize the data so that "<PAD>" has id 0, "<UNK>" has id 1, and the label 'O' has id 0 (and all words are still represented). Make sure to also change word_to_id and label_to_id.
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[lf]: lf for lf in lf_to_id}

    def switch_idx(sents, i, j):
        D = {i:j, j:i}
        return [[D.get(word, word) for word in sent] for sent in sents]

    for i, j in [(word_to_id[PAD_TOKEN], 0), (word_to_id[UNK_TOKEN], 1)]:
        train_sents = switch_idx(train_sents, i, j)
        dev_sents = switch_idx(dev_sents, i, j)
        test_sents = switch_idx(test_sents, i, j)

    train_lfs = switch_idx(train_lfs, lf_to_id["O"], 0)
    dev_lfs = switch_idx(dev_lfs, lf_to_id["O"], 0)
    test_lfs = switch_idx(test_lfs, lf_to_id["O"], 0)
    word_to_id[id_to_word[0]] = word_to_id[PAD_TOKEN]
    word_to_id[id_to_word[1]] = word_to_id[UNK_TOKEN]
    lf_to_id[id_to_lf[0]] = lf_to_id["O"]
    lf_to_id["O"] = 0
    word_to_id[PAD_TOKEN] = 0
    word_to_id[UNK_TOKEN] = 1
    id_to_word = {word_to_id[word]: word for word in word_to_id}
    id_to_lf = {lf_to_id[label]: label for label in lf_to_id}

    def do_padding(sequences, length = MAX_LENGTH):
        return pad_sequences(sequences, maxlen = length)

    train_sents_padded = do_padding(train_sents)
    dev_sents_padded = do_padding(dev_sents)
    test_sents_padded = do_padding(test_sents)

    lf_to_newlabel = dict()
    for oldlabel in lf_to_id.keys():
        newlabel = oldlabel.split(".")[0].split("-")[-1]
        lf_to_newlabel[oldlabel] = newlabel
    del lf_to_newlabel["O"]

    # 1. use original labels, without "O" to represent Z-matrix
    train_lfs_padded = to_categorical(do_padding(train_lfs), NUM_LFs)[:,:,1:]
    dev_lfs_padded = to_categorical(do_padding(dev_lfs), NUM_LFs)[:,:,1:]
    test_lfs_padded = to_categorical(do_padding(test_lfs), NUM_LFs)[:,:,1:]

    newlabel_to_id = dict()
    OTHER_ID = 0
    OTHER_LABEL = "O"
    newlabel_to_id[OTHER_LABEL] = OTHER_ID
    for i, nl in enumerate(set(lf_to_newlabel.values())):
        newlabel_to_id[nl] = i + 1

    lfid_to_nlid = dict()
    for lf_id in id_to_lf:
        if lf_id == 0:
            continue
        lf_name = id_to_lf[lf_id]
        newlabel_name = lf_to_newlabel[lf_name]
        nl_id = newlabel_to_id[newlabel_name]
        # "O" is not a lf in the 1-hot encoding of the original labels
        lfid_to_nlid[lf_id - 1] = nl_id

    t_matrix = np.zeros((len(lfid_to_nlid), len(newlabel_to_id)))
    for lfid in lfid_to_nlid:
        t_matrix[lfid, lfid_to_nlid[lfid]] = 1

    id_to_label = {i:l for l,i in newlabel_to_id.items()}

    dev_labels_padded = majority_vote(dev_lfs_padded, t_matrix).argmax(axis=2)
    #test_labels = majority_vote(test_labels_padded, t_matrix).argmax(axis=2)
    return train_sents_padded, train_lfs_padded, dev_sents_padded, dev_labels_padded, \
           t_matrix, id_to_word, id_to_lf, id_to_label

def accuracy_padded(predicted, gold, mask):
    # Expects label ids, NOT 1-hot encodings.
    #return np.sum((predicted == gold) * mask) / np.sum(mask)
    return sum(sum((predicted == gold) * mask)) / sum(sum(mask))


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.
    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    #print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    #print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class SeqDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
        self.data_size = len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.labels[i]

    def __len__(self):
        return self.data_size