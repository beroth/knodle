import numpy as np

import torch
from torch.utils.data import TensorDataset
from knodle.trainer.snorkel.utils import (
    z_t_matrix_to_snorkel_matrix,
    prepare_empty_rule_matches,
    add_labels_for_empty_examples
)


def test_z_t_matrix_to_snorkel_matrix():
    z = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    t = np.array([
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ])

    snorkel_gold = np.array([
        [-1, 1, -1, -1],
        [-1, -1, 0, -1]
    ])

    snorkel_test = z_t_matrix_to_snorkel_matrix(z, t)
    np.testing.assert_equal(snorkel_gold, snorkel_test)


def test_label_model_data():
    num_samples = 5
    num_features = 16
    num_rules = 6

    x_np = np.ones((num_samples, num_features)).astype(np.float32)
    x_tensor = torch.from_numpy(x_np)
    model_input_x = TensorDataset(x_tensor)

    rule_matches_z = np.ones((num_samples, num_rules))
    rule_matches_z[[1, 4]] = 0

    # test with filtering
    non_zero_mask, out_rule_matches_z, out_model_input_x = prepare_empty_rule_matches(
        rule_matches_z=rule_matches_z,
        model_input_x=model_input_x,
        filter_non_labelled=True
    )

    expected_mask = np.array([True, False, True, True, False])
    expected_rule_matches = np.ones((3, num_rules))

    np.testing.assert_equal(non_zero_mask, expected_mask)
    np.testing.assert_equal(out_rule_matches_z, expected_rule_matches)
    assert len(out_model_input_x) == 3

    # test without filtering
    non_zero_mask, out_rule_matches_z, out_model_input_x = prepare_empty_rule_matches(
        rule_matches_z=rule_matches_z,
        model_input_x=model_input_x,
        filter_non_labelled=False
    )

    expected_mask = np.array([True, False, True, True, False])
    expected_rule_matches = np.ones((3, num_rules))

    np.testing.assert_equal(non_zero_mask, expected_mask)
    np.testing.assert_equal(out_rule_matches_z, expected_rule_matches)
    assert len(out_model_input_x) == 5

def test_other_class_labels():
    label_probs_gen = np.array([
        [0.3, 0.6, 0.0, 0.1],
        [0.2, 0.2, 0.2, 0.4],
        [1.0, 0.0, 0.0, 0.0]
    ])
    output_classes = 5
    other_class_id = 4

    # test without empty rows
    non_zero_mask = np.array([True, True, True])
    expected_probs = np.array([
        [0.3, 0.6, 0.0, 0.1, 0.0],
        [0.2, 0.2, 0.2, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    label_probs = add_labels_for_empty_examples(label_probs_gen, non_zero_mask, output_classes, other_class_id)

    np.testing.assert_equal(label_probs, expected_probs)

    # test with empty rows
    non_zero_mask = np.array([True, False, False, True, True])
    expected_probs = np.array([
        [0.3, 0.6, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.2, 0.2, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0]
    ])
    label_probs = add_labels_for_empty_examples(label_probs_gen, non_zero_mask, output_classes, other_class_id)

    np.testing.assert_equal(label_probs, expected_probs)
