"""Rare-boundary supervision must survive sequence masks and class imbalance."""

import numpy as np
import pytest
import torch

from smg_rl.config import load_config
from smg_rl.prediction_audit import validation_batches
from smg_rl.replay import replay_signature
from smg_rl.world_model import prediction_mean, weighted_mean


def test_balanced_prediction_is_invariant_to_repeated_live_class():
    values = torch.tensor([1., 9.], requires_grad=True)
    boundary = torch.tensor([0., 1.])
    assert prediction_mean(values, torch.ones(2), boundary, True).item() == 5
    repeated = torch.cat([values[:1].repeat(99), values[1:]])
    terminal = torch.cat([torch.zeros(99), torch.ones(1)])
    assert prediction_mean(repeated, torch.ones(100), terminal, True).item() == 5
    assert prediction_mean(repeated, torch.ones(100), terminal).item() < 2


def test_boundary_masks_exclude_padding_burn_in_and_empty_classes():
    values = torch.tensor([[1000., 2., 6., 1000.]], requires_grad=True)
    mask = torch.tensor([[0., 1., 1., 0.]])
    terminal = torch.tensor([[1., 0., 1., 0.]])
    loss = prediction_mean(values, mask, terminal, True)
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0., .5, .5, 0.]]))
    assert loss.item() == 4
    torch.testing.assert_close(prediction_mean(values, mask, terminal, False),
                               weighted_mean(values, mask))
    assert prediction_mean(values, torch.zeros_like(mask), terminal, True).item() == 0
    assert prediction_mean(values, mask, torch.zeros_like(mask), True).item() == 4


def test_prediction_audit_preserves_next_frame_labels_and_masks_padding():
    traces = []
    for length in (3, 10, 50, 70):
        traces.append({'images': np.arange(length + 1, dtype=np.uint8)[:, None, None],
                       'actions': np.zeros(length, np.int64), 'rewards': np.arange(length, dtype=np.float32),
                       'terminated': np.arange(length) == length - 1, 'truncated': np.zeros(length, bool),
                       'is_first': np.arange(length + 1) == 0, 'scenario': np.ones(1, np.float32)})
    settings = {'sequence_length': 32, 'burn_in': 8, 'batch_size': 8}
    batches = validation_batches(traces, settings)
    assert len(batches) == 8
    for batch in batches:
        active = batch['mask'].astype(bool)
        assert np.array_equal(batch['images'][:, 1:, 0, 0][active], batch['rewards'][active] + 1)
        assert not (batch['terminated'] & ~active).any()
    assert all(b['terminated'].sum() == 8 for b in batches[4:])
    with pytest.raises(ValueError, match='four'):
        validation_batches(traces[:3], settings)


def test_repaired_profile_preserves_original_quality_gates_and_reward_identity():
    original = load_config('config/motor8-smg1.yaml')
    repaired = load_config('config/motor8-smg1-boundary-repaired.yaml')
    assert replay_signature(original) == replay_signature(repaired)
    for key, value in original['world_model'].items():
        if key.startswith(('minimum_', 'maximum_')) or key == 'persistence_ratio':
            assert repaired['world_model'][key] == value
    assert repaired['world_model']['balance_terminal_losses'] is True
    assert repaired['world_model']['disagreement_scale'] == 0
