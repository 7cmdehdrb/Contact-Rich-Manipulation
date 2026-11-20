"""Binary two-of-three contact credit, parent behavior and reset boundaries."""
import importlib
import itertools

import pytest
import torch


@pytest.fixture
def states(load_mdp):
    return (importlib.import_module('_baseline_cpu_pkg.mdp.common.sweep_v0_state'),
            importlib.import_module('_baseline_cpu_pkg.mdp.common.stabilize_sweep_state'))


def advance(state, contact, *, valid=True, height=1.1, force=2.):
    n = len(state.contact_history)
    # Previous goal distance 1.0 -> current .99 gives 1cm progress.
    state.previous_distance.fill_(1.)
    state.finish(torch.tensor([[.01, 0., 0.]]).expand(n, -1),
                 torch.tensor([[1., 0., 0.]]).expand(n, -1),
                 torch.tensor([[force, 0., 0.]]).expand(n, -1),
                 torch.full((n,), contact), torch.full((n,), valid),
                 torch.full((n,), height), torch.ones(n))


def test_all_histories_use_boolean_recognition_without_counter(states):
    single, _ = states
    histories = list(itertools.product([False, True], repeat=3))
    for initial in (False, True):
        state = single.SingleContactSweepState(8, 'cpu')
        for row, (current, previous, earlier) in enumerate(histories):
            state.contact_history[row, :2] = torch.tensor([previous, earlier])
        assert not hasattr(state, 'contact_count')
        assert state.contact_recognized.dtype == torch.bool
        state.contact_recognized.fill_(initial)
        # Batch real finish calls, including the existing current-contact force gate.
        state.previous_distance.fill_(1.)
        state.finish(torch.tensor([[.01, 0., 0.]]).expand(8, -1),
                     torch.tensor([[1., 0., 0.]]).expand(8, -1),
                     torch.tensor([[2., 0., 0.]]).expand(8, -1),
                     torch.tensor([h[0] for h in histories]), torch.ones(8, dtype=torch.bool),
                     torch.full((8,), 1.1), torch.ones(8))
        recognized = torch.tensor([sum(h) >= 2 for h in histories])
        assert state.contact_recognized.tolist() == recognized.tolist()
        expected = torch.tensor([.01 if sum(h) >= 2 and h[0] else 0. for h in histories])
        torch.testing.assert_close(state.positive, expected)


def test_parent_ramp_and_single_contact_sequence(states):
    single, parent = states
    binary, ramp = single.SingleContactSweepState(1, 'cpu'), parent.SweepOnlyState(1, 'cpu')
    observed = []
    for contact in (True, True, True, False, False, True, True):
        advance(binary, contact)
        advance(ramp, contact)
        observed.append(binary.contact_recognized.item())
        assert not hasattr(binary, 'contact_count')
    assert observed == [False, True, True, True, False, False, True]
    assert ramp.contact_count.item() == 1
    torch.testing.assert_close(binary.positive, torch.tensor([.01]))
    torch.testing.assert_close(ramp.positive, torch.tensor([.0005]))


@pytest.mark.parametrize('kwargs', [{'valid': False}, {'height': 1.12}, {'height': float('nan')}])
def test_validity_and_height_gates_are_preserved(states, kwargs):
    single, _ = states
    state = single.SingleContactSweepState(1, 'cpu')
    state.contact_history.fill_(True)
    state.contact_recognized.fill_(True)
    advance(state, True, **kwargs)
    assert state.positive.item() == 0.
    if 'height' in kwargs:
        assert not state.contact_recognized.item()


def test_partial_reset_and_reacquisition(states):
    single, _ = states
    state = single.SingleContactSweepState(2, 'cpu')
    state.contact_history.fill_(True)
    state.contact_recognized.fill_(True)
    state.reset(torch.tensor([0]), torch.zeros(1, 3), torch.tensor([[1., 0., 0., 0.]]),
                torch.tensor([[1., 0., 0.]]))
    assert not state.contact_history[0].any() and not state.contact_recognized[0]
    assert state.contact_history[1].all() and state.contact_recognized[1]
    advance(state, True)
    assert state.contact_recognized.tolist() == [False, True]
    advance(state, True)
    assert state.contact_recognized.tolist() == [True, True]


def test_force_gate_retreat_and_unscaled_count_setting(states):
    single, parent = states
    state = single.SingleContactSweepState(1, 'cpu')
    state.contact_history.fill_(True)
    advance(state, True, force=.99)
    assert state.contact_recognized.item() and state.positive.item() == 0.
    torch.testing.assert_close(state._contact_factor(torch.tensor([True]), False), torch.ones(1))
    # Retreat is independent of the contact credit implementation.
    other = parent.SweepOnlyState(1, 'cpu')
    for item in (state, other):
        item.previous_distance.fill_(.98)
        item.finish(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), torch.zeros(1, 3),
                    torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
                    torch.tensor([1.1]), torch.ones(1))
    torch.testing.assert_close(state.retreat, other.retreat)
