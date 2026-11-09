"""Refresh, stage invalidation and bounded reuse bookkeeping without Isaac Sim."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline/mdp/common/grouped_reset_pool.py'
spec = importlib.util.spec_from_file_location('grouped_reset_pool', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fill_reuse_refresh_and_curriculum_invalidation():
    bank = module.GroupedResetPool(2, 2, 3, 42)
    for g in (0, 1):
        for slot in (0, 1):
            assert bank.select(g, 'stage0') == (slot, True)
            bank.entries[g][slot] = (g, slot)
        assert bank.select(g, 'stage0')[1] is False
        assert bank.select(g, 'stage0')[1] is False
        assert bank.select(g, 'stage0') == (0, True)
    assert bank.select(0, 'stage1') == (0, True)
    assert bank.entries[1] == []
    assert bank.select(1, 'stage1') == (0, True)
    assert bank.select(0, 'stage0') == (0, True)  # rewind invalidates too


def test_seed_reproduces_selection_and_partial_groups_do_not_advance():
    a = module.GroupedResetPool(3, 3, 4, 7)
    b = module.GroupedResetPool(3, 3, 4, 7)
    for _ in range(30):
        assert a.select(2, 0) == b.select(2, 0)
    assert a.calls[:2] == [0, 0]
    assert a.entries[:2] == [[], []]
    assert len(a.entries[2]) == 3
