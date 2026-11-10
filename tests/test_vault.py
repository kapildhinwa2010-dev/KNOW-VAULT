import numpy as np
import pytest

from know_vault.vault import effective_rank, slot_estimate, vault_report


def random_keys(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def duplicated_keys(seed):
    base = random_keys(8, 128, seed)
    return np.repeat(base, 2, axis=0)


def test_random_keys_have_capacity():
    keys = random_keys(32, 128, seed=0)
    assert slot_estimate(keys) >= 24
    assert effective_rank(keys) > 20.0


def test_duplicated_keys_lose_capacity():
    keys = duplicated_keys(seed=1)
    # every direction is stored twice, so at most 8 slots survive
    assert slot_estimate(keys) <= 8
    rand = random_keys(32, 128, seed=0)
    assert slot_estimate(keys) < slot_estimate(rand)
    assert effective_rank(keys) < effective_rank(rand)


def test_slot_tolerance_matters():
    keys = duplicated_keys(seed=1)
    strict = slot_estimate(keys, cos_tol=0.1)
    loose = slot_estimate(keys, cos_tol=0.99)
    assert strict <= loose


def test_vault_report_without_activations():
    keys = random_keys(16, 64, seed=2)
    r = vault_report(keys)
    assert r["n_neurons"] == 16
    assert r["dim"] == 64
    assert r["slot_estimate"] == slot_estimate(keys)
    assert r["dead_neurons"] == []
    assert r["occupancy_rate"] is None


def test_vault_report_dead_neurons():
    keys = random_keys(16, 64, seed=3)
    acts = np.abs(random_keys(10, 16, seed=4)) + 0.1
    acts[:, 0] = 0.0
    acts[:, 1] = 0.0
    r = vault_report(keys, activations=acts)
    assert r["dead_neurons"] == [0, 1]
    assert r["occupancy_rate"] == pytest.approx(14.0 / 16.0)


def test_errors():
    with pytest.raises(ValueError):
        effective_rank(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        slot_estimate(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        vault_report(np.zeros((4, 2)), activations=np.zeros((3, 5)))