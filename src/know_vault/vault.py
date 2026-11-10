"""Estimate knowledge-slot capacity and occupancy inside key matrices.

The vault model treats memory rows (keys) as directions: a key can
store a distinct fact only if it does not heavily overlap another
key. KNOW-VAULT estimates how many mutually non-overlapping slots
exist, the effective rank of the vault, and which neurons are dead
given observed activations. Pure NumPy.
"""

import numpy as np


def _unit_rows(keys):
    k = np.asarray(keys, dtype=float)
    if k.ndim != 2:
        raise ValueError("keys must have shape (n_neurons, dim)")
    norm = np.clip(np.linalg.norm(k, axis=1, keepdims=True), 1e-12, None)
    return k / norm


def effective_rank(keys):
    """Participation ratio of the squared singular values."""
    k = np.asarray(keys, dtype=float)
    if k.ndim != 2:
        raise ValueError("keys must have shape (n_neurons, dim)")
    s = np.linalg.svd(k, compute_uv=False)
    lam = np.clip(s, 1e-12, None) ** 2
    return float(np.sum(lam) ** 2 / np.sum(lam ** 2))


def slot_estimate(keys, cos_tol=0.5):
    """Greedy count of mutually non-overlapping key directions.

    Keys are scanned in order; a key is kept only when its absolute
    cosine with every already-kept key is at most cos_tol.
    """
    k = _unit_rows(keys)
    if k.shape[0] == 0:
        raise ValueError("keys must contain at least one row")
    kept = []
    for i in range(k.shape[0]):
        ok = True
        for j in kept:
            if abs(float(np.dot(k[i], k[j]))) > cos_tol:
                ok = False
                break
        if ok:
            kept.append(i)
    return len(kept)


def vault_report(keys, activations=None, cos_tol=0.5, act_threshold=1e-6):
    """Full vault report for one memory layer.

    Without activations the report covers capacity only. With
    ``activations`` of shape (n_samples, n_neurons) it also reports
    dead neurons (never above threshold) and the occupancy rate.
    """
    k = np.asarray(keys, dtype=float)
    if k.ndim != 2 or k.shape[0] == 0:
        raise ValueError("keys must have shape (n_neurons, dim)")
    report = {
        "n_neurons": int(k.shape[0]),
        "dim": int(k.shape[1]),
        "effective_rank": effective_rank(k),
        "slot_estimate": slot_estimate(k, cos_tol),
        "dead_neurons": [],
        "occupancy_rate": None,
    }
    if activations is not None:
        a = np.asarray(activations, dtype=float)
        if a.ndim != 2 or a.shape[1] != k.shape[0]:
            raise ValueError("activations must have shape (n_samples, n_neurons)")
        active = np.any(a > act_threshold, axis=0)
        report["dead_neurons"] = [int(i) for i in np.where(~active)[0]]
        report["occupancy_rate"] = float(np.mean(active))
    return report