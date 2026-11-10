# KNOW-VAULT

Census the storage vault inside a memory layer: how many slots, how many dead.

KNOW-VAULT treats memory keys as directions. A key can hold a distinct fact only if it does not overlap another key. The vault report estimates the number of mutually non-overlapping slots, the effective rank of the key matrix, and given activations which neurons never fire.

## What it does

- Slot estimate: greedy count of mutually non-overlapping key directions.
- Effective rank via the participation ratio of squared singular values.
- Dead neuron census and occupancy rate from observed activations.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from know_vault.vault import vault_report

keys = np.load("mlp_keys.npy")       # shape (n_neurons, dim)
acts = np.load("mlp_acts.npy")       # shape (n_samples, n_neurons)
report = vault_report(keys, activations=acts)
print(report["slot_estimate"], report["dead_neurons"])
```

## How it works

Keys are unit-normalized. A greedy pass keeps every key whose absolute cosine with all kept keys stays under the tolerance, giving a lower bound on distinct storage slots. Duplicated directions lose their capacity, which the effective rank exposes as well.

## Honest scope

- A slot estimate is a geometric proxy, not a count of real facts.
- Greedy order matters: the estimate is a lower bound, not the optimum packing.
- Tests use synthetic key matrices, so no model download is needed.

## License

MIT