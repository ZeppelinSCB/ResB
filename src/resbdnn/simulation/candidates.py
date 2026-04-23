from functools import lru_cache
from itertools import combinations

import numpy as np
import torch


@lru_cache(maxsize=None)
def candidate_combo_groups(n_t: int, s: int) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    groups = []
    for na in range(1, n_t // 2 + 1):
        combos = tuple(list(combinations(range(n_t), na))[:s])
        if len(combos) < s:
            raise ValueError(
                f"Not enough combinations for n_t={n_t}, na={na}, s={s}; got {len(combos)}"
            )
        groups.append((na, combos))
    return tuple(groups)


@lru_cache(maxsize=None)
def flat_candidate_combos(n_t: int, s: int) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    table = []
    for na_label, (na, combos) in enumerate(candidate_combo_groups(n_t, s)):
        for s_idx, combo in enumerate(combos):
            table.append((na_label, s_idx, combo))
    return tuple(table)


@lru_cache(maxsize=None)
def candidate_group_arrays(n_t: int, s: int) -> tuple[tuple[int, tuple[int, ...], np.ndarray], ...]:
    groups = []
    offset = 0
    for na, combos in candidate_combo_groups(n_t, s):
        candidate_indices = tuple(range(offset, offset + len(combos)))
        groups.append((na, candidate_indices, np.asarray(combos, dtype=np.int64)))
        offset += len(combos)
    return tuple(groups)


@lru_cache(maxsize=None)
def candidate_index_data(n_t: int, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = flat_candidate_combos(n_t, s)
    max_na = n_t // 2
    combo_idx = np.zeros((len(table), max_na), dtype=np.int64)
    combo_mask = np.zeros((len(table), max_na), dtype=np.float32)
    attrs = np.empty((len(table), 2), dtype=np.float32)
    na_den = max(max_na - 1, 1)
    s_den = max(s - 1, 1)

    for row_idx, (na_label, s_idx, combo) in enumerate(table):
        combo_idx[row_idx, : len(combo)] = combo
        combo_mask[row_idx, : len(combo)] = 1.0
        attrs[row_idx, 0] = np.float32(na_label / na_den)
        attrs[row_idx, 1] = np.float32(s_idx / s_den)

    return combo_idx, combo_mask, attrs


def candidate_index_tensors(
    n_t: int,
    s: int,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    combo_idx, combo_mask, attrs = candidate_index_data(n_t, s)
    return (
        torch.as_tensor(combo_idx, dtype=torch.long, device=device),
        torch.as_tensor(combo_mask, dtype=torch.float32, device=device),
        torch.as_tensor(attrs, dtype=torch.float32, device=device),
    )
