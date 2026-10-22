"""Optional, fold-local deltaG input features shared by the LoRA trainers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


def load_deltag(path, name_col="name", value_col="deltaG_pred"):
    table = pd.read_csv(path, usecols=[name_col, value_col])
    if table[name_col].isna().any():
        raise ValueError("deltaG table contains missing names")
    table[name_col] = table[name_col].astype(str).str.strip().str.lower()
    table[value_col] = pd.to_numeric(table[value_col], errors="raise")
    if not np.isfinite(table[value_col]).all():
        raise ValueError("deltaG table contains missing or non-finite values")
    conflicts = table.groupby(name_col)[value_col].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting deltaG values: {conflicts[conflicts > 1].index[:5].tolist()}")
    return table.drop_duplicates(name_col).set_index(name_col)[value_col]


def lookup_deltag(names, mapping):
    values = names.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        raise ValueError(f"Missing deltaG for: {names[values.isna()].head().tolist()}")
    return values.astype(float)


def prepare_fold_deltag(mapping, abs_train, pair_train, single_eval, pair_eval,
                        name_col, wt_name_col, fold_dir, source):
    """Fit on unique training proteins only; never on validation/test proteins."""
    train_names = pd.concat([abs_train["name"], pair_train[name_col],
                             pair_train[wt_name_col]], ignore_index=True)
    train_names = train_names.astype(str).str.strip().str.lower().drop_duplicates()
    values = lookup_deltag(train_names, mapping)
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Cannot fit deltaG normalization on empty/non-finite training data")
    if std < 1e-8:
        std = 1.0
    for frame, key in [(abs_train, "name"), *[(f, name_col) for f in single_eval]]:
        frame["deltaG_input"] = lookup_deltag(frame[key], mapping)
    for frame in [pair_train, *pair_eval]:
        frame["deltaG_input"] = lookup_deltag(frame[name_col], mapping)
        frame["wt_deltaG_input"] = lookup_deltag(frame[wt_name_col], mapping)
    metadata = dict(mean=mean, std=std, n_train_unique=len(train_names),
                    source=str(Path(source).resolve()), feature="deltaG_pred",
                    head="masked_mean_hidden_concat_standardized_deltaG_mlp128")
    Path(fold_dir, "deltag_features.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return mean, std


def row_deltag(row, pair=False):
    if "deltaG_input" not in row.index:
        return {}
    result = {"deltaG": torch.tensor(float(row["deltaG_input"]), dtype=torch.float32)}
    if pair:
        result["wt_deltaG"] = torch.tensor(float(row["wt_deltaG_input"]), dtype=torch.float32)
    return result


def collate_deltag(batch):
    if "deltaG" not in batch[0]:
        return {}
    keys = ["deltaG", "wt_deltaG"] if "wt_deltaG" in batch[0] else ["deltaG"]
    return {key: torch.stack([item[key] for item in batch]) for key in keys}


class DeltaGFusionHead(nn.Module):
    """Accept raw deltaG; normalization travels with the model state_dict.

    use_deltag=False gives condition B: the identical MLP head with the deltaG
    input removed, so the head-capacity change can be measured on its own.
    """
    def __init__(self, hidden_size, mean, std, use_deltag=True):
        super().__init__()
        self.use_deltag = bool(use_deltag)
        self.register_buffer("deltaG_mean", torch.tensor(float(mean)))
        self.register_buffer("deltaG_std", torch.tensor(float(std)))
        in_dim = hidden_size + 1 if self.use_deltag else hidden_size
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, pooled, deltaG=None):
        if not self.use_deltag:
            return self.net(pooled).squeeze(-1)
        if deltaG is None:
            raise ValueError("This model requires raw deltaG input for every protein")
        values = torch.as_tensor(deltaG, device=pooled.device, dtype=pooled.dtype).reshape(-1, 1)
        if values.shape[0] != pooled.shape[0] or not torch.isfinite(values).all():
            raise ValueError("deltaG must contain one finite value per protein")
        z = (values - self.deltaG_mean) / self.deltaG_std
        return self.net(torch.cat([pooled, z], dim=-1)).squeeze(-1)
