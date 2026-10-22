#!/usr/bin/env python
"""
Ensemble prediction with the 5-fold SaProt-1.3B LoRA monomeric-yield models.

Mirrors scripts/predict_esmc_residue_pinned.py; the differences are all SaProt-specific:
  - EsmTokenizer / SaProtLoRAResidueMeanRegressor instead of the ESMC pair
  - target_modules must match training: ["query", "key", "value", "attention.output.dense"]
  - the input column is `saprot_seq` (aa + lowercase 3Di interleaved), NOT `aa_seq`
  - hidden_size is read from the model config when the checkpoint does not record one
"""

import os
import argparse
import numpy as np
import pandas as pd

import torch
from adapter_io import load_state_dict_flexible
from torch.utils.data import Dataset, DataLoader

from transformers import EsmTokenizer

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

# Model class comes from the trainer that produced these checkpoints
# (train_saprot_lora_mlp_head.py): masked-mean pooling -> PoolingHead.
from train_saprot_lora_mlp_head import (
    SaProtLoRAResidueMeanRegressor,
)


# Must match the training run; a different set silently produces a state_dict
# whose LoRA keys do not line up.
TARGET_MODULES = ["query", "key", "value", "attention.output.dense"]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


# ============================================================
# Metrics
# ============================================================

def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return stats.pearsonr(y_pred, y_true)[0]


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 3:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return stats.spearmanr(y_pred, y_true)[0]


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "pearson": float(safe_pearson(y_true, y_pred)),
        "spearman": float(safe_spearman(y_true, y_pred)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "true_mean": float(np.mean(y_true)),
        "true_std": float(np.std(y_true)),
    }


# ============================================================
# Dataset
# ============================================================

class ExternalSequenceDataset(Dataset):
    def __init__(self, df, tokenizer, name_col="name", seq_col="saprot_seq",
                 max_length=512, y_col=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.name_col = name_col
        self.seq_col = seq_col
        self.max_length = max_length
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])

        enc = self.tokenizer(
            seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {
            "name": str(row[self.name_col]),
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }

        if self.y_col is not None and self.y_col in row.index:
            item["label"] = torch.tensor(float(row[self.y_col]), dtype=torch.float)

        return item


def make_collate_fn(tokenizer, has_label=False):
    def collate_fn(batch):
        names = [x["name"] for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )

        out = {"name": names, "input_ids": input_ids, "attention_mask": attention_mask}
        if has_label:
            out["labels"] = torch.stack([x["label"] for x in batch])
        return out

    return collate_fn


def make_loader(df, tokenizer, name_col, seq_col, max_length, batch_size,
                num_workers, y_col=None):
    has_label = y_col is not None and y_col in df.columns

    dataset = ExternalSequenceDataset(
        df=df, tokenizer=tokenizer, name_col=name_col, seq_col=seq_col,
        max_length=max_length, y_col=y_col if has_label else None,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, has_label=has_label),
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, has_label


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_loader(model, loader, device, tokenizer):
    model.eval()

    names, preds, labels = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        pred, _, _ = model.predict_value(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        names.extend(batch["name"])
        preds.append(pred.detach().cpu())

        if "labels" in batch:
            labels.append(batch["labels"].detach().cpu())

    # numpy has no bfloat16, so bring predictions back to fp32 before converting
    preds = torch.cat(preds).float().numpy()
    labels = torch.cat(labels).numpy() if len(labels) > 0 else None

    return names, preds, labels


def load_model_from_checkpoint(checkpoint_path, device, fallback_model_name=None,
                               fallback_hidden_size=None, model_revision=None, dtype=None):
    # Always land the checkpoint on CPU first: map_location=device puts the fp32 state
    # dict straight onto the GPU, before any cast can shrink it.
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    saved_args = ckpt.get("args", {})

    model_name = saved_args.get("model_name", fallback_model_name)
    if model_name is None:
        raise ValueError("model_name is not in the checkpoint args. Pass --model_name.")

    # None is fine: SaProtLoRAResidueMeanRegressor then reads it off the model config,
    # exactly as training did.
    hidden_size = saved_args.get("hidden_size", fallback_hidden_size)

    lora_r = saved_args.get("lora_r", 4)
    lora_alpha = saved_args.get("lora_alpha", 8)
    lora_dropout = saved_args.get("lora_dropout", 0.15)

    print("loading:", checkpoint_path)
    print("  model_name:", model_name)
    print("  hidden_size:", hidden_size)
    print("  lora_r:", lora_r)

    # Read the head shape from the checkpoint rather than assuming it.
    architecture = saved_args.get("architecture")
    if architecture is None:
        raise ValueError(
            "checkpoint has no 'architecture' in args -- it was not produced by "
            "train_saprot_lora_mlp_head.py, so this predictor cannot reconstruct its head."
        )
    print("  architecture:", architecture,
          "| mlp_width:", saved_args.get("mlp_width"),
          "| attn_width:", saved_args.get("attn_width"))

    model = SaProtLoRAResidueMeanRegressor(
        model_name=model_name,
        model_revision=model_revision,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hidden_size=hidden_size,
        target_modules=TARGET_MODULES,
        architecture=architecture,
    )

    load_state_dict_flexible(model, ckpt["model_state_dict"], adapter_only=bool(ckpt.get("adapter_only", False)))
    print("  head params:", sum(p.numel() for p in model.regression_head.parameters()))

    # Cast on CPU, then move. Moving in fp32 first would need the full fp32 footprint
    # on the GPU, which is the thing bf16 is here to avoid.
    if dtype is not None and dtype != torch.float32:
        model = model.to(dtype)
    model.to(device)

    return model, saved_args


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--external_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--out_csv", default="external_ensemble_predictions_saprot.csv")

    parser.add_argument("--folds", nargs="+",
                        default=["group_1", "group_2", "group_3", "group_4", "group_5"])
    parser.add_argument("--pt_folder_names", nargs="+",
                        default=["group_1", "group_2", "group_3", "group_4", "group_5"])

    parser.add_argument("--checkpoint_name", default="best_model.pt")

    parser.add_argument("--name_col", default="name")
    parser.add_argument("--seq_col", default="saprot_seq")
    parser.add_argument("--y_col", default=None)

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model_revision", default=None,
                        help="Pin the HF revision, matching the one used for training.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--hidden_size", type=int, default=None)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_csv(args.external_csv)

    required = [args.name_col, args.seq_col]
    if args.y_col is not None:
        required.append(args.y_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in external_csv: {missing}")

    df = df.dropna(subset=[args.name_col, args.seq_col]).copy()

    # A '#' here means the row fell back to the unknown-structure encoding, which is a
    # different input distribution from the one these weights were trained on.
    n_hash = int(df[args.seq_col].astype(str).str.contains("#").sum())
    if n_hash:
        raise ValueError(
            f"{n_hash} rows in {args.seq_col} use unknown-structure '#' tokens; "
            "the training data was structure-aware, so refuse rather than predict."
        )

    # Truncation would silently drop the tail of a SaProt string, and because the string
    # is aa/3Di interleaved it is twice the residue count - easy to overrun unnoticed.
    too_long = df[df[args.seq_col].astype(str).str.len() + 2 > args.max_length]
    if len(too_long) > 0:
        print(f"WARNING: {len(too_long)} sequences exceed --max_length {args.max_length} "
              f"and will be truncated. Longest: {df[args.seq_col].astype(str).str.len().max()}")

    first_ckpt_path = os.path.join(args.model_dir, args.pt_folder_names[0], args.checkpoint_name)
    first_ckpt = torch.load(first_ckpt_path, map_location="cpu")
    first_args = first_ckpt.get("args", {})

    tokenizer_model_name = first_args.get("model_name", args.model_name)
    if tokenizer_model_name is None:
        raise ValueError("model_name is not found. Pass --model_name.")

    tokenizer = EsmTokenizer.from_pretrained(
        tokenizer_model_name,
        revision=args.model_revision,
    )

    loader, has_label = make_loader(
        df=df, tokenizer=tokenizer, name_col=args.name_col, seq_col=args.seq_col,
        max_length=args.max_length, batch_size=args.batch_size,
        num_workers=args.num_workers, y_col=args.y_col,
    )

    out_df = df.copy()
    fold_pred_cols = []

    if len(args.folds) != len(args.pt_folder_names):
        raise ValueError(
            "--folds and --pt_folder_names must have the same number of entries: "
            f"{len(args.folds)} != {len(args.pt_folder_names)}"
        )

    for fold, pt_folder_name in zip(args.folds, args.pt_folder_names):
        ckpt_path = os.path.join(args.model_dir, pt_folder_name, args.checkpoint_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model, saved_args = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            device=device,
            fallback_model_name=args.model_name,
            fallback_hidden_size=args.hidden_size,
            model_revision=args.model_revision,
            dtype=DTYPES[args.dtype],
        )

        names, preds, labels = predict_loader(
            model=model, loader=loader, device=device, tokenizer=tokenizer,
        )

        pred_col = f"saprot_pred_{fold}"
        out_df[pred_col] = preds
        fold_pred_cols.append(pred_col)

        del model
        torch.cuda.empty_cache()

    pred_mat = out_df[fold_pred_cols].values

    out_df["saprot_pred_mean"] = np.mean(pred_mat, axis=1)
    out_df["saprot_pred_std"] = np.std(pred_mat, axis=1)
    out_df["saprot_pred_min"] = np.min(pred_mat, axis=1)
    out_df["saprot_pred_max"] = np.max(pred_mat, axis=1)

    out_df.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

    if args.y_col is not None and args.y_col in out_df.columns:
        metrics = calc_metrics(out_df[args.y_col].values, out_df["saprot_pred_mean"].values)
        print("external ensemble metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
