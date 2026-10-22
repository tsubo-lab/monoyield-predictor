#!/usr/bin/env python

import os
import copy
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def infer_esmc_hidden_size(model_name):
    name = model_name.lower()

    if "6b" in name:
        return 2560
    elif "600m" in name:
        return 1152
    elif "300m" in name:
        return 960
    else:
        raise ValueError(
            f"Could not infer hidden_size from model_name={model_name}. "
            "Please specify --hidden_size manually."
        )


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


def topk_enrichment(y_true, y_pred, top_frac=0.1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    if n == 0:
        return np.nan, np.nan

    k = max(1, int(n * top_frac))

    true_top_idx = set(np.argsort(y_true)[-k:])
    pred_top_idx = set(np.argsort(y_pred)[-k:])

    overlap = len(true_top_idx & pred_top_idx)
    precision = overlap / k
    enrichment = precision / top_frac

    return enrichment, precision


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "top1_enrichment": np.nan,
            "top5_enrichment": np.nan,
            "top10_enrichment": np.nan,
            "top1_precision": np.nan,
            "top5_precision": np.nan,
            "top10_precision": np.nan,
            "pred_mean": np.nan,
            "pred_std": np.nan,
            "true_mean": np.nan,
            "true_std": np.nan,
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan

    pearson = float(safe_pearson(y_true, y_pred))
    spearman = float(safe_spearman(y_true, y_pred))

    enrich_1, prec_1 = topk_enrichment(y_true, y_pred, top_frac=0.01)
    enrich_5, prec_5 = topk_enrichment(y_true, y_pred, top_frac=0.05)
    enrich_10, prec_10 = topk_enrichment(y_true, y_pred, top_frac=0.10)

    return {
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "top1_enrichment": enrich_1,
        "top5_enrichment": enrich_5,
        "top10_enrichment": enrich_10,
        "top1_precision": prec_1,
        "top5_precision": prec_5,
        "top10_precision": prec_10,
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "true_mean": float(np.mean(y_true)),
        "true_std": float(np.std(y_true)),
    }


# ============================================================
# Dataset: single absolute value
# denovo + unique DMS WT
# ============================================================

class ProteinRegressionDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        seq_col="aa_seq",
        y_col="monomer_log_mgL",
        name_col="name",
        max_length=512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.y_col = y_col
        self.name_col = name_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = str(row[self.seq_col])
        y = float(row[self.y_col])

        enc = self.tokenizer(
            seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "name": str(row[self.name_col]),
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.float),
        }


def make_single_collate_fn(tokenizer):
    def collate_fn(batch):
        names = [x["name"] for x in batch]
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = torch.stack([x["labels"] for x in batch])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        return {
            "name": names,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


# ============================================================
# Dataset: DMS mutant-WT pair
# mutant is used only for delta loss
# ============================================================

class ProteinPairRegressionDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        seq_col="aa_seq",
        wt_seq_col="wt_seq",
        y_col="monomer_log_mgL",
        wt_y_col="wt_monomer_log_mgL",
        name_col="name",
        max_length=512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.wt_seq_col = wt_seq_col
        self.y_col = y_col
        self.wt_y_col = wt_y_col
        self.name_col = name_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mut_seq = str(row[self.seq_col])
        wt_seq = str(row[self.wt_seq_col])

        y_mut = float(row[self.y_col])
        y_wt = float(row[self.wt_y_col])

        mut_enc = self.tokenizer(
            mut_seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        wt_enc = self.tokenizer(
            wt_seq,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "name": str(row[self.name_col]),
            "mut_input_ids": torch.tensor(mut_enc["input_ids"], dtype=torch.long),
            "mut_attention_mask": torch.tensor(mut_enc["attention_mask"], dtype=torch.long),
            "wt_input_ids": torch.tensor(wt_enc["input_ids"], dtype=torch.long),
            "wt_attention_mask": torch.tensor(wt_enc["attention_mask"], dtype=torch.long),
            "y_mut": torch.tensor(y_mut, dtype=torch.float),
            "y_wt": torch.tensor(y_wt, dtype=torch.float),
        }


def make_pair_collate_fn(tokenizer):
    def collate_fn(batch):
        names = [x["name"] for x in batch]

        mut_input_ids = [x["mut_input_ids"] for x in batch]
        mut_attention_mask = [x["mut_attention_mask"] for x in batch]

        wt_input_ids = [x["wt_input_ids"] for x in batch]
        wt_attention_mask = [x["wt_attention_mask"] for x in batch]

        y_mut = torch.stack([x["y_mut"] for x in batch])
        y_wt = torch.stack([x["y_wt"] for x in batch])

        mut_input_ids = torch.nn.utils.rnn.pad_sequence(
            mut_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        mut_attention_mask = torch.nn.utils.rnn.pad_sequence(
            mut_attention_mask,
            batch_first=True,
            padding_value=0,
        )

        wt_input_ids = torch.nn.utils.rnn.pad_sequence(
            wt_input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        wt_attention_mask = torch.nn.utils.rnn.pad_sequence(
            wt_attention_mask,
            batch_first=True,
            padding_value=0,
        )

        return {
            "name": names,
            "mut_input_ids": mut_input_ids,
            "mut_attention_mask": mut_attention_mask,
            "wt_input_ids": wt_input_ids,
            "wt_attention_mask": wt_attention_mask,
            "y_mut": y_mut,
            "y_wt": y_wt,
        }

    return collate_fn


# ============================================================
# Residue-wise ESMC LoRA regressor
# token-wise scalar -> special-token-excluded residue mean
# ============================================================

class EsmCLoRAResidueMeanRegressor(nn.Module):
    def __init__(
        self,
        model_name="biohub/ESMC-600M",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        hidden_size=1152,
        target_modules=["out_proj"],
        model_revision=None,
    ):
        super().__init__()

        self.esm = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.esm = get_peft_model(self.esm, peft_config)

        self.residue_head = nn.Linear(hidden_size, 1)

    def make_residue_mask(self, input_ids, attention_mask, tokenizer):
        mask = attention_mask.bool()

        special_ids = set()

        for tok_id in [
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
        ]:
            if tok_id is not None:
                special_ids.add(tok_id)

        for tok_id in special_ids:
            mask = mask & (input_ids != tok_id)

        return mask

    def predict_value(self, input_ids, attention_mask, tokenizer):
        # For biohub/ESMC models, call LoRA-injected base model directly.
        outputs = self.esm.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
        else:
            hidden = outputs[0]

        # [B, L, H] -> [B, L]
        residue_logits = self.residue_head(hidden).squeeze(-1)

        residue_mask = self.make_residue_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        residue_logits_masked = residue_logits * residue_mask.to(residue_logits.dtype)

        denom = residue_mask.sum(dim=1).clamp(min=1).to(residue_logits.dtype)
        pred = residue_logits_masked.sum(dim=1) / denom

        return pred, residue_logits, residue_mask


# ============================================================
# Optimizer
# ============================================================

def make_optimizer(
    model,
    lora_lr=5e-5,
    head_lr=1e-4,
    weight_decay=0.01,
):
    lora_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "lora_" in name:
            lora_params.append(param)
        elif "residue_head" in name or "regression_head" in name:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    return optimizer


# ============================================================
# Loaders
# ============================================================

def make_single_loader(
    df,
    tokenizer,
    seq_col,
    y_col,
    name_col,
    max_length,
    batch_size,
    shuffle,
    num_workers,
):
    dataset = ProteinRegressionDataset(
        df=df,
        tokenizer=tokenizer,
        seq_col=seq_col,
        y_col=y_col,
        name_col=name_col,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_single_collate_fn(tokenizer),
        num_workers=num_workers,
        pin_memory=True,
    )


def make_pair_loader(
    df,
    tokenizer,
    seq_col,
    wt_seq_col,
    y_col,
    wt_y_col,
    name_col,
    max_length,
    batch_size,
    shuffle,
    num_workers,
):
    dataset = ProteinPairRegressionDataset(
        df=df,
        tokenizer=tokenizer,
        seq_col=seq_col,
        wt_seq_col=wt_seq_col,
        y_col=y_col,
        wt_y_col=wt_y_col,
        name_col=name_col,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_pair_collate_fn(tokenizer),
        num_workers=num_workers,
        pin_memory=True,
    )


# ============================================================
# Prepare one-stage data
# ============================================================

def make_denovo_split_and_filter_dms_by_rep(
    denovo_df,
    dms_df,
    test_group,
    groups,
    split_col="split",
    rep_col="rep",
    wt_rep_col=None,
    exclude_val_reps_from_dms_train=True,
):
    test_idx = groups.index(test_group)
    val_group = groups[(test_idx + 1) % len(groups)]

    train_groups = [
        g for g in groups
        if g not in [test_group, val_group]
    ]

    denovo_train = denovo_df[denovo_df[split_col].isin(train_groups)].copy()
    denovo_val = denovo_df[denovo_df[split_col] == val_group].copy()
    denovo_test = denovo_df[denovo_df[split_col] == test_group].copy()

    test_reps = set(denovo_test[rep_col].dropna().astype(str).unique())
    val_reps = set(denovo_val[rep_col].dropna().astype(str).unique())

    excluded_reps = set(test_reps)
    if exclude_val_reps_from_dms_train:
        excluded_reps |= val_reps

    dms_exclude_mask = dms_df[rep_col].astype(str).isin(excluded_reps)

    if wt_rep_col is not None and wt_rep_col in dms_df.columns:
        dms_exclude_mask = (
            dms_exclude_mask
            | dms_df[wt_rep_col].astype(str).isin(excluded_reps)
        )

    dms_train = dms_df[~dms_exclude_mask].copy()

    # Diagnostic DMS val/test by overlap with denovo val/test reps.
    dms_val_mask = dms_df[rep_col].astype(str).isin(val_reps)
    dms_test_mask = dms_df[rep_col].astype(str).isin(test_reps)

    if wt_rep_col is not None and wt_rep_col in dms_df.columns:
        dms_val_mask = (
            dms_val_mask
            | dms_df[wt_rep_col].astype(str).isin(val_reps)
        )
        dms_test_mask = (
            dms_test_mask
            | dms_df[wt_rep_col].astype(str).isin(test_reps)
        )

    dms_val = dms_df[dms_val_mask].copy()
    dms_test = dms_df[dms_test_mask].copy()

    return {
        "train_groups": train_groups,
        "val_group": val_group,
        "test_group": test_group,
        "test_reps": test_reps,
        "val_reps": val_reps,
        "excluded_reps": excluded_reps,
        "denovo_train": denovo_train,
        "denovo_val": denovo_val,
        "denovo_test": denovo_test,
        "dms_train": dms_train,
        "dms_val": dms_val,
        "dms_test": dms_test,
    }


def make_unique_dms_wt_abs_df(
    dms_df,
    wt_name_col="wt_name",
    wt_seq_col="wt_seq",
    wt_y_col="wt_monomer_log_mgL",
    rep_col="rep",
    out_y_col="monomer_log_mgL",
):
    cols = [wt_name_col, wt_seq_col, wt_y_col]

    optional_cols = []
    if rep_col in dms_df.columns:
        optional_cols.append(rep_col)

    wt_df = dms_df[cols + optional_cols].dropna(subset=cols).copy()

    wt_df = wt_df.drop_duplicates(subset=[wt_name_col]).copy()

    wt_df = wt_df.rename(
        columns={
            wt_name_col: "name",
            wt_seq_col: "aa_seq",
            wt_y_col: out_y_col,
        }
    )

    wt_df["source"] = "dms_wt"

    return wt_df


def make_abs_train_df(
    denovo_train_df,
    dms_train_df,
    name_col="name",
    seq_col="aa_seq",
    y_col="monomer_log_mgL",
    wt_name_col="wt_name",
    wt_seq_col="wt_seq",
    wt_y_col="wt_monomer_log_mgL",
    rep_col="rep",
):
    denovo_abs = denovo_train_df[[name_col, seq_col, y_col]].dropna().copy()
    denovo_abs = denovo_abs.rename(
        columns={
            name_col: "name",
            seq_col: "aa_seq",
            y_col: "monomer_log_mgL",
        }
    )
    denovo_abs["source"] = "denovo"

    dms_wt_abs = make_unique_dms_wt_abs_df(
        dms_df=dms_train_df,
        wt_name_col=wt_name_col,
        wt_seq_col=wt_seq_col,
        wt_y_col=wt_y_col,
        rep_col=rep_col,
        out_y_col="monomer_log_mgL",
    )

    abs_df = pd.concat(
        [
            denovo_abs[["name", "aa_seq", "monomer_log_mgL", "source"]],
            dms_wt_abs[["name", "aa_seq", "monomer_log_mgL", "source"]],
        ],
        ignore_index=True,
    )

    abs_df = abs_df.dropna(subset=["name", "aa_seq", "monomer_log_mgL"]).copy()

    return abs_df


def make_dms_pair_df(
    dms_df,
    seq_col="aa_seq",
    wt_seq_col="wt_seq",
    y_col="monomer_log_mgL",
    wt_y_col="wt_monomer_log_mgL",
):
    required = [seq_col, wt_seq_col, y_col, wt_y_col]
    pair_df = dms_df.dropna(subset=required).copy()

    # Remove WT self-pairs.
    pair_df = pair_df[
        pair_df[seq_col].astype(str) != pair_df[wt_seq_col].astype(str)
    ].copy()

    pair_df["delta_y"] = pair_df[y_col].astype(float) - pair_df[wt_y_col].astype(float)

    return pair_df


# ============================================================
# Train / predict
# ============================================================

def train_one_stage_abs_plus_delta_epoch_residue_mean(
    model,
    abs_loader,
    pair_loader,
    optimizer,
    device,
    tokenizer,
    grad_accum_steps=8,
    grad_clip=1.0,
    lambda_abs=1.0,
    lambda_delta=0.5,
    abs_loss_type="huber",
    delta_loss_type="mse",
):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    abs_iter = iter(abs_loader)
    pair_iter = iter(pair_loader)

    n_steps = max(len(abs_loader), len(pair_loader))

    losses = []
    abs_losses = []
    delta_losses = []

    for step in range(1, n_steps + 1):
        # -------------------------
        # Absolute batch:
        # denovo + unique DMS WT
        # -------------------------
        try:
            abs_batch = next(abs_iter)
        except StopIteration:
            abs_iter = iter(abs_loader)
            abs_batch = next(abs_iter)

        input_ids = abs_batch["input_ids"].to(device)
        attention_mask = abs_batch["attention_mask"].to(device)
        labels = abs_batch["labels"].to(device)

        pred_abs, _, _ = model.predict_value(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        if abs_loss_type == "huber":
            loss_abs = nn.functional.huber_loss(
                pred_abs,
                labels.float(),
                delta=1.0,
            )
        elif abs_loss_type == "mse":
            loss_abs = nn.functional.mse_loss(
                pred_abs,
                labels.float(),
            )
        else:
            raise ValueError(f"Unknown abs_loss_type: {abs_loss_type}")

        # -------------------------
        # Delta batch:
        # DMS mutant-WT pair
        # mutant absolute loss is NOT used
        # -------------------------
        try:
            pair_batch = next(pair_iter)
        except StopIteration:
            pair_iter = iter(pair_loader)
            pair_batch = next(pair_iter)

        mut_input_ids = pair_batch["mut_input_ids"].to(device)
        mut_attention_mask = pair_batch["mut_attention_mask"].to(device)

        wt_input_ids = pair_batch["wt_input_ids"].to(device)
        wt_attention_mask = pair_batch["wt_attention_mask"].to(device)

        y_mut = pair_batch["y_mut"].to(device)
        y_wt = pair_batch["y_wt"].to(device)

        pred_mut, _, _ = model.predict_value(
            input_ids=mut_input_ids,
            attention_mask=mut_attention_mask,
            tokenizer=tokenizer,
        )

        pred_wt, _, _ = model.predict_value(
            input_ids=wt_input_ids,
            attention_mask=wt_attention_mask,
            tokenizer=tokenizer,
        )

        pred_delta = pred_mut - pred_wt
        true_delta = y_mut - y_wt

        if delta_loss_type == "mse":
            loss_delta = nn.functional.mse_loss(
                pred_delta,
                true_delta,
            )
        elif delta_loss_type == "huber":
            loss_delta = nn.functional.huber_loss(
                pred_delta,
                true_delta,
                delta=1.0,
            )
        else:
            raise ValueError(f"Unknown delta_loss_type: {delta_loss_type}")

        loss = (
            lambda_abs * loss_abs
            + lambda_delta * loss_delta
        )

        loss_for_backward = loss / grad_accum_steps
        loss_for_backward.backward()

        if step % grad_accum_steps == 0 or step == n_steps:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))
        abs_losses.append(float(loss_abs.detach().cpu()))
        delta_losses.append(float(loss_delta.detach().cpu()))

    return {
        "loss": float(np.mean(losses)),
        "loss_abs": float(np.mean(abs_losses)),
        "loss_delta": float(np.mean(delta_losses)),
    }


@torch.no_grad()
def predict_single_loader_residue_mean(
    model,
    loader,
    device,
    tokenizer,
):
    model.eval()

    names = []
    y_true = []
    y_pred = []
    losses = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pred, _, _ = model.predict_value(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        loss = nn.functional.huber_loss(
            pred,
            labels.float(),
            delta=1.0,
        )

        names.extend(batch["name"])
        y_true.append(labels.detach().cpu())
        y_pred.append(pred.detach().cpu())
        losses.append(float(loss.detach().cpu()))

    y_true = torch.cat(y_true).numpy() if len(y_true) > 0 else np.array([])
    y_pred = torch.cat(y_pred).numpy() if len(y_pred) > 0 else np.array([])

    return {
        "name": names,
        "y_true": y_true,
        "y_pred": y_pred,
        "loss": float(np.mean(losses)) if len(losses) > 0 else np.nan,
    }


@torch.no_grad()
def predict_dms_pair_loader_residue_mean(
    model,
    loader,
    device,
    tokenizer,
):
    model.eval()

    names = []

    y_mut_true = []
    y_mut_pred = []

    y_wt_true = []
    y_wt_pred = []

    y_delta_true = []
    y_delta_pred = []

    for batch in loader:
        mut_input_ids = batch["mut_input_ids"].to(device)
        mut_attention_mask = batch["mut_attention_mask"].to(device)

        wt_input_ids = batch["wt_input_ids"].to(device)
        wt_attention_mask = batch["wt_attention_mask"].to(device)

        y_mut = batch["y_mut"].to(device)
        y_wt = batch["y_wt"].to(device)

        pred_mut, _, _ = model.predict_value(
            input_ids=mut_input_ids,
            attention_mask=mut_attention_mask,
            tokenizer=tokenizer,
        )

        pred_wt, _, _ = model.predict_value(
            input_ids=wt_input_ids,
            attention_mask=wt_attention_mask,
            tokenizer=tokenizer,
        )

        pred_delta = pred_mut - pred_wt
        true_delta = y_mut - y_wt

        names.extend(batch["name"])

        y_mut_true.append(y_mut.detach().cpu())
        y_mut_pred.append(pred_mut.detach().cpu())

        y_wt_true.append(y_wt.detach().cpu())
        y_wt_pred.append(pred_wt.detach().cpu())

        y_delta_true.append(true_delta.detach().cpu())
        y_delta_pred.append(pred_delta.detach().cpu())

    def cat(xs):
        return torch.cat(xs).numpy() if len(xs) > 0 else np.array([])

    return {
        "name": names,
        "y_mut_true": cat(y_mut_true),
        "y_mut_pred": cat(y_mut_pred),
        "y_wt_true": cat(y_wt_true),
        "y_wt_pred": cat(y_wt_pred),
        "y_delta_true": cat(y_delta_true),
        "y_delta_pred": cat(y_delta_pred),
    }


def run_one_stage_training(
    model,
    abs_train_loader,
    dms_pair_train_loader,
    denovo_val_loader,
    dms_val_loader,
    device,
    tokenizer,
    args,
):
    best_score = -np.inf
    best_state = None
    best_epoch = None
    no_improve = 0

    history = []

    optimizer = make_optimizer(
        model,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_stage_abs_plus_delta_epoch_residue_mean(
            model=model,
            abs_loader=abs_train_loader,
            pair_loader=dms_pair_train_loader,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            lambda_abs=args.lambda_abs,
            lambda_delta=args.lambda_delta,
            abs_loss_type=args.abs_loss_type,
            delta_loss_type=args.delta_loss_type,
        )

        denovo_val_pred = predict_single_loader_residue_mean(
            model=model,
            loader=denovo_val_loader,
            device=device,
            tokenizer=tokenizer,
        )

        denovo_val_metrics = calc_metrics(
            denovo_val_pred["y_true"],
            denovo_val_pred["y_pred"],
        )

        dms_delta_metrics = {}
        if dms_val_loader is not None:
            dms_val_pred = predict_dms_pair_loader_residue_mean(
                model=model,
                loader=dms_val_loader,
                device=device,
                tokenizer=tokenizer,
            )

            dms_delta_metrics = calc_metrics(
                dms_val_pred["y_delta_true"],
                dms_val_pred["y_delta_pred"],
            )

        if args.early_stop_metric == "rmse":
            score = -denovo_val_metrics["rmse"]
        else:
            score = denovo_val_metrics[args.early_stop_metric]

        print(
            f"[OneStageResidue] epoch={epoch:03d} "
            f"loss={train_loss['loss']:.5f} "
            f"abs={train_loss['loss_abs']:.5f} "
            f"delta={train_loss['loss_delta']:.5f} "
            f"denovo_val_pearson={denovo_val_metrics['pearson']:.4f} "
            f"denovo_val_spearman={denovo_val_metrics['spearman']:.4f} "
            f"denovo_val_top10={denovo_val_metrics['top10_enrichment']:.3f} "
            f"dms_val_delta_spearman={dms_delta_metrics.get('spearman', np.nan):.4f}"
        )

        history.append({
            "stage": "one_stage_abs_wt_delta_residue_mean",
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_loss.items()},
            **{f"denovo_val_{k}": v for k, v in denovo_val_metrics.items()},
            **{f"dms_val_delta_{k}": v for k, v in dms_delta_metrics.items()},
        })

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_info = {
        "best_stage": "one_stage_abs_wt_delta_residue_mean",
        "best_epoch": best_epoch,
        "best_score": best_score,
    }

    print("best_epoch:", best_epoch)
    print("best_score:", best_score)

    return model, pd.DataFrame(history), best_info


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--denovo_csv", required=True)
    parser.add_argument("--dms_csv", required=True)
    parser.add_argument("--out_dir", default="esmc_lora_one_stage_abs_wt_delta_residue_mean")

    parser.add_argument("--model_name", default="biohub/ESMC-600M")
    parser.add_argument("--hidden_size",type=int,default=None, help="Hidden size of ESMC model. If None, inferred from --model_name.",)

    parser.add_argument("--name_col", default="name")
    parser.add_argument("--seq_col", default="aa_seq")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--rep_col", default="rep")
    parser.add_argument("--wt_rep_col", default=None)

    parser.add_argument("--y_col", default="monomer_log_mgL")
    parser.add_argument("--wt_name_col", default="wt_name")
    parser.add_argument("--wt_seq_col", default="wt_seq")
    parser.add_argument("--wt_y_col", default="wt_monomer_log_mgL")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lora_lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--clip_y_min", type=float, default=None)
    parser.add_argument("--clip_y_max", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--lambda_abs", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=0.5)

    parser.add_argument("--abs_loss_type", default="huber", choices=["huber", "mse"])
    parser.add_argument("--delta_loss_type", default="mse", choices=["mse", "huber"])

    parser.add_argument(
        "--early_stop_metric",
        default="spearman",
        choices=[
            "rmse",
            "pearson",
            "spearman",
            "top1_enrichment",
            "top5_enrichment",
            "top10_enrichment",
        ],
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["group_1", "group_2", "group_3", "group_4", "group_5"],
    )

    parser.add_argument(
        "--run_folds",
        nargs="+",
        default=None,
        help="Folds to actually run. If None, run all folds.",
    )

    parser.add_argument(
        "--allow_dms_train_overlap_with_denovo_val_reps",
        action="store_true",
        help="If set, only denovo test reps are excluded from DMS train. By default, test+val reps are excluded.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    denovo_df = pd.read_csv(args.denovo_csv)
    dms_df = pd.read_csv(args.dms_csv)

    denovo_required = [
        args.name_col,
        args.seq_col,
        args.rep_col,
        args.split_col,
        args.y_col,
    ]

    dms_required = [
        args.name_col,
        args.seq_col,
        args.wt_name_col,
        args.wt_seq_col,
        args.rep_col,
        args.y_col,
        args.wt_y_col,
    ]

    if args.wt_rep_col is not None:
        dms_required.append(args.wt_rep_col)

    missing_denovo = [c for c in denovo_required if c not in denovo_df.columns]
    missing_dms = [c for c in dms_required if c not in dms_df.columns]

    if missing_denovo:
        raise ValueError(f"Missing columns in denovo_csv: {missing_denovo}")
    if missing_dms:
        raise ValueError(f"Missing columns in dms_csv: {missing_dms}")

    if args.clip_y_min is not None or args.clip_y_max is not None:
        denovo_df[args.y_col] = denovo_df[args.y_col].clip(
            lower=args.clip_y_min,
            upper=args.clip_y_max,
        )

        dms_df[args.y_col] = dms_df[args.y_col].clip(
            lower=args.clip_y_min,
            upper=args.clip_y_max,
        )

        dms_df[args.wt_y_col] = dms_df[args.wt_y_col].clip(
            lower=args.clip_y_min,
            upper=args.clip_y_max,
        )

    denovo_df = denovo_df[denovo_required].dropna().copy()
    dms_df = dms_df[dms_required].dropna().copy()

    denovo_df[args.split_col] = denovo_df[args.split_col].astype(str)
    denovo_df[args.rep_col] = denovo_df[args.rep_col].astype(str)
    dms_df[args.rep_col] = dms_df[args.rep_col].astype(str)

    if args.wt_rep_col is not None and args.wt_rep_col in dms_df.columns:
        dms_df[args.wt_rep_col] = dms_df[args.wt_rep_col].astype(str)

    print("denovo rows:", len(denovo_df))
    print(denovo_df[args.split_col].value_counts().sort_index())

    print("DMS rows:", len(dms_df))
    print("DMS unique mutant reps:", dms_df[args.rep_col].nunique())

    all_folds = args.folds

    if args.run_folds is None:
        run_folds = all_folds
    else:
        run_folds = args.run_folds

    for g in run_folds:
        if g not in all_folds:
            raise ValueError(f"{g} is not in --folds: {all_folds}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    exclude_val_reps_from_dms_train = (
        not args.allow_dms_train_overlap_with_denovo_val_reps
    )

    all_metric_records = []
    all_denovo_pred_records = []
    all_dms_pred_records = []

    for test_group in run_folds:
        print("=" * 100)
        print("Running fold:", test_group)

        split_data = make_denovo_split_and_filter_dms_by_rep(
            denovo_df=denovo_df,
            dms_df=dms_df,
            test_group=test_group,
            groups=all_folds,
            split_col=args.split_col,
            rep_col=args.rep_col,
            wt_rep_col=args.wt_rep_col,
            exclude_val_reps_from_dms_train=exclude_val_reps_from_dms_train,
        )

        train_groups = split_data["train_groups"]
        val_group = split_data["val_group"]

        denovo_train_df = split_data["denovo_train"]
        denovo_val_df = split_data["denovo_val"]
        denovo_test_df = split_data["denovo_test"]

        dms_train_df = split_data["dms_train"]
        dms_val_df = split_data["dms_val"]
        dms_test_df = split_data["dms_test"]

        print("train_groups:", train_groups)
        print("val_group:", val_group)
        print("test_group:", test_group)

        print("n_denovo_train:", len(denovo_train_df))
        print("n_denovo_val:", len(denovo_val_df))
        print("n_denovo_test:", len(denovo_test_df))

        print("n_denovo_test_reps:", len(split_data["test_reps"]))
        print("n_denovo_val_reps:", len(split_data["val_reps"]))
        print("n_excluded_reps_from_dms_train:", len(split_data["excluded_reps"]))

        print("n_dms_train_after_rep_exclusion:", len(dms_train_df))
        print("n_dms_val_overlap_denovo_val_reps:", len(dms_val_df))
        print("n_dms_test_overlap_denovo_test_reps:", len(dms_test_df))

        # -------------------------
        # Build one-stage training data
        # -------------------------
        abs_train_df = make_abs_train_df(
            denovo_train_df=denovo_train_df,
            dms_train_df=dms_train_df,
            name_col=args.name_col,
            seq_col=args.seq_col,
            y_col=args.y_col,
            wt_name_col=args.wt_name_col,
            wt_seq_col=args.wt_seq_col,
            wt_y_col=args.wt_y_col,
            rep_col=args.rep_col,
        )

        dms_pair_train_df = make_dms_pair_df(
            dms_train_df,
            seq_col=args.seq_col,
            wt_seq_col=args.wt_seq_col,
            y_col=args.y_col,
            wt_y_col=args.wt_y_col,
        )

        dms_pair_val_df = make_dms_pair_df(
            dms_val_df,
            seq_col=args.seq_col,
            wt_seq_col=args.wt_seq_col,
            y_col=args.y_col,
            wt_y_col=args.wt_y_col,
        )

        dms_pair_test_df = make_dms_pair_df(
            dms_test_df,
            seq_col=args.seq_col,
            wt_seq_col=args.wt_seq_col,
            y_col=args.y_col,
            wt_y_col=args.wt_y_col,
        )

        print("n_abs_train_denovo_plus_unique_dms_wt:", len(abs_train_df))
        print(abs_train_df["source"].value_counts())
        print("n_dms_pair_train_mutants:", len(dms_pair_train_df))
        print("n_dms_pair_val_mutants:", len(dms_pair_val_df))
        print("n_dms_pair_test_mutants:", len(dms_pair_test_df))

        if len(abs_train_df) == 0:
            raise ValueError("abs_train_df is empty.")
        if len(dms_pair_train_df) == 0:
            raise ValueError("dms_pair_train_df is empty.")

        fold_dir = os.path.join(args.out_dir, test_group)
        os.makedirs(fold_dir, exist_ok=True)

        abs_train_loader = make_single_loader(
            abs_train_df,
            tokenizer,
            seq_col="aa_seq",
            y_col="monomer_log_mgL",
            name_col="name",
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        denovo_val_loader = make_single_loader(
            denovo_val_df,
            tokenizer,
            seq_col=args.seq_col,
            y_col=args.y_col,
            name_col=args.name_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        denovo_test_loader = make_single_loader(
            denovo_test_df,
            tokenizer,
            seq_col=args.seq_col,
            y_col=args.y_col,
            name_col=args.name_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        dms_pair_train_loader = make_pair_loader(
            dms_pair_train_df,
            tokenizer,
            seq_col=args.seq_col,
            wt_seq_col=args.wt_seq_col,
            y_col=args.y_col,
            wt_y_col=args.wt_y_col,
            name_col=args.name_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        dms_pair_val_loader = None
        if len(dms_pair_val_df) > 0:
            dms_pair_val_loader = make_pair_loader(
                dms_pair_val_df,
                tokenizer,
                seq_col=args.seq_col,
                wt_seq_col=args.wt_seq_col,
                y_col=args.y_col,
                wt_y_col=args.wt_y_col,
                name_col=args.name_col,
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        dms_pair_test_loader = None
        if len(dms_pair_test_df) > 0:
            dms_pair_test_loader = make_pair_loader(
                dms_pair_test_df,
                tokenizer,
                seq_col=args.seq_col,
                wt_seq_col=args.wt_seq_col,
                y_col=args.y_col,
                wt_y_col=args.wt_y_col,
                name_col=args.name_col,
                max_length=args.max_length,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )

        hidden_size = args.hidden_size
        
        if hidden_size is None:
            hidden_size = infer_esmc_hidden_size(args.model_name)
        
        print("ESMC hidden_size:", hidden_size)
        
        model = EsmCLoRAResidueMeanRegressor(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            hidden_size=hidden_size,
            target_modules=["out_proj"],
        )

        model.to(device)
        model.esm.print_trainable_parameters()

        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"total trainable params including residue head: "
            f"{total_trainable:,} / {total_params:,} "
            f"({100 * total_trainable / total_params:.4f}%)"
        )

        model, history_df, best_info = run_one_stage_training(
            model=model,
            abs_train_loader=abs_train_loader,
            dms_pair_train_loader=dms_pair_train_loader,
            denovo_val_loader=denovo_val_loader,
            dms_val_loader=dms_pair_val_loader,
            device=device,
            tokenizer=tokenizer,
            args=args,
        )

        history_df["fold"] = test_group
        history_df["validation_group"] = val_group
        history_df["train_groups"] = ",".join(train_groups)
        history_df.to_csv(
            os.path.join(fold_dir, "history.csv"),
            index=False,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "best_info": best_info,
                "test_group": test_group,
                "val_group": val_group,
                "train_groups": train_groups,
                "excluded_reps": sorted(list(split_data["excluded_reps"])),
            },
            os.path.join(fold_dir, "best_model.pt"),
        )

        # -------------------------
        # Final denovo evaluation
        # -------------------------
        for split_name, loader in [
            ("validation", denovo_val_loader),
            ("test", denovo_test_loader),
        ]:
            pred = predict_single_loader_residue_mean(
                model=model,
                loader=loader,
                device=device,
                tokenizer=tokenizer,
            )

            metrics = calc_metrics(
                pred["y_true"],
                pred["y_pred"],
            )

            metrics["loss"] = pred["loss"]

            record = {
                "fold": test_group,
                "data_type": "denovo",
                "target_type": "absolute",
                "eval_split": split_name,
                "train_groups": ",".join(train_groups),
                "validation_group": val_group,
                "test_group": test_group,
                "best_stage": best_info["best_stage"],
                "best_epoch": best_info["best_epoch"],
                "best_score": best_info["best_score"],
                "n": len(pred["y_true"]),
            }

            record.update(metrics)
            all_metric_records.append(record)

            pred_df = pd.DataFrame({
                "fold": test_group,
                "data_type": "denovo",
                "eval_split": split_name,
                "name": pred["name"],
                "y_true": pred["y_true"],
                "y_pred": pred["y_pred"],
            })

            all_denovo_pred_records.append(pred_df)

        # -------------------------
        # Final DMS delta evaluation
        # -------------------------
        for split_name, loader in [
            ("validation_overlap_denovo_val_reps", dms_pair_val_loader),
            ("test_overlap_denovo_test_reps", dms_pair_test_loader),
        ]:
            if loader is None:
                continue

            pred = predict_dms_pair_loader_residue_mean(
                model=model,
                loader=loader,
                device=device,
                tokenizer=tokenizer,
            )

            mut_metrics = calc_metrics(
                pred["y_mut_true"],
                pred["y_mut_pred"],
            )
            wt_metrics = calc_metrics(
                pred["y_wt_true"],
                pred["y_wt_pred"],
            )
            delta_metrics = calc_metrics(
                pred["y_delta_true"],
                pred["y_delta_pred"],
            )

            for target_type, metrics in [
                ("mut_absolute_diagnostic", mut_metrics),
                ("wt_absolute_diagnostic", wt_metrics),
                ("delta", delta_metrics),
            ]:
                record = {
                    "fold": test_group,
                    "data_type": "dms",
                    "target_type": target_type,
                    "eval_split": split_name,
                    "train_groups": ",".join(train_groups),
                    "validation_group": val_group,
                    "test_group": test_group,
                    "best_stage": best_info["best_stage"],
                    "best_epoch": best_info["best_epoch"],
                    "best_score": best_info["best_score"],
                    "n": len(pred["name"]),
                }

                record.update(metrics)
                all_metric_records.append(record)

            dms_pred_df = pd.DataFrame({
                "fold": test_group,
                "data_type": "dms",
                "eval_split": split_name,
                "name": pred["name"],
                "y_mut_true": pred["y_mut_true"],
                "y_mut_pred": pred["y_mut_pred"],
                "y_wt_true": pred["y_wt_true"],
                "y_wt_pred": pred["y_wt_pred"],
                "y_delta_true": pred["y_delta_true"],
                "y_delta_pred": pred["y_delta_pred"],
            })

            all_dms_pred_records.append(dms_pred_df)

        # Partial saves
        metric_df = pd.DataFrame(all_metric_records)
        metric_df.to_csv(
            os.path.join(args.out_dir, "metrics_partial.csv"),
            index=False,
        )

        if len(all_denovo_pred_records) > 0:
            pd.concat(all_denovo_pred_records, ignore_index=True).to_csv(
                os.path.join(args.out_dir, "denovo_predictions_partial.csv"),
                index=False,
            )

        if len(all_dms_pred_records) > 0:
            pd.concat(all_dms_pred_records, ignore_index=True).to_csv(
                os.path.join(args.out_dir, "dms_predictions_partial.csv"),
                index=False,
            )

        del model
        torch.cuda.empty_cache()

    # Final saves
    metric_df = pd.DataFrame(all_metric_records)
    metric_path = os.path.join(args.out_dir, "metrics_5fold.csv")
    metric_df.to_csv(metric_path, index=False)

    print("saved:", metric_path)

    if len(all_denovo_pred_records) > 0:
        denovo_pred_df = pd.concat(all_denovo_pred_records, ignore_index=True)
        denovo_pred_path = os.path.join(args.out_dir, "denovo_predictions_5fold.csv")
        denovo_pred_df.to_csv(denovo_pred_path, index=False)
        print("saved:", denovo_pred_path)

    if len(all_dms_pred_records) > 0:
        dms_pred_df = pd.concat(all_dms_pred_records, ignore_index=True)
        dms_pred_path = os.path.join(args.out_dir, "dms_predictions_5fold.csv")
        dms_pred_df.to_csv(dms_pred_path, index=False)
        print("saved:", dms_pred_path)

    print("denovo test summary:")
    denovo_test_summary = (
        metric_df
        .query("data_type == 'denovo' and eval_split == 'test'")
        .agg({
            "rmse": ["mean", "std"],
            "r2": ["mean", "std"],
            "pearson": ["mean", "std"],
            "spearman": ["mean", "std"],
            "top1_enrichment": ["mean", "std"],
            "top5_enrichment": ["mean", "std"],
            "top10_enrichment": ["mean", "std"],
        })
    )
    print(denovo_test_summary)

    dms_delta_query = (
        "data_type == 'dms' and target_type == 'delta' "
        "and eval_split == 'test_overlap_denovo_test_reps'"
    )

    if len(metric_df.query(dms_delta_query)) > 0:
        print("DMS delta test-overlap summary:")
        dms_delta_summary = (
            metric_df
            .query(dms_delta_query)
            .agg({
                "rmse": ["mean", "std"],
                "r2": ["mean", "std"],
                "pearson": ["mean", "std"],
                "spearman": ["mean", "std"],
                "top1_enrichment": ["mean", "std"],
                "top5_enrichment": ["mean", "std"],
                "top10_enrichment": ["mean", "std"],
            })
        )
        print(dms_delta_summary)


if __name__ == "__main__":
    main()