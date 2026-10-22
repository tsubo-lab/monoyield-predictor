#!/usr/bin/env python

import os
import copy
import random
import argparse
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_esmc_hidden_size(model_name):
    name = model_name.lower()
    if "6b" in name:
        return 2560
    if "600m" in name:
        return 1152
    if "300m" in name:
        return 960
    raise ValueError(f"Could not infer hidden_size from model_name={model_name}. Use --hidden_size.")


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(stats.pearsonr(y_pred, y_true)[0])


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(stats.spearmanr(y_pred, y_true)[0])


def topk_enrichment(y_true, y_pred, top_frac=0.1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan
    k = max(1, int(n * top_frac))
    true_top_idx = set(np.argsort(y_true)[-k:])
    pred_top_idx = set(np.argsort(y_pred)[-k:])
    precision = len(true_top_idx & pred_top_idx) / k
    return float(precision / top_frac), float(precision)


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {k: np.nan for k in [
            "rmse", "r2", "pearson", "spearman", "top1_enrichment",
            "top5_enrichment", "top10_enrichment", "top1_precision",
            "top5_precision", "top10_precision", "pred_mean", "pred_std",
            "true_mean", "true_std", "slope_pred_vs_true"
        ]}
    enrich_1, prec_1 = topk_enrichment(y_true, y_pred, 0.01)
    enrich_5, prec_5 = topk_enrichment(y_true, y_pred, 0.05)
    enrich_10, prec_10 = topk_enrichment(y_true, y_pred, 0.10)
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        slope = float(np.polyfit(y_true, y_pred, 1)[0])
    else:
        slope = np.nan
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
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
        "slope_pred_vs_true": slope,
    }


class ProteinRegressionDataset(Dataset):
    def __init__(self, df, tokenizer, seq_col, y_col, name_col, max_length, y_mean=0.0, y_std=1.0):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.y_col = y_col
        self.name_col = name_col
        self.max_length = max_length
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 0 else 1.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        y_raw = float(row[self.y_col])
        y = (y_raw - self.y_mean) / self.y_std
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
            "labels_raw": torch.tensor(y_raw, dtype=torch.float),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        return {
            "name": [x["name"] for x in batch],
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "labels": torch.stack([x["labels"] for x in batch]),
            "labels_raw": torch.stack([x["labels_raw"] for x in batch]),
        }
    return collate_fn


class EsmCLoRAResidueMeanRegressor(nn.Module):
    def __init__(
        self,
        model_name="biohub/ESMC-600M",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        hidden_size=1152,
        target_modules=("out_proj",),
    ):
        super().__init__()
        self.esm = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
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
        outputs = self.esm.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
        else:
            hidden = outputs[0]
        residue_logits = self.residue_head(hidden).squeeze(-1)
        residue_mask = self.make_residue_mask(input_ids, attention_mask, tokenizer)
        residue_logits_masked = residue_logits * residue_mask.to(residue_logits.dtype)
        denom = residue_mask.sum(dim=1).clamp(min=1).to(residue_logits.dtype)
        pred = residue_logits_masked.sum(dim=1) / denom
        return pred, residue_logits, residue_mask


def make_optimizer(model, lora_lr, head_lr, weight_decay, train_lora=True, train_head=True):
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "lora_" in name and train_lora:
            param.requires_grad = True
            lora_params.append(param)
        elif ("residue_head" in name or "regression_head" in name) and train_head:
            param.requires_grad = True
            head_params.append(param)
    groups = []
    if len(lora_params) > 0:
        groups.append({"params": lora_params, "lr": lora_lr})
    if len(head_params) > 0:
        groups.append({"params": head_params, "lr": head_lr})
    if len(groups) == 0:
        raise ValueError("No trainable parameters. Set --train_lora or --train_head.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def make_loader(df, tokenizer, args, batch_size, shuffle, y_mean, y_std):
    ds = ProteinRegressionDataset(
        df=df,
        tokenizer=tokenizer,
        seq_col=args.seq_col,
        y_col=args.y_col,
        name_col=args.name_col,
        max_length=args.max_length,
        y_mean=y_mean,
        y_std=y_std,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=args.num_workers,
        pin_memory=True,
    )


def regression_loss(pred, labels, loss_type, huber_delta):
    if loss_type == "mse":
        return nn.functional.mse_loss(pred, labels.float())
    if loss_type == "huber":
        return nn.functional.huber_loss(pred, labels.float(), delta=huber_delta)
    if loss_type == "mae":
        return nn.functional.l1_loss(pred, labels.float())
    raise ValueError(f"Unknown loss_type: {loss_type}")


def std_match_loss(pred, labels, eps=1e-6):
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    pred_std = pred.std(unbiased=False)
    label_std = labels.float().std(unbiased=False)
    return (pred_std - label_std).pow(2) / (label_std.detach().pow(2) + eps)


def train_epoch(model, loader, optimizer, device, tokenizer, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    base_losses = []
    range_losses = []
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pred, _, _ = model.predict_value(input_ids, attention_mask, tokenizer)
        base = regression_loss(pred, labels, args.loss_type, args.huber_delta)
        range_penalty = std_match_loss(pred, labels) if args.range_loss_weight > 0 else pred.new_tensor(0.0)
        loss = base + args.range_loss_weight * range_penalty
        (loss / args.grad_accum_steps).backward()
        if step % args.grad_accum_steps == 0 or step == len(loader):
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        base_losses.append(float(base.detach().cpu()))
        range_losses.append(float(range_penalty.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "base_loss": float(np.mean(base_losses)),
        "range_loss": float(np.mean(range_losses)),
    }


@torch.no_grad()
def predict_loader(model, loader, device, tokenizer, y_mean, y_std):
    model.eval()
    names = []
    y_true = []
    y_pred = []
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        pred_norm, _, _ = model.predict_value(input_ids, attention_mask, tokenizer)
        loss = nn.functional.mse_loss(pred_norm, labels.float())
        pred_raw = pred_norm.detach().cpu().numpy() * y_std + y_mean
        labels_raw = batch["labels_raw"].detach().cpu().numpy()
        names.extend(batch["name"])
        y_true.extend(labels_raw.tolist())
        y_pred.extend(pred_raw.tolist())
        losses.append(float(loss.detach().cpu()))
    return {
        "name": names,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "loss_norm_mse": float(np.mean(losses)) if len(losses) > 0 else np.nan,
    }


def fit_linear_calibration(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_pred) == 0:
        return 1.0, 0.0
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)



def expand_split_values(values):
    """Allow both space-separated argparse values and compact group1+group2 syntax."""
    if values is None:
        return []
    out = []
    for value in values:
        for part in str(value).replace(",", "+").split("+"):
            part = part.strip()
            if part:
                out.append(part)
    return out


def safe_filename(value):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "value"


def split_sort_key(value):
    value = str(value)
    m = re.search(r"(\d+)$", value)
    if m:
        prefix = value[:m.start()]
        return (prefix, int(m.group(1)), value)
    return (value, -1, value)


def prepare_dataframe(df, args):
    df = df.dropna(subset=[args.name_col, args.seq_col, args.y_col]).copy()
    df[args.y_col] = df[args.y_col].astype(float)
    if args.clip_y_min is not None or args.clip_y_max is not None:
        df[args.y_col] = df[args.y_col].clip(args.clip_y_min, args.clip_y_max)
    return df


def split_dataframe_by_values(df, args, train_values, val_values, test_values):
    if args.split_col is None or args.split_col not in df.columns:
        raise ValueError(f"split_col={args.split_col} is not present in the dataframe.")

    train_values = expand_split_values(train_values)
    val_values = expand_split_values(val_values)
    test_values = expand_split_values(test_values)

    s = df[args.split_col].astype(str)
    train_df = df[s.isin(train_values)].copy()
    val_df = df[s.isin(val_values)].copy()
    test_df = df[s.isin(test_values)].copy()

    if len(train_df) == 0:
        raise ValueError(f"train_df is empty. train_values={train_values}")
    if len(val_df) == 0:
        raise ValueError(f"val_df is empty. val_values={val_values}")
    if len(test_df) == 0:
        raise ValueError(f"test_df is empty. test_values={test_values}")

    overlap = (
        set(train_df[args.name_col]) & set(val_df[args.name_col])
        or set(train_df[args.name_col]) & set(test_df[args.name_col])
        or set(val_df[args.name_col]) & set(test_df[args.name_col])
    )
    if overlap:
        raise ValueError("Some names appear in multiple splits. Check split assignments.")

    return train_df, val_df, test_df


def random_split_dataframe(df, args):
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(round(len(df) * args.test_frac))
    n_val = int(round(len(df) * args.val_frac))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    if len(train_df) == 0:
        raise ValueError("train_df is empty. Check --test_frac/--val_frac.")
    if len(val_df) == 0:
        raise ValueError("val_df is empty. Check --val_frac.")
    return train_df, val_df, test_df


def infer_5fold_values(df, args):
    if args.split_col is None or args.split_col not in df.columns:
        raise ValueError(f"--run_5fold requires split_col={args.split_col} in nanobody_csv.")

    observed = sorted(df[args.split_col].dropna().astype(str).unique().tolist(), key=split_sort_key)
    observed_set = set(observed)

    if args.fold_values is not None:
        fold_values = expand_split_values(args.fold_values)
        missing = [v for v in fold_values if v not in observed_set]
        if missing:
            raise ValueError(
                f"--fold_values contains values not present in {args.split_col}: {missing}. "
                f"Observed values: {observed}"
            )
        if len(fold_values) != 5:
            raise ValueError(f"Expected exactly 5 fold values, got {len(fold_values)}: {fold_values}")
        return fold_values

    candidates = [
        [f"group{i}" for i in range(1, 6)],
        [f"group_{i}" for i in range(1, 6)],
        [str(i) for i in range(1, 6)],
    ]
    for candidate in candidates:
        if set(candidate).issubset(observed_set):
            return candidate

    if len(observed) == 5:
        return observed

    counts = df[args.split_col].astype(str).value_counts().to_dict()
    raise ValueError(
        "Could not infer 5 fold values automatically. "
        "Pass them explicitly, for example: --fold_values group1 group2 group3 group4 group5. "
        f"Observed split counts: {counts}"
    )


def make_5fold_plan(fold_values, val_fold_offset=1):
    if len(fold_values) != 5:
        raise ValueError(f"Expected 5 fold values, got {len(fold_values)}")
    if val_fold_offset % len(fold_values) == 0:
        raise ValueError("--val_fold_offset cannot point validation to the same fold as test.")

    plan = []
    n = len(fold_values)
    for i, test_value in enumerate(fold_values):
        val_value = fold_values[(i + val_fold_offset) % n]
        train_values = [v for v in fold_values if v not in {test_value, val_value}]
        fold_name = f"fold_{i + 1}_test_{safe_filename(test_value)}"
        plan.append({
            "fold_index": i,
            "fold_num": i + 1,
            "fold_name": fold_name,
            "train_values": train_values,
            "val_values": [val_value],
            "test_values": [test_value],
        })
    return plan


def pick_score(metrics, early_stop_metric):
    if early_stop_metric in ["rmse", "loss_norm_mse"]:
        return -metrics[early_stop_metric]
    return metrics[early_stop_metric]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_init_pt(args, fold_info=None):
    if args.init_pt_template is None:
        return args.init_pt
    if fold_info is None:
        raise ValueError("--init_pt_template can only be used with --run_5fold.")
    test_value = fold_info["test_values"][0]
    return args.init_pt_template.format(
        fold=fold_info["fold_num"],
        fold_num=fold_info["fold_num"],
        fold_index=fold_info["fold_index"],
        test_value=test_value,
        test_split=test_value,
        test_value_safe=safe_filename(test_value),
        test_split_safe=safe_filename(test_value),
    )


def train_one_split(args, df, out_dir, train_values, val_values, test_values, fold_info=None):
    os.makedirs(out_dir, exist_ok=True)

    seed_offset = 0 if fold_info is None else fold_info["fold_index"]
    set_seed(args.seed + seed_offset)

    init_pt = resolve_init_pt(args, fold_info)
    print("init_pt:", init_pt)
    ckpt = load_checkpoint(init_pt)
    old_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    model_name = args.model_name or old_args.get("model_name", "biohub/ESMC-600M")
    hidden_size = args.hidden_size or old_args.get("hidden_size") or infer_esmc_hidden_size(model_name)
    lora_r = args.lora_r if args.lora_r is not None else old_args.get("lora_r", 4)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else old_args.get("lora_alpha", 8)
    lora_dropout = args.lora_dropout if args.lora_dropout is not None else old_args.get("lora_dropout", 0.05)

    print("model_name:", model_name)
    print("hidden_size:", hidden_size)
    print("lora_r/lora_alpha/lora_dropout:", lora_r, lora_alpha, lora_dropout)
    print("train_values:", train_values)
    print("val_values:", val_values)
    print("test_values:", test_values)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = EsmCLoRAResidueMeanRegressor(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hidden_size=hidden_size,
        target_modules=args.target_modules,
    )

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    load_result = model.load_state_dict(state_dict, strict=False)
    print("missing_keys:", len(load_result.missing_keys))
    print("unexpected_keys:", len(load_result.unexpected_keys))
    if len(load_result.missing_keys) > 0:
        print("first_missing_keys:", load_result.missing_keys[:10])
    if len(load_result.unexpected_keys) > 0:
        print("first_unexpected_keys:", load_result.unexpected_keys[:10])

    train_df, val_df, test_df = split_dataframe_by_values(df, args, train_values, val_values, test_values)
    y_mean = float(train_df[args.y_col].mean())
    y_std = float(train_df[args.y_col].std(ddof=0))
    if y_std == 0 or np.isnan(y_std):
        y_std = 1.0

    print("n_train/n_val/n_test:", len(train_df), len(val_df), len(test_df))
    print("train_y_mean/train_y_std:", y_mean, y_std)

    train_df.to_csv(os.path.join(out_dir, "train_rows.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val_rows.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_rows.csv"), index=False)

    train_loader = make_loader(train_df, tokenizer, args, args.batch_size, True, y_mean, y_std)
    val_loader = make_loader(val_df, tokenizer, args, args.batch_size, False, y_mean, y_std)
    test_loader = make_loader(test_df, tokenizer, args, args.batch_size, False, y_mean, y_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model.to(device)
    optimizer = make_optimizer(
        model,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        train_lora=args.train_lora,
        train_head=args.train_head,
    )
    if hasattr(model.esm, "print_trainable_parameters"):
        model.esm.print_trainable_parameters()
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params: {total_trainable:,} / {total_params:,} ({100 * total_trainable / total_params:.4f}%)")

    best_score = -np.inf
    best_state = None
    best_epoch = None
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer, args)
        val_pred = predict_loader(model, val_loader, device, tokenizer, y_mean, y_std)
        val_metrics = calc_metrics(val_pred["y_true"], val_pred["y_pred"])
        val_metrics["loss_norm_mse"] = val_pred["loss_norm_mse"]
        score = pick_score(val_metrics, args.early_stop_metric)

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_loss.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        if fold_info is not None:
            record.update({
                "fold": fold_info["fold_name"],
                "fold_num": fold_info["fold_num"],
                "test_split_value": test_values[0],
                "val_split_value": val_values[0],
            })
        history.append(record)
        fold_label = "" if fold_info is None else f"[{fold_info['fold_name']}] "
        print(
            f"{fold_label}[NanobodyFT] epoch={epoch:03d} "
            f"loss={train_loss['loss']:.5f} "
            f"base={train_loss['base_loss']:.5f} "
            f"range={train_loss['range_loss']:.5f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_pearson={val_metrics['pearson']:.4f} "
            f"val_spearman={val_metrics['spearman']:.4f} "
            f"val_pred_std={val_metrics['pred_std']:.4f} "
            f"val_true_std={val_metrics['true_std']:.4f} "
            f"val_slope={val_metrics['slope_pred_vs_true']:.3f}"
        )

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

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    val_pred = predict_loader(model, val_loader, device, tokenizer, y_mean, y_std)
    val_metrics = calc_metrics(val_pred["y_true"], val_pred["y_pred"])
    val_metrics["loss_norm_mse"] = val_pred["loss_norm_mse"]

    cal_a, cal_b = (1.0, 0.0)
    if not args.no_linear_calibration:
        cal_a, cal_b = fit_linear_calibration(val_pred["y_true"], val_pred["y_pred"])
        val_pred["y_pred_calibrated"] = cal_a * val_pred["y_pred"] + cal_b
        val_cal_metrics = calc_metrics(val_pred["y_true"], val_pred["y_pred_calibrated"])
    else:
        val_pred["y_pred_calibrated"] = val_pred["y_pred"]
        val_cal_metrics = val_metrics.copy()

    all_metrics = [
        {"split": "validation", "calibrated": False, "n": len(val_pred["y_true"]), **val_metrics},
        {"split": "validation", "calibrated": True, "n": len(val_pred["y_true"]), **val_cal_metrics},
    ]

    val_pred_df = pd.DataFrame(val_pred)
    val_pred_df.insert(0, "split", "validation")
    val_pred_df.to_csv(os.path.join(out_dir, "validation_predictions.csv"), index=False)

    test_pred = predict_loader(model, test_loader, device, tokenizer, y_mean, y_std)
    test_metrics = calc_metrics(test_pred["y_true"], test_pred["y_pred"])
    test_pred["y_pred_calibrated"] = cal_a * test_pred["y_pred"] + cal_b
    test_cal_metrics = calc_metrics(test_pred["y_true"], test_pred["y_pred_calibrated"])
    all_metrics.extend([
        {"split": "test", "calibrated": False, "n": len(test_pred["y_true"]), **test_metrics},
        {"split": "test", "calibrated": True, "n": len(test_pred["y_true"]), **test_cal_metrics},
    ])
    test_pred_df = pd.DataFrame(test_pred)
    test_pred_df.insert(0, "split", "test")
    test_pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    metric_df = pd.DataFrame(all_metrics)
    if fold_info is not None:
        metric_df.insert(0, "fold", fold_info["fold_name"])
        metric_df.insert(1, "fold_num", fold_info["fold_num"])
        metric_df.insert(2, "test_split_value", test_values[0])
        metric_df.insert(3, "val_split_value", val_values[0])
        train_label = "+".join(train_values)
        metric_df.insert(4, "train_split_values", train_label)
        for pred_df in [val_pred_df, test_pred_df]:
            pred_df.insert(0, "fold", fold_info["fold_name"])
            pred_df.insert(1, "fold_num", fold_info["fold_num"])
            pred_df.insert(2, "test_split_value", test_values[0])
            pred_df.insert(3, "val_split_value", val_values[0])
            pred_df.insert(4, "train_split_values", train_label)

    metric_df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(metric_df)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "source_checkpoint_args": old_args,
            "fold_info": fold_info,
            "init_pt_used": init_pt,
            "split_values": {
                "train": train_values,
                "validation": val_values,
                "test": test_values,
            },
            "best_info": {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "early_stop_metric": args.early_stop_metric,
            },
            "target_normalization": {"y_mean": y_mean, "y_std": y_std},
            "linear_calibration": {"a": cal_a, "b": cal_b},
        },
        os.path.join(out_dir, "best_nanobody_finetuned_model.pt"),
    )
    print("saved:", os.path.join(out_dir, "best_nanobody_finetuned_model.pt"))

    del model, optimizer, train_loader, val_loader, test_loader, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metric_df, val_pred_df, test_pred_df, history_df


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_pt", required=True, help="Path to previous best_model.pt. Used for all folds unless --init_pt_template is set.")
    parser.add_argument("--init_pt_template", default=None, help="Optional template for fold-specific checkpoints, e.g. outputs/pretrain/group_{fold_num}/best_model.pt")
    parser.add_argument("--nanobody_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--model_name", default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--name_col", default="name")
    parser.add_argument("--seq_col", default="aa_seq")
    parser.add_argument("--y_col", default="monomer_log_mgL")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--train_split_values", nargs="+", default=["train"])
    parser.add_argument("--val_split_values", nargs="+", default=["val", "validation"])
    parser.add_argument("--test_split_values", nargs="+", default=["test"])
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)

    parser.add_argument("--run_5fold", action="store_true", help="Run 5 fine-tuning jobs using split_col groups.")
    parser.add_argument("--fold_values", nargs="+", default=None, help="Five split values in fold order. If omitted, group1-group5 or group_1-group_5 are inferred.")
    parser.add_argument("--val_fold_offset", type=int, default=1, help="Validation fold is offset from test fold by this many positions. Default: next fold.")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--target_modules", nargs="+", default=["out_proj"])

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_lora", action="store_true", help="Fine-tune LoRA as well as the head.")
    parser.add_argument("--train_head", action="store_true", help="Train regression head. Usually set this.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss_type", default="mse", choices=["mse", "huber", "mae"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--range_loss_weight", type=float, default=0.1)
    parser.add_argument("--early_stop_metric", default="rmse", choices=["rmse", "r2", "pearson", "spearman", "top1_enrichment", "top5_enrichment", "top10_enrichment", "loss_norm_mse"])
    parser.add_argument("--clip_y_min", type=float, default=None)
    parser.add_argument("--clip_y_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_linear_calibration", action="store_true")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not args.train_lora and not args.train_head:
        # Safe default for fine-tuning from a previous checkpoint.
        args.train_head = True
        args.train_lora = True

    args.train_split_values = expand_split_values(args.train_split_values)
    args.val_split_values = expand_split_values(args.val_split_values)
    args.test_split_values = expand_split_values(args.test_split_values)
    if args.fold_values is not None:
        args.fold_values = expand_split_values(args.fold_values)

    os.makedirs(args.out_dir, exist_ok=True)

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    df = pd.read_csv(args.nanobody_csv)
    required = [args.name_col, args.seq_col, args.y_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in nanobody_csv: {missing}")
    df = prepare_dataframe(df, args)

    if args.split_col is not None and args.split_col in df.columns:
        split_counts = df[args.split_col].astype(str).value_counts().sort_index()
        split_counts.to_csv(os.path.join(args.out_dir, "split_counts.csv"), header=["n"])
        print("split counts:")
        print(split_counts)

    if args.run_5fold:
        fold_values = infer_5fold_values(df, args)
        plan = make_5fold_plan(fold_values, val_fold_offset=args.val_fold_offset)
        pd.DataFrame(plan).to_csv(os.path.join(args.out_dir, "fold_plan.csv"), index=False)
        print("5-fold plan:")
        for item in plan:
            print(item)

        all_metrics = []
        all_val_preds = []
        all_test_preds = []
        all_histories = []

        for fold_info in plan:
            print("=" * 80)
            print("Starting", fold_info["fold_name"])
            fold_out_dir = os.path.join(args.out_dir, fold_info["fold_name"])
            metric_df, val_pred_df, test_pred_df, history_df = train_one_split(
                args=args,
                df=df,
                out_dir=fold_out_dir,
                train_values=fold_info["train_values"],
                val_values=fold_info["val_values"],
                test_values=fold_info["test_values"],
                fold_info=fold_info,
            )
            all_metrics.append(metric_df)
            all_val_preds.append(val_pred_df)
            all_test_preds.append(test_pred_df)
            all_histories.append(history_df)

        all_metrics_df = pd.concat(all_metrics, ignore_index=True)
        all_val_preds_df = pd.concat(all_val_preds, ignore_index=True)
        all_test_preds_df = pd.concat(all_test_preds, ignore_index=True)
        all_histories_df = pd.concat(all_histories, ignore_index=True)

        all_metrics_df.to_csv(os.path.join(args.out_dir, "all_folds_metrics.csv"), index=False)
        all_val_preds_df.to_csv(os.path.join(args.out_dir, "all_folds_validation_predictions.csv"), index=False)
        all_test_preds_df.to_csv(os.path.join(args.out_dir, "all_folds_test_predictions.csv"), index=False)
        all_histories_df.to_csv(os.path.join(args.out_dir, "all_folds_history.csv"), index=False)

        numeric_cols = all_metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in {"fold_num"}]
        summary_df = all_metrics_df.groupby(["split", "calibrated"], dropna=False)[numeric_cols].agg(["mean", "std"])
        summary_df.columns = ["_".join([str(x) for x in col if str(x)]) for col in summary_df.columns]
        summary_df = summary_df.reset_index()
        summary_df.to_csv(os.path.join(args.out_dir, "all_folds_metrics_summary.csv"), index=False)
        print("5-fold summary:")
        print(summary_df)
        print("saved:", os.path.join(args.out_dir, "all_folds_metrics.csv"))
        return

    # Backward-compatible single split mode.
    if args.split_col is not None and args.split_col in df.columns:
        train_values = args.train_split_values
        val_values = args.val_split_values
        test_values = args.test_split_values
    else:
        train_df, val_df, test_df = random_split_dataframe(df, args)
        # Save random split labels into a temporary column so the rest of the code path is the same.
        random_split_col = "__random_split__"
        df[random_split_col] = "unused"
        df.loc[train_df.index, random_split_col] = "train"
        df.loc[val_df.index, random_split_col] = "validation"
        df.loc[test_df.index, random_split_col] = "test"
        args.split_col = random_split_col
        train_values = ["train"]
        val_values = ["validation"]
        test_values = ["test"]

    train_one_split(
        args=args,
        df=df,
        out_dir=args.out_dir,
        train_values=train_values,
        val_values=val_values,
        test_values=test_values,
        fold_info=None,
    )


if __name__ == "__main__":
    main()
