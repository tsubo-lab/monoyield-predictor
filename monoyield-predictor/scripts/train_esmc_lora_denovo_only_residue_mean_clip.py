#!/usr/bin/env python

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# ============================================================
# Utility
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_esmc_hidden_size(model_name: str) -> int:
    name = model_name.lower()

    if "6b" in name:
        return 2560
    if "600m" in name:
        return 1152
    if "300m" in name:
        return 960

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
# Dataset: denovo absolute-value regression only
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
        target_modules=None,
    ):
        super().__init__()

        if target_modules is None:
            target_modules = ["out_proj"]

        self.esm = AutoModel.from_pretrained(
            model_name,
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
        # For biohub/ESMC models, call the LoRA-injected base model directly.
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

    if not lora_params:
        raise ValueError("No trainable LoRA parameters were found.")
    if not head_params:
        raise ValueError("No trainable regression-head parameters were found.")

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


# ============================================================
# Denovo split
# ============================================================


def make_denovo_split(
    denovo_df,
    test_group,
    groups,
    split_col="split",
):
    test_idx = groups.index(test_group)
    val_group = groups[(test_idx + 1) % len(groups)]

    train_groups = [
        group for group in groups
        if group not in [test_group, val_group]
    ]

    denovo_train = denovo_df[denovo_df[split_col].isin(train_groups)].copy()
    denovo_val = denovo_df[denovo_df[split_col] == val_group].copy()
    denovo_test = denovo_df[denovo_df[split_col] == test_group].copy()

    return {
        "train_groups": train_groups,
        "val_group": val_group,
        "test_group": test_group,
        "denovo_train": denovo_train,
        "denovo_val": denovo_val,
        "denovo_test": denovo_test,
    }


# ============================================================
# Train / predict
# ============================================================


def calculate_absolute_loss(pred, labels, loss_type):
    if loss_type == "huber":
        return nn.functional.huber_loss(pred, labels.float(), delta=1.0)
    if loss_type == "mse":
        return nn.functional.mse_loss(pred, labels.float())
    raise ValueError(f"Unknown abs_loss_type: {loss_type}")


def train_denovo_epoch_residue_mean(
    model,
    train_loader,
    optimizer,
    device,
    tokenizer,
    grad_accum_steps=8,
    grad_clip=1.0,
    abs_loss_type="huber",
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses = []
    n_steps = len(train_loader)

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pred, _, _ = model.predict_value(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        loss = calculate_absolute_loss(pred, labels, abs_loss_type)
        (loss / grad_accum_steps).backward()

        if step % grad_accum_steps == 0 or step == n_steps:
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "loss_abs": float(np.mean(losses)) if losses else np.nan,
    }


@torch.no_grad()
def predict_single_loader_residue_mean(
    model,
    loader,
    device,
    tokenizer,
    abs_loss_type="huber",
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

        loss = calculate_absolute_loss(pred, labels, abs_loss_type)

        names.extend(batch["name"])
        y_true.append(labels.detach().cpu())
        y_pred.append(pred.detach().cpu())
        losses.append(float(loss.detach().cpu()))

    y_true = torch.cat(y_true).numpy() if y_true else np.array([])
    y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])

    return {
        "name": names,
        "y_true": y_true,
        "y_pred": y_pred,
        "loss": float(np.mean(losses)) if losses else np.nan,
    }


def metric_to_early_stop_score(metrics, metric_name):
    value = metrics[metric_name]

    # NaN validation metrics cannot be considered an improvement.
    if not np.isfinite(value):
        return -np.inf

    if metric_name == "rmse":
        return -value

    return value


def run_denovo_training(
    model,
    train_loader,
    val_loader,
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
        train_loss = train_denovo_epoch_residue_mean(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            tokenizer=tokenizer,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            abs_loss_type=args.abs_loss_type,
        )

        val_pred = predict_single_loader_residue_mean(
            model=model,
            loader=val_loader,
            device=device,
            tokenizer=tokenizer,
            abs_loss_type=args.abs_loss_type,
        )

        val_metrics = calc_metrics(val_pred["y_true"], val_pred["y_pred"])
        score = metric_to_early_stop_score(val_metrics, args.early_stop_metric)

        print(
            f"[DenovoOnlyResidue] epoch={epoch:03d} "
            f"loss={train_loss['loss']:.5f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_pearson={val_metrics['pearson']:.4f} "
            f"val_spearman={val_metrics['spearman']:.4f} "
            f"val_top10={val_metrics['top10_enrichment']:.3f}"
        )

        history.append(
            {
                "stage": "denovo_only_absolute_residue_mean",
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_loss.items()},
                **{f"denovo_val_{key}": value for key, value in val_metrics.items()},
                "denovo_val_loss": val_pred["loss"],
            }
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

    if best_state is None:
        raise RuntimeError(
            "No valid best checkpoint was selected. "
            "Check that the validation split has enough non-constant observations "
            f"for early_stop_metric={args.early_stop_metric}."
        )

    model.load_state_dict(best_state)

    best_info = {
        "best_stage": "denovo_only_absolute_residue_mean",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "early_stop_metric": args.early_stop_metric,
    }

    print("best_epoch:", best_epoch)
    print("best_score:", best_score)

    return model, pd.DataFrame(history), best_info


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train an ESMC LoRA residue-mean regressor using only the denovo CSV. "
            "No DMS input or DMS loss is used."
        )
    )

    parser.add_argument("--denovo_csv", required=True)
    parser.add_argument(
        "--out_dir",
        default="esmc_lora_denovo_only_residue_mean",
    )

    parser.add_argument("--model_name", default="biohub/ESMC-600M")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size of ESMC model. If None, inferred from --model_name.",
    )

    parser.add_argument("--name_col", default="name")
    parser.add_argument("--seq_col", default="aa_seq")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--rep_col", default="rep")
    parser.add_argument("--y_col", default="monomer_log_mgL")

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
    parser.add_argument(
        "--abs_loss_type",
        default="huber",
        choices=["huber", "mse"],
    )

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

    args = parser.parse_args()
    set_seed(args.seed)

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    denovo_df = pd.read_csv(args.denovo_csv)

    denovo_required = [
        args.name_col,
        args.seq_col,
        args.rep_col,
        args.split_col,
        args.y_col,
    ]

    missing_denovo = [
        column for column in denovo_required
        if column not in denovo_df.columns
    ]
    if missing_denovo:
        raise ValueError(f"Missing columns in denovo_csv: {missing_denovo}")

    if args.clip_y_min is not None or args.clip_y_max is not None:
        denovo_df[args.y_col] = denovo_df[args.y_col].clip(
            lower=args.clip_y_min,
            upper=args.clip_y_max,
        )

    denovo_df = denovo_df[denovo_required].dropna().copy()
    denovo_df[args.split_col] = denovo_df[args.split_col].astype(str)
    denovo_df[args.rep_col] = denovo_df[args.rep_col].astype(str)

    print("denovo rows:", len(denovo_df))
    print(denovo_df[args.split_col].value_counts().sort_index())

    all_folds = args.folds
    run_folds = all_folds if args.run_folds is None else args.run_folds

    for group in run_folds:
        if group not in all_folds:
            raise ValueError(f"{group} is not in --folds: {all_folds}")

    observed_groups = set(denovo_df[args.split_col].unique())
    missing_groups = [group for group in all_folds if group not in observed_groups]
    if missing_groups:
        raise ValueError(
            f"The following --folds are absent from {args.split_col}: {missing_groups}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    all_metric_records = []
    all_prediction_records = []

    for test_group in run_folds:
        print("=" * 100)
        print("Running fold:", test_group)

        split_data = make_denovo_split(
            denovo_df=denovo_df,
            test_group=test_group,
            groups=all_folds,
            split_col=args.split_col,
        )

        train_groups = split_data["train_groups"]
        val_group = split_data["val_group"]
        train_df = split_data["denovo_train"]
        val_df = split_data["denovo_val"]
        test_df = split_data["denovo_test"]

        print("train_groups:", train_groups)
        print("val_group:", val_group)
        print("test_group:", test_group)
        print("n_denovo_train:", len(train_df))
        print("n_denovo_val:", len(val_df))
        print("n_denovo_test:", len(test_df))

        if len(train_df) == 0:
            raise ValueError(f"Training data is empty for fold {test_group}.")
        if len(val_df) == 0:
            raise ValueError(f"Validation data is empty for fold {test_group}.")
        if len(test_df) == 0:
            raise ValueError(f"Test data is empty for fold {test_group}.")

        fold_dir = os.path.join(args.out_dir, test_group)
        os.makedirs(fold_dir, exist_ok=True)

        train_loader = make_single_loader(
            train_df,
            tokenizer,
            seq_col=args.seq_col,
            y_col=args.y_col,
            name_col=args.name_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        val_loader = make_single_loader(
            val_df,
            tokenizer,
            seq_col=args.seq_col,
            y_col=args.y_col,
            name_col=args.name_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        test_loader = make_single_loader(
            test_df,
            tokenizer,
            seq_col=args.seq_col,
            y_col=args.y_col,
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

        total_trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        total_params = sum(parameter.numel() for parameter in model.parameters())
        print(
            "total trainable params including residue head: "
            f"{total_trainable:,} / {total_params:,} "
            f"({100 * total_trainable / total_params:.4f}%)"
        )

        model, history_df, best_info = run_denovo_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
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
            },
            os.path.join(fold_dir, "best_model.pt"),
        )

        for split_name, loader in [
            ("validation", val_loader),
            ("test", test_loader),
        ]:
            pred = predict_single_loader_residue_mean(
                model=model,
                loader=loader,
                device=device,
                tokenizer=tokenizer,
                abs_loss_type=args.abs_loss_type,
            )

            metrics = calc_metrics(pred["y_true"], pred["y_pred"])
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
                "early_stop_metric": best_info["early_stop_metric"],
                "n": len(pred["y_true"]),
            }
            record.update(metrics)
            all_metric_records.append(record)

            prediction_df = pd.DataFrame(
                {
                    "fold": test_group,
                    "data_type": "denovo",
                    "eval_split": split_name,
                    "name": pred["name"],
                    "y_true": pred["y_true"],
                    "y_pred": pred["y_pred"],
                }
            )
            all_prediction_records.append(prediction_df)

        pd.DataFrame(all_metric_records).to_csv(
            os.path.join(args.out_dir, "metrics_partial.csv"),
            index=False,
        )

        if all_prediction_records:
            pd.concat(all_prediction_records, ignore_index=True).to_csv(
                os.path.join(args.out_dir, "denovo_predictions_partial.csv"),
                index=False,
            )

        del model
        torch.cuda.empty_cache()

    metric_df = pd.DataFrame(all_metric_records)
    metric_path = os.path.join(args.out_dir, "metrics_5fold.csv")
    metric_df.to_csv(metric_path, index=False)
    print("saved:", metric_path)

    if all_prediction_records:
        prediction_df = pd.concat(all_prediction_records, ignore_index=True)
        prediction_path = os.path.join(
            args.out_dir,
            "denovo_predictions_5fold.csv",
        )
        prediction_df.to_csv(prediction_path, index=False)
        print("saved:", prediction_path)

    print("denovo test summary:")
    test_summary = (
        metric_df
        .query("data_type == 'denovo' and eval_split == 'test'")
        .agg(
            {
                "rmse": ["mean", "std"],
                "r2": ["mean", "std"],
                "pearson": ["mean", "std"],
                "spearman": ["mean", "std"],
                "top1_enrichment": ["mean", "std"],
                "top5_enrichment": ["mean", "std"],
                "top10_enrichment": ["mean", "std"],
            }
        )
    )
    print(test_summary)


if __name__ == "__main__":
    main()
