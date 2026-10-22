#!/usr/bin/env python
"""
nanobodyでfine-tuneしたESMC-6B LoRAモデルで、denovo配列(denovo_5fold.csv)を予測する。

nanobody finetuneのベースとなったone-stage LoRA checkpoint(source_checkpoint_args)には
denovoのtrain/val/test group分割が記録されている。ここではそのtrain_groupsと重ならない
denovoのsplit(=repも重複しない。denovo_5fold.csvはrep単位でsplitが割られているため)
のみを予測対象にする。
"""

import os
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


def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return {
        "rmse": rmse,
        "r2": r2,
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
    }


class ProteinRegressionDataset(Dataset):
    def __init__(self, df, tokenizer, seq_col, y_col, name_col, max_length):
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
        y_raw = float(row[self.y_col])
        enc = self.tokenizer(
            seq, padding=False, truncation=True, max_length=self.max_length, return_tensors=None,
        )
        return {
            "name": str(row[self.name_col]),
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
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
            "labels_raw": torch.stack([x["labels_raw"] for x in batch]),
        }
    return collate_fn


class EsmCLoRAResidueMeanRegressor(nn.Module):
    def __init__(self, model_name, lora_r, lora_alpha, lora_dropout, hidden_size, target_modules):
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
            tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id,
            tokenizer.bos_token_id, tokenizer.sep_token_id,
        ]:
            if tok_id is not None:
                special_ids.add(tok_id)
        for tok_id in special_ids:
            mask = mask & (input_ids != tok_id)
        return mask

    def predict_value(self, input_ids, attention_mask, tokenizer):
        outputs = self.esm.base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        residue_logits = self.residue_head(hidden).squeeze(-1)
        residue_mask = self.make_residue_mask(input_ids, attention_mask, tokenizer)
        residue_logits_masked = residue_logits * residue_mask.to(residue_logits.dtype)
        denom = residue_mask.sum(dim=1).clamp(min=1).to(residue_logits.dtype)
        pred = residue_logits_masked.sum(dim=1) / denom
        return pred


@torch.no_grad()
def predict_loader(model, loader, device, tokenizer, y_mean, y_std, cal_a, cal_b):
    model.eval()
    names, y_true, y_pred_raw, y_pred_cal = [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pred_norm = model.predict_value(input_ids, attention_mask, tokenizer)
        pred_raw = pred_norm.detach().cpu().numpy() * y_std + y_mean
        pred_cal = cal_a * pred_raw + cal_b
        names.extend(batch["name"])
        y_true.extend(batch["labels_raw"].numpy().tolist())
        y_pred_raw.extend(pred_raw.tolist())
        y_pred_cal.extend(pred_cal.tolist())
    return {
        "name": names,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred_raw": np.asarray(y_pred_raw, dtype=float),
        "y_pred_calibrated": np.asarray(y_pred_cal, dtype=float),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/nanobody_ft/group_1/best_nanobody_finetuned_model.pt")
    parser.add_argument("--denovo_csv", default="denovo_5fold.csv")
    parser.add_argument("--out_csv", default="outputs/nanobody_ft/group_1/denovo_holdout_predictions.csv")

    parser.add_argument("--name_col", default="name")
    parser.add_argument("--seq_col", default="aa_seq")
    parser.add_argument("--y_col", default="monomer_log_mgL")
    parser.add_argument("--split_col", default="split")

    parser.add_argument("--eval_splits", nargs="+", default=None,
                         help="denovo split values to evaluate. If None, inferred as the "
                              "test+val groups of the base one-stage checkpoint (i.e. groups "
                              "NOT used to train the ESMC-6B LoRA backbone this nanobody "
                              "model was fine-tuned from).")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    if "HF_TOKEN" in os.environ:
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    nb_args = ckpt.get("args", {})
    base_args = ckpt.get("source_checkpoint_args", {})

    model_name = nb_args.get("model_name") or base_args.get("model_name", "biohub/ESMC-6B")
    hidden_size = nb_args.get("hidden_size") or infer_esmc_hidden_size(model_name)
    lora_r = nb_args.get("lora_r") or base_args.get("lora_r", 4)
    lora_alpha = nb_args.get("lora_alpha") or base_args.get("lora_alpha", 8)
    lora_dropout = nb_args.get("lora_dropout") if nb_args.get("lora_dropout") is not None else base_args.get("lora_dropout", 0.05)
    target_modules = nb_args.get("target_modules", ["out_proj"])

    print("model_name:", model_name, "hidden_size:", hidden_size)
    print("lora_r/alpha/dropout:", lora_r, lora_alpha, lora_dropout)
    print("base_checkpoint train_groups (denovo, base ESMC-6B LoRA):",
          base_args.get("run_folds") or base_args.get("folds"))

    y_mean = ckpt["target_normalization"]["y_mean"]
    y_std = ckpt["target_normalization"]["y_std"]
    cal_a = ckpt["linear_calibration"]["a"]
    cal_b = ckpt["linear_calibration"]["b"]
    print("target_normalization:", ckpt["target_normalization"])
    print("linear_calibration:", ckpt["linear_calibration"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = EsmCLoRAResidueMeanRegressor(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        hidden_size=hidden_size,
        target_modules=target_modules,
    )
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("missing_keys:", len(load_result.missing_keys), "unexpected_keys:", len(load_result.unexpected_keys))
    model.to(device)

    denovo_df = pd.read_csv(args.denovo_csv)
    denovo_df = denovo_df.dropna(subset=[args.name_col, args.seq_col, args.y_col, args.split_col]).copy()

    eval_splits = args.eval_splits
    if eval_splits is None:
        # Base one-stage checkpoint's denovo 5-fold scheme: test_group = the fold whose
        # best_model.pt was loaded as init_pt; val_group = the next fold in rotation;
        # both were excluded from base LoRA training (train_groups below).
        train_groups = set(base_args.get("run_folds") or [])
        all_groups = base_args.get("folds", ["group_1", "group_2", "group_3", "group_4", "group_5"])
        if not train_groups:
            # run_folds was None in this project's runs -> all folds were run as test_group
            # in 5-fold CV, but a SPECIFIC init_pt (one fold's best_model.pt) was used here.
            # Recover it from the init_pt path itself.
            init_pt = nb_args.get("init_pt", "")
            test_group = next((g for g in all_groups if f"/{g}/" in init_pt), None)
            if test_group is None:
                raise ValueError(
                    f"Could not infer base test_group from init_pt={init_pt!r}; "
                    "pass --eval_splits explicitly."
                )
            idx = all_groups.index(test_group)
            val_group = all_groups[(idx + 1) % len(all_groups)]
            train_groups = set(all_groups) - {test_group, val_group}
        eval_splits = [g for g in all_groups if g not in train_groups]

    print("evaluating denovo splits (held out from base ESMC-6B LoRA training):", eval_splits)

    eval_df = denovo_df[denovo_df[args.split_col].isin(eval_splits)].copy()
    print("n_denovo_eval_rows:", len(eval_df))
    if len(eval_df) == 0:
        raise ValueError(f"No denovo rows found for splits {eval_splits}")

    ds = ProteinRegressionDataset(eval_df, tokenizer, args.seq_col, args.y_col, args.name_col, args.max_length)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collate_fn(tokenizer), num_workers=args.num_workers, pin_memory=True,
    )

    pred = predict_loader(model, loader, device, tokenizer, y_mean, y_std, cal_a, cal_b)

    out_df = pd.DataFrame({
        "name": pred["name"],
        "split": eval_df[args.split_col].values,
        "y_true": pred["y_true"],
        "y_pred_raw": pred["y_pred_raw"],
        "y_pred_calibrated": pred["y_pred_calibrated"],
    })
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)

    print("metrics (raw):", calc_metrics(out_df["y_true"], out_df["y_pred_raw"]))
    print("metrics (calibrated):", calc_metrics(out_df["y_true"], out_df["y_pred_calibrated"]))

    for split_name, sub in out_df.groupby("split"):
        print(f"  [{split_name}] n={len(sub)} raw={calc_metrics(sub['y_true'], sub['y_pred_raw'])} "
              f"calibrated={calc_metrics(sub['y_true'], sub['y_pred_calibrated'])}")


if __name__ == "__main__":
    main()
