#!/usr/bin/env python
"""Strip a training checkpoint (best_model.pt = full base-model state, ~25 GB for ESMC-6B) down to the trained parameters only:
LoRA matrices (keys containing `lora_`) and the regression head (every key that does not start with `esm.`).
The frozen base model is re-loaded from Hugging Face at the pinned revision at prediction time.

usage: export_adapter_weights.py --in_dir DIR --out_dir DIR --name NAME [--ckpt_glob 'group_*/best_model.pt'] [--model_revision HASH]
Output: OUT_DIR/NAME_1.pt ... NAME_5.pt (k = fold number taken from the folder name, e.g. group_3 -> NAME_3.pt).
Exported files carry `"adapter_only": True`."""
import argparse, glob, os, re, torch

def is_trained(key):
    return "lora_" in key or not key.startswith("esm.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", required=True, help="file prefix, e.g. esmc_monoyield -> esmc_monoyield_1.pt")
    ap.add_argument("--ckpt_glob", default="group_*/best_model.pt")
    ap.add_argument("--model_revision", default=None, help="HF revision of the base model used for training (recorded in the file)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.in_dir, a.ckpt_glob)))
    assert files, f"no checkpoint under {a.in_dir}/{a.ckpt_glob}"
    for f in files:
        ck = torch.load(f, map_location="cpu", mmap=True, weights_only=False)   # mmap: only the selected tensors are read
        sd = {k: v.clone() for k, v in ck["model_state_dict"].items() if is_trained(k)}
        n_all = len(ck["model_state_dict"])
        out = {"model_state_dict": sd, "args": ck.get("args", {}), "best_info": ck.get("best_info"), "test_group": ck.get("test_group"),
               "val_group": ck.get("val_group"), "adapter_only": True, "base_model_revision": a.model_revision}
        rel = os.path.relpath(f, a.in_dir); k = re.search(r"(\d+)", os.path.basename(os.path.dirname(rel))).group(1)
        dst = os.path.join(a.out_dir, f"{a.name}_{k}.pt"); os.makedirs(a.out_dir, exist_ok=True)
        torch.save(out, dst)
        print(f"{rel}: kept {len(sd)}/{n_all} tensors, {os.path.getsize(dst)/1e6:.1f} MB (from {os.path.getsize(f)/1e9:.1f} GB)", flush=True)

if __name__ == "__main__":
    main()
