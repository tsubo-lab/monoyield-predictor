"""Loading helper that accepts both full training checkpoints and adapter-only exports (see export_adapter_weights.py)."""

def load_state_dict_flexible(model, sd, adapter_only=False):
    """adapter_only=False: strict load (original behaviour).
    adapter_only=True : the checkpoint holds only LoRA + head tensors; every other key must belong to the frozen base model
    (`esm.*` without `lora_`), which was already loaded from Hugging Face when the model was constructed."""
    if not adapter_only:
        return model.load_state_dict(sd, strict=True)
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, f"unexpected keys: {res.unexpected_keys[:5]}"
    bad = [k for k in res.missing_keys if "lora_" in k or not k.startswith("esm.")]
    assert not bad, f"trained parameters missing from the adapter checkpoint: {bad[:5]}"
    print(f"  adapter-only checkpoint: loaded {len(sd)} tensors; {len(res.missing_keys)} frozen base tensors taken from the pretrained model")
    return res


import os, re

def checkpoint_path(model_dir, folder, checkpoint_name="best_model.pt"):
    """`model_dir/<folder>/<checkpoint_name>` (training layout) or, if that does not exist, the released flat layout
    `model_dir/<model_dir name>_<k>.pt` (e.g. weights/esmc_monoyield/esmc_monoyield_3.pt for folder `group_3`)."""
    p = os.path.join(model_dir, folder, checkpoint_name)
    if os.path.exists(p):
        return p
    m = re.search(r"(\d+)$", folder)
    if m:
        flat = os.path.join(model_dir, f"{os.path.basename(os.path.normpath(model_dir))}_{m.group(1)}.pt")
        if os.path.exists(flat):
            return flat
    return p
