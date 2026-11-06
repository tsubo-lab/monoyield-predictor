"""Loading shim for the `esm` 3.4.x package (PyPI `esm==3.4.2`) with a stock PyPI `transformers==4.57.6`.

The scripts in `scripts/` load ESMC with `transformers.AutoModel` / `AutoTokenizer`, which needs the transformers build the paper
environment was made with (native ESMC support). Importing this module first replaces those two names in `transformers` with loaders
that return `esm`'s own `EsmcModel` / `EsmcTokenizer`; the model, the LoRA keys and the heads are then used unchanged.
"""
import transformers
from esm.models.esmc.model import EsmcModel
from esm.models.esmc.tokenizer import EsmcTokenizer


class AutoModel:
    @staticmethod
    def from_pretrained(name, revision=None, trust_remote_code=None, **kwargs):
        return EsmcModel.from_pretrained(name, revision=revision, **kwargs)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(name, revision=None, trust_remote_code=None, **kwargs):
        return EsmcTokenizer.from_pretrained(name, revision=revision, **kwargs)


transformers.AutoModel = AutoModel
transformers.AutoTokenizer = AutoTokenizer
