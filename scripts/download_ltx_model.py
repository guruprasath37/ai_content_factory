"""Download the LTX-Video 2B weights (about 27 GB: the 19 GB text encoder, the transformer and the VAE).

    python scripts/download_ltx_model.py            # -> models/LTX-Video
Needs: pip install huggingface_hub
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

dest = Path(__file__).resolve().parent.parent / (sys.argv[1] if len(sys.argv) > 1 else "models/LTX-Video")
print(f"Downloading to {dest} ...")
snapshot_download("Lightricks/LTX-Video", local_dir=str(dest),
                  allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"])
print("Done.")
