# %% [markdown]
# # Gemma 4 31B shard streaming on Kaggle TPU
#
# This notebook installs only the small runtime helpers needed to read
# safetensors shards and tokenize the prompt. The model weights are attached as
# a Kaggle model input and streamed layer by layer by `main.py`.

# %%
import os
import sys
import subprocess

os.environ["KERAS_BACKEND"] = "jax"

PACKAGES = [
    "safetensors>=0.5.3",
    "tokenizers>=0.21.0",
]

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *PACKAGES])
