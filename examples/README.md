# Examples

Each example is a directory pushed via `kagglet push <dir>`. The directory holds:

- `notebook.toml` — slug + title (required) and optional kernel settings
- one or more percent-format `.py` source files

## Prereqs

1. Kaggle account.
2. OAuth login:
   `KAGGLE_ENABLE_OAUTH=1 kaggle auth login`.
3. Confirm the active Kaggle user with `pixi run kagglet whoami`.

Each example's `slug` is just the kernel name (e.g. `"kagglet-hello"`); the CLI
prepends your active Kaggle username automatically. Use `"alice/kagglet-hello"`
only when explicitly pushing to a different owner.

## Run

```bash
KAGGLE_ENABLE_OAUTH=1 kaggle auth login
pixi run kagglet push examples/hello --poll   # push + wait + print logs
pixi run kagglet show examples/hello          # dry-run: print kernel-metadata.json
```

## Available examples

| Dir | What it shows |
|-----|---------------|
| [`hello/`](hello/) | Two percent-format `.py` sources auto-discovered into one notebook. No deps, CPU-only. |
| [`gemma4-keras-tpu/`](gemma4-keras-tpu/) | Gemma 4 31B inference on Kaggle TPU (`TpuV5E8`) with resident JAX tensor sharding. |
