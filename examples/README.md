# Examples

A standalone pixi workspace that pulls `kagglet` from `../src` and exposes one
task per example. Each example is a directory with:

- `notebook.yaml` — `kernel.name` and `sources` are required; optional
  `kernel.owner` / `kernel.title` and other kernel settings. `sources` accepts
  explicit paths (e.g. `[bootstrap.py, main.py]`) or globs (e.g. `["*.py"]`,
  expanded against the example directory and sorted)
- one or more percent-format `.py` source files

## Prereqs

1. Kaggle account.
2. OAuth login: `KAGGLE_ENABLE_OAUTH=1 kaggle auth login`.
3. Confirm the active Kaggle user: `pixi run whoami`.

Each example omits `kernel.owner`; the CLI fills it from your active Kaggle
account. Set `kernel.owner: alice` only when explicitly pushing to a different
owner.

## Run

From this directory:

```bash
pixi run hello                  # push hello + poll until it finishes
pixi run hello-show             # dry-run: print derived kernel-metadata.json
pixi run gemma4-keras-tpu       # push gemma4-keras-tpu + poll
pixi run gemma4-keras-tpu-show  # dry-run for the gemma example
```

Each task is a thin wrapper over `kagglet push <dir> --poll` (or `kagglet show
<dir>`); see `pixi.toml`. Add a new example by dropping a directory next to the
existing ones and registering one or two `[tasks.*]` entries.

## Available examples

| Dir | What it shows |
|-----|---------------|
| [`hello/`](hello/) | Two percent-format `.py` sources auto-discovered into one notebook. No deps, CPU-only. |
| [`gemma4-keras-tpu/`](gemma4-keras-tpu/) | Gemma 4 31B inference on Kaggle TPU (`TpuV5E8`) with resident JAX tensor sharding. Design notes and runtime env overrides live in the markdown cell at the top of `main.py`. |
