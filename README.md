# kagglet

Automation toolkit for Kaggle. Like `kubelet` / `raylet`, `kagglet` is the small agent
you bolt onto a Kaggle workflow — it packages artifacts, pushes notebooks, and bridges
cell execution to your own runtime.

## What's in the box

| Module | Purpose |
|--------|---------|
| `kagglet.asset`    | `KaggleAsset` — shared `{owner}/{name}` identity for Kaggle resources |
| `kagglet.kernel`   | `KaggleKernel` — Kaggle kernel metadata (`title`, `accelerator`, dataset/model sources, internet) |
| `kagglet.model`    | `KaggleModel` — versioned artifact with `fetch` / `upload` / `wait_ready` / `needs_update` on top of notes-based cache invalidation |
| `kagglet.notebook` | `NotebookProject` — local notebook orchestration with `sources`, deps DAG (`plan` / `push` / `poll`), and percent-format `.py` conversion |
| `kagglet.tar`      | `TarExtractor` — 4 interchangeable tar / tar.zst extraction strategies |
| `kagglet.api`      | `kaggle_api()` cached singleton + `parallel_kaggle_uploads()` context manager (monkey-patches the kaggle client to upload model files concurrently) |
| `kagglet.relay`    | `RelaySession` — hijacks IPython `run_cell` on the Kaggle kernel and forwards every cell to an external Jupyter kernel you control |
| `kagglet.stream`   | `stream_logs()` — real-time kernel logs via Firebase SSE (browser-cookie auth) |

`relay` and `stream` require optional deps: `pip install 'kagglet[relay,stream]'`.

## Minimal example

```python
from pathlib import Path

from kagglet import KaggleKernel, KaggleModel, NotebookProject

class EnvArtifact(KaggleModel):
    MARKER = "env.tar.zst"

    def expected_notes(self):
        import hashlib
        return {"lock": hashlib.sha256(Path("pixi.lock").read_bytes()).hexdigest()[:12]}

    def build(self):
        ...  # produce env.tar.zst, return Path

ENV = EnvArtifact(owner="your-owner", name="my-env")

NB = NotebookProject(
    kernel=KaggleKernel(owner="your-owner", name="my-nb"),
    sources=["bootstrap.py", "main.py"],
    sources_dir=Path(__file__).resolve().parent,
    deps=[ENV],
)

for nb in NB.plan():    # uploads ENV if stale, returns notebooks to rebuild
    nb.push()
    nb.poll()
```

## Streaming logs

```python
from kagglet.stream import stream_logs, cookies_from_chrome

stream_logs("your-owner/my-nb", cookies=cookies_from_chrome())
```

On non-Linux or non-Chrome setups, pass `cookies={...}` yourself (see
`kagglet.stream.KAGGLE_COOKIE_NAMES` for the required names).

## Relay session

Install a Jupyter kernelspec from your pixi/conda env into the Kaggle kernel's
search path, then:

```python
import pathlib
from kagglet.relay import RelaySession

relay = RelaySession(
    "my-kernel",
    cwd=pathlib.Path("/kaggle/working/project"),
    cleanup_paths=[pathlib.Path("/kaggle/working/project/.pixi")],
)
relay.start()
# subsequent cells now execute inside `my-kernel`
# ...
relay.cleanup()
```

## Install

```bash
pip install kagglet                 # core
pip install 'kagglet[relay,stream]' # with optional features
```

Or with pixi:

```bash
pixi add --pypi kagglet
```

## Examples

Runnable examples live under [`examples/`](examples/). Each one is a directory
with `notebook.yaml` (`kernel.name`, optional `kernel.owner` / `kernel.title`)
and one or more percent-format `.py` sources; push via:

```bash
pixi run kagglet push examples/hello --poll
```

Start with [`examples/hello/`](examples/hello/) — the smallest end-to-end push.
