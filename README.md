# kagglet

Automation toolkit for Kaggle. Like `kubelet` / `raylet`, `kagglet` is the small agent
you bolt onto a Kaggle workflow — it packages artifacts, pushes notebooks, and bridges
cell execution to your own runtime.

## What's in the box

| Module | Purpose |
|--------|---------|
| `kagglet.assets`   | `KaggleAsset` (`{owner}/{name}` identity) and the resource subclasses: `KaggleDataset`, `KaggleKernel` (with `Accelerator`), `KaggleModel` (versioned artifact with `fetch` / `upload` / `wait_ready` / `needs_update` on top of notes-based cache invalidation) |
| `kagglet.notebook` | `NotebookProject` — local notebook orchestration with `sources`, deps DAG (`plan` / `push` / `poll`), and percent-format `.py` conversion |
| `kagglet.api`      | `Kaggle` — facade over the kaggle SDK with kernel/model methods and `parallel_uploads()` (monkey-patches `upload_files` for concurrent uploads); `kaggle()` returns a cached, authenticated singleton |
| `kagglet.stream`   | `stream_logs()` — real-time kernel logs via Firebase SSE (browser-cookie auth) |

`stream` requires optional deps: `pip install 'kagglet[stream]'`.

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

## Install

```bash
pip install kagglet            # core
pip install 'kagglet[stream]'  # with optional log-streaming deps
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
