"""Local notebook project orchestration.

`NotebookProject` models local kagglet workflow state with:
  * source files (percent-format `.py`, concatenated and converted to `.ipynb`)
  * optional `output` model (result uploaded back as a model artifact)
  * `deps` — other Kaggle assets required as inputs

`plan()` walks the DAG and returns only the projects whose outputs are stale.
`push()` uploads the current notebook. `poll()` waits for completion + prints logs.
"""

import time
import pathlib

import pydantic

from kagglet.model import KaggleModel
from kagglet.kernel import KaggleKernel
from kagglet.dataset import KaggleDataset
from kagglet.api.meta import KernelMeta
from kagglet.api.client import kaggle_api
from kagglet.api.kernels import push_kernel, fetch_kernel_logs, poll_kernel_terminal

_JUPYTEXT_HEADER = """\
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
"""

_PYTHON_LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}


def _normalize_notebook_for_kaggle(nb):
    nb.metadata.pop("jupytext", None)
    nb.metadata["language_info"] = dict(_PYTHON_LANGUAGE_INFO)
    nb.nbformat = 4
    nb.nbformat_minor = 4
    for cell in nb.cells:
        cell.pop("id", None)
    return nb


def percent_to_notebook(source: str):
    """Convert a percent-format `.py` source string to a Kaggle-ready nbformat notebook."""
    import jupytext

    return _normalize_notebook_for_kaggle(jupytext.reads(_JUPYTEXT_HEADER + "\n" + source, fmt="py:percent"))


class NotebookProject(pydantic.BaseModel):
    """Local notebook project spec.

    `sources` are paths to percent-format `.py` fragments concatenated to form
    the notebook. Paths are resolved against `sources_dir` if relative.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    kernel: KaggleKernel
    sources: list[str] = pydantic.Field(default_factory=list, repr=False)
    sources_dir: pathlib.Path | None = pydantic.Field(default=None, repr=False)
    output: KaggleModel | None = pydantic.Field(default=None, repr=False)
    deps: list["NotebookProject | KaggleModel | KaggleDataset"] = pydantic.Field(default_factory=list, repr=False)

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def inputs(self) -> list[KaggleModel]:
        models = []
        for d in self.deps:
            if isinstance(d, NotebookProject) and d.output:
                models.append(d.output)
            elif isinstance(d, KaggleModel):
                models.append(d)
        return models

    @property
    def datasets(self) -> list[KaggleDataset]:
        return [d for d in self.deps if isinstance(d, KaggleDataset)]

    @property
    def metadata(self) -> KernelMeta:
        meta = self.kernel.metadata
        meta.dataset_sources = [*meta.dataset_sources, *(d.slug for d in self.datasets)]
        return meta

    def plan(self, force: bool = False, timeout: int = 600) -> list["NotebookProject"]:
        """Walk deps, upload any stale KaggleModel, return notebooks whose outputs need rebuild."""
        import graphlib

        graph: dict[NotebookProject, set[NotebookProject]] = {}
        model_deps: set[KaggleModel] = set()
        stack = [self]
        while stack:
            nb = stack.pop()
            if nb in graph:
                continue
            nb_deps = set()
            for d in nb.deps:
                if isinstance(d, NotebookProject):
                    nb_deps.add(d)
                elif isinstance(d, KaggleModel):
                    model_deps.add(d)
            graph[nb] = nb_deps
            stack.extend(nb_deps)

        for m in model_deps:
            m.fetch()
            if not force and not m.needs_update():
                print(f"{m.name} up to date — skip")
                continue
            m.upload_file(m.build(), notes=m.expected_notes())
            if not m.wait_ready(timeout=timeout):
                raise RuntimeError(f"timeout waiting for {m.slug}/{m.version}")

        pending: list[NotebookProject] = []
        for nb in graphlib.TopologicalSorter(graph).static_order():
            if nb.output:
                nb.output.fetch()
                if not force and not nb.output.needs_update():
                    print(f"{nb.output.name} up to date — skip")
                    continue
                print(f"{nb.output.name} changed: {nb.output.notes} -> {nb.output.expected_notes()}")
            pending.append(nb)
        return pending

    def push(self):
        """Build notebook from sources, upload as a new kernel version."""
        meta = self.metadata
        model_sources = list(meta.model_sources or [])
        if self.inputs:
            for m in self.inputs:
                if m.version <= 0:
                    m.fetch()
                model_sources.append(f"{m.slug}/{m.version}")
                if not m.wait_ready(timeout=600):
                    raise RuntimeError(f"timeout waiting for {m.slug}/{m.version}")
        if model_sources:
            meta.model_sources = model_sources

        nb = percent_to_notebook(self._build_source())
        meta_json = meta.to_json()
        print(meta_json)
        result = push_kernel(kaggle_api(), meta_json, nb)
        print(f"kernel version {result.versionNumber} pushed: {result.ref}")

    def _resolve_source(self, source: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(source)
        if path.is_absolute():
            return path
        if self.sources_dir is None:
            raise ValueError(
                f"sources_dir not set but source {source!r} is relative; "
                "pass absolute paths or set NotebookProject(sources_dir=...)"
            )
        return self.sources_dir / path

    def _build_source(self) -> str:
        return "\n".join(self._resolve_source(s).read_text() for s in self.sources)

    def poll(self, interval: int = 10):
        """Block until kernel finishes, then print the downloaded `.log` files."""
        api = kaggle_api()
        t0 = time.monotonic()

        def tick(resp):
            elapsed = int(time.monotonic() - t0)
            mm, ss = divmod(elapsed, 60)
            print(f"\r\033[K{mm:02d}:{ss:02d}  {resp.status}", end="", flush=True)

        resp, status = poll_kernel_terminal(api, self.kernel.slug, interval=interval, on_tick=tick)
        print()

        for name, content in fetch_kernel_logs(api, self.kernel.slug):
            print(f"--- {name} ---")
            print(content)

        if "error" in status or "cancel" in status:
            msg = resp.failure_message or resp.status
            raise RuntimeError(f"kernel {self.kernel.slug} failed: {msg}")


NotebookProject.model_rebuild()
