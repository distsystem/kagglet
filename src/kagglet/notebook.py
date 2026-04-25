"""Kaggle notebook push DAG.

`KaggleNotebook` models a notebook kernel with:
  * source files (percent-format `.py`, concatenated and converted to `.ipynb`)
  * optional `output` model (result uploaded back as a model artifact)
  * `deps` — other `KaggleNotebook` or `KaggleModel` instances required as inputs

`plan()` walks the DAG and returns only the notebooks whose outputs are stale.
`push()` uploads the current notebook. `poll()` waits for completion + prints logs.
"""

import time
import pathlib

from pydantic import Field, BaseModel, ConfigDict

from kagglet.api import kaggle_api
from kagglet.model import KaggleModel

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


class KernelMeta(BaseModel):
    """`kernel-metadata.json` body for `kaggle kernels push` (snake_case schema)."""

    model_config = ConfigDict(extra="forbid")

    code_file: str = "notebook.ipynb"
    language: str = "python"
    kernel_type: str = "notebook"
    is_private: str = "true"
    enable_gpu: str
    machine_shape: str | None
    id: str
    title: str
    enable_internet: str
    dataset_sources: list[str] = Field(default_factory=list)
    competition_sources: list[str] = Field(default_factory=list)
    kernel_sources: list[str] = Field(default_factory=list)
    model_sources: list[str] | None = None

    def to_json(self) -> str:
        # Drop `model_sources` when unset so we match the historical layout
        # (Kaggle accepts either, but the diff stays smaller).
        exclude = {"model_sources"} if self.model_sources is None else set()
        return self.model_dump_json(indent=2, exclude=exclude)


class KaggleNotebook(BaseModel):
    """Kaggle kernel spec.

    `sources` are paths to percent-format `.py` fragments concatenated to form
    the notebook. Paths are resolved against `sources_dir` if relative.
    """

    model_config = ConfigDict(extra="forbid")

    slug: str
    title: str
    sources: list[str] = Field(default_factory=list, repr=False)
    sources_dir: pathlib.Path | None = Field(default=None, repr=False)
    internet: bool = True
    competition: str = ""
    accelerator: str = ""
    output: KaggleModel | None = Field(default=None, repr=False)
    deps: list["KaggleNotebook | KaggleModel"] = Field(default_factory=list, repr=False)

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def inputs(self) -> list[KaggleModel]:
        models = []
        for d in self.deps:
            if isinstance(d, KaggleNotebook) and d.output:
                models.append(d.output)
            elif isinstance(d, KaggleModel):
                models.append(d)
        return models

    @property
    def metadata(self) -> KernelMeta:
        return KernelMeta(
            id=self.slug,
            title=self.title,
            enable_gpu=str(bool(self.accelerator)).lower(),
            machine_shape=self.accelerator or None,
            enable_internet=str(self.internet).lower(),
            competition_sources=[self.competition] if self.competition else [],
        )

    def plan(self, force: bool = False, timeout: int = 600) -> list["KaggleNotebook"]:
        """Walk deps, upload any stale KaggleModel, return notebooks whose outputs need rebuild."""
        import graphlib

        graph: dict[KaggleNotebook, set[KaggleNotebook]] = {}
        model_deps: set[KaggleModel] = set()
        stack = [self]
        while stack:
            nb = stack.pop()
            if nb in graph:
                continue
            nb_deps = set()
            for d in nb.deps:
                if isinstance(d, KaggleNotebook):
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

        pending: list[KaggleNotebook] = []
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
        import tempfile

        import nbformat

        api = kaggle_api()
        meta = self.metadata
        if self.inputs:
            model_sources = []
            for m in self.inputs:
                if m.version <= 0:
                    m.fetch()
                model_sources.append(f"{m.slug}/{m.version}")
                if not m.wait_ready(timeout=600):
                    raise RuntimeError(f"timeout waiting for {m.slug}/{m.version}")
            meta.model_sources = model_sources

        nb = percent_to_notebook(self._build_source())
        meta_json = meta.to_json()
        print(meta_json)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            with (tmp_path / "notebook.ipynb").open("w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            (tmp_path / "kernel-metadata.json").write_text(meta_json + "\n")
            result = api.kernels_push(tmp)

        if result.error:
            raise RuntimeError(f"kernel push failed: {result.error}")
        print(f"kernel version {result.versionNumber} pushed: {result.ref}")

    def _resolve_source(self, source: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(source)
        if path.is_absolute():
            return path
        if self.sources_dir is None:
            raise ValueError(
                f"sources_dir not set but source {source!r} is relative; "
                "pass absolute paths or set KaggleNotebook(sources_dir=...)"
            )
        return self.sources_dir / path

    def _build_source(self) -> str:
        return "\n".join(self._resolve_source(s).read_text() for s in self.sources)

    def poll(self, interval: int = 10):
        """Block until kernel finishes, then print the downloaded `.log` files."""
        import tempfile

        api = kaggle_api()
        t0 = time.monotonic()
        while True:
            resp = api.kernels_status(self.slug)
            elapsed = int(time.monotonic() - t0)
            mm, ss = divmod(elapsed, 60)
            print(f"\r\033[K{mm:02d}:{ss:02d}  {resp.status}", end="", flush=True)

            status = str(resp.status).lower()
            if "complete" in status or "error" in status or "cancel" in status:
                print()
                break

            time.sleep(interval)

        with tempfile.TemporaryDirectory() as tmp:
            api.kernels_output(self.slug, tmp)
            for log in sorted(pathlib.Path(tmp).glob("*.log")):
                print(f"--- {log.name} ---")
                print(log.read_text())

        if "error" in status or "cancel" in status:
            msg = resp.failure_message or resp.status
            raise RuntimeError(f"kernel {self.slug} failed: {msg}")


KaggleNotebook.model_rebuild()
