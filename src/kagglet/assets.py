"""Kaggle resource types: shared `{owner}/{name}` identity plus dataset, kernel, and model.

`KaggleModel` also wraps the versioned-artifact workflow (fetch / build / upload /
ready-polling) on top of `kagglet.api.Kaggle`. `notes` (recorded server-side)
drives the cache-invalidation handshake with `expected_notes()`.
"""

import enum
import json
import time
import pathlib
from typing import ClassVar

import pydantic

from kagglet.api import kaggle

KAGGLE_INPUT = pathlib.Path("/kaggle/input/models")


def encode_notes(notes: dict | None) -> str:
    """Serialize the notes dict for upload (`"auto"` when empty/None)."""
    return json.dumps(notes) if notes else "auto"


def decode_notes(raw: str | None) -> dict:
    """Parse the version_notes string from the API back into a dict."""
    raw = raw or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


def split_slug(slug: str) -> tuple[str, str]:
    parts = slug.split("/", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


class KaggleAsset(pydantic.BaseModel):
    """Base identity for Kaggle assets addressed as `{owner}/{name}`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    owner: str = ""
    name: str
    version: int = pydantic.Field(default=0, repr=False)

    @pydantic.model_validator(mode="before")
    @classmethod
    def parse_slug(cls, data):
        if not isinstance(data, dict) or "slug" not in data:
            return data
        data = dict(data)
        slug = data.pop("slug")
        if "name" not in data:
            data["owner"], data["name"] = split_slug(str(slug))
        return data

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.name}" if self.owner else self.name

    @property
    def slug(self) -> str:
        return self.ref

    @property
    def versioned_slug(self) -> str:
        return f"{self.slug}/{self.version}"


class KaggleDataset(KaggleAsset):
    """Dataset identity placeholder for DAG inputs and future dataset workflows."""


class Accelerator(enum.StrEnum):
    NONE = ""
    GPU_H100 = "H100"
    GPU_P100 = "P100"
    GPU_T4_X2 = "T4x2"
    TPU_V5E8 = "TpuV5E8"

    @property
    def uses_gpu(self) -> bool:
        return bool(self.value) and not self.uses_tpu

    @property
    def uses_tpu(self) -> bool:
        return self.value.lower().startswith("tpu")

    @property
    def machine_shape(self) -> str | None:
        return self.value or None


class KaggleKernel(KaggleAsset):
    """Kaggle kernel resource metadata."""

    title: str = ""
    dataset_sources: list[str] = pydantic.Field(default_factory=list)
    model_sources: list[str] = pydantic.Field(default_factory=list)
    internet: bool = True
    competition: str = ""
    accelerator: Accelerator = Accelerator.NONE

    @property
    def display_title(self) -> str:
        return self.title or self.name.replace("-", " ").replace("_", " ")

    def save_request(self, text: str):
        """Build an `ApiSaveKernelRequest` for `Kaggle.push_kernel`."""
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

        request = ApiSaveKernelRequest()
        request.slug = self.slug
        request.new_title = self.display_title
        request.text = text
        request.language = "python"
        request.kernel_type = "notebook"
        request.is_private = True
        request.enable_gpu = self.accelerator.uses_gpu
        request.enable_tpu = self.accelerator.uses_tpu
        if self.accelerator.machine_shape is not None:
            request.machine_shape = self.accelerator.machine_shape
        request.enable_internet = self.internet
        request.dataset_data_sources = list(self.dataset_sources)
        if self.competition:
            request.competition_data_sources = [self.competition]
        request.model_data_sources = list(self.model_sources)
        return request


class KaggleModel(KaggleAsset):
    """Versioned Kaggle model artifact.

    Subclasses set `MARKER` (the filename indicating the artifact is installed)
    and override `expected_notes()` / `build()`. `notes` records what the current
    upload represents; `needs_update()` compares against `expected_notes()`.
    """

    framework: str = "other"
    variation: str = "default"
    notes: dict = pydantic.Field(default_factory=dict, repr=False)
    expect: dict = pydantic.Field(default_factory=dict, repr=False)

    MARKER: ClassVar[str] = ""

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}/{self.framework}/{self.variation}"

    def find(self) -> pathlib.Path:
        """Locate the artifact marker under KAGGLE_INPUT (raises if missing)."""
        path = self.version_path() / self.MARKER
        if not path.exists():
            raise FileNotFoundError(f"{self.MARKER!r} not found at {path}")
        return path

    def version_path(self) -> pathlib.Path:
        root = KAGGLE_INPUT / self.slug
        versions = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
        if not versions:
            raise FileNotFoundError(f"no version directories found under {root}")
        return max(versions, key=lambda path: int(path.name))

    def fetch(self):
        """Pull latest version + notes from Kaggle into this instance."""
        inst = kaggle().find_instance(self.ref, self.variation)
        if inst is None:
            raise RuntimeError(f"instance not found: {self.slug}")
        self.notes = decode_notes(inst.version_notes)
        self.version = inst.version_number

    def wait_ready(self, timeout: int = 120) -> bool:
        """Poll until the current `version` reaches READY status."""
        print(f"  waiting for {self.slug}/{self.version} ...", end="", flush=True)
        t0 = time.monotonic()
        inst = kaggle().poll_ready(
            self.ref,
            self.variation,
            self.version,
            timeout,
            on_tick=lambda: print(".", end="", flush=True),
        )
        if inst is None:
            print(" timeout!")
            return False
        self.version = inst.version_number
        self.notes = decode_notes(inst.version_notes)
        print(f" ready ({int(time.monotonic() - t0)}s)")
        return True

    def expected_notes(self) -> dict:
        """Notes describing the build this subclass would produce now.

        Compared against `self.notes` (remote) by `needs_update()`.
        """
        return dict(self.expect)

    def needs_update(self) -> bool:
        expected = self.expected_notes()
        return any(self.notes.get(k) != v for k, v in expected.items())

    def upload_file(self, artifact: pathlib.Path, notes: dict | None = None):
        """Upload a single file as this model version (wraps `upload` in a tmp dir)."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(artifact, pathlib.Path(tmp) / artifact.name)
            self.upload(tmp, notes=notes)

    def upload(self, src_dir: str, notes: dict | None = None, quiet: bool = False):
        """Upload `src_dir` as the next version, creating model/instance if needed."""
        api = kaggle()
        notes_str = encode_notes(notes)
        target = api.find_or_create_target(self.ref, self.variation, self.owner, self.name)
        if target is None:
            api.create_instance(self.owner, self.name, self.variation, self.framework, src_dir, quiet=quiet)
            self.version = 1
        else:
            self.version = api.create_version(self.slug, src_dir, notes_str, quiet=quiet)
        self.notes = notes or {}
        print(f"uploaded {self.slug} v{self.version} [{notes_str}]")
