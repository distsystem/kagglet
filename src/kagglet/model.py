"""KaggleModel: versioned model artifact workflow.

Orchestrates fetch / build / upload / ready-polling on top of the kaggle SDK
operations in `kagglet.api.models`. `notes` (recorded server-side) drives the
cache invalidation handshake with `expected_notes()`.
"""

import time
import pathlib
from typing import ClassVar

import pydantic

from kagglet.asset import KaggleAsset
from kagglet.api.client import kaggle_api
from kagglet.api.models import (
    poll_ready,
    decode_notes,
    encode_notes,
    find_instance,
    create_version,
    create_instance,
    find_or_create_target,
)

KAGGLE_INPUT = pathlib.Path("/kaggle/input/models")


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
        inst = find_instance(kaggle_api(), self.ref, self.variation)
        if inst is None:
            raise RuntimeError(f"instance not found: {self.slug}")
        self.notes = decode_notes(inst.version_notes)
        self.version = inst.version_number

    def wait_ready(self, timeout: int = 120) -> bool:
        """Poll until the current `version` reaches READY status."""
        print(f"  waiting for {self.slug}/{self.version} ...", end="", flush=True)
        t0 = time.monotonic()
        inst = poll_ready(
            kaggle_api(),
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
        api = kaggle_api()
        notes_str = encode_notes(notes)
        target = find_or_create_target(api, self.ref, self.variation, self.owner, self.name)
        if target is None:
            create_instance(api, self.owner, self.name, self.variation, self.framework, src_dir, quiet=quiet)
            self.version = 1
        else:
            self.version = create_version(api, self.slug, src_dir, notes_str, quiet=quiet)
        self.notes = notes or {}
        print(f"uploaded {self.slug} v{self.version} [{notes_str}]")
