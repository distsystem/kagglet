"""Kaggle model artifact: fetch, build, upload, and wait on Kaggle models."""

import json
import time
import pathlib
from typing import ClassVar

from pydantic import Field, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from kagglet.api import kaggle_api

KAGGLE_INPUT = pathlib.Path("/kaggle/input/models")


class CamelMeta(BaseModel):
    """Base for Kaggle metadata schemas that serialize as camelCase JSON."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)


class ModelMeta(CamelMeta):
    """`model-metadata.json` body for `kaggle models create new`."""

    owner_slug: str
    slug: str
    title: str
    is_private: bool = True
    description: str = ""


class InstanceMeta(CamelMeta):
    """`model-instance-metadata.json` body for `kaggle models instances create`."""

    owner_slug: str
    model_slug: str
    instance_slug: str
    framework: str
    license_name: str = "Apache 2.0"
    overview: str = ""
    usage: str = ""


class KaggleModel(BaseModel):
    """Versioned Kaggle model artifact.

    Subclasses set `MARKER` (the filename indicating the artifact is installed)
    and override `expected_notes()` / `build()`. `notes` records what the current
    upload represents; `needs_update()` compares against `expected_notes()`.
    """

    model_config = ConfigDict(extra="forbid")

    owner: str
    name: str
    framework: str = "other"
    variation: str = "default"
    version: int = Field(default=0, repr=False)
    notes: dict = Field(default_factory=dict, repr=False)
    expect: dict = Field(default_factory=dict, repr=False)

    MARKER: ClassVar[str] = ""

    # Identity-based hash/eq so instances stay usable as dict/set keys even
    # though the model is mutable.
    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}/{self.framework}/{self.variation}"

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.name}"

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
        inst = self._find_instance(kaggle_api())
        if inst is None:
            raise RuntimeError(f"instance not found: {self.slug}")
        self.notes = _parse_notes(inst.version_notes)
        self.version = inst.version_number

    def wait_ready(self, timeout: int = 120) -> bool:
        """Poll until the current `version` reaches READY status."""
        from kagglesdk.datasets.types.dataset_enums import DatabundleVersionStatus

        api = kaggle_api()
        target = self.version
        print(f"  waiting for {self.slug}/{target} ...", end="", flush=True)
        t0 = time.monotonic()
        while True:
            try:
                inst = self._find_instance(api)
            except Exception:
                inst = None
            if inst and inst.version_number >= target and inst.status == DatabundleVersionStatus.READY:
                self.version = inst.version_number
                self.notes = _parse_notes(inst.version_notes)
                print(f" ready ({int(time.monotonic() - t0)}s)")
                return True
            if time.monotonic() - t0 > timeout:
                print(" timeout!")
                return False
            print(".", end="", flush=True)
            time.sleep(10)

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

    def _find_instance(self, api):
        resp = api.model_instances_list(self.ref)
        return next((inst for inst in resp.instances or [] if inst and inst.slug == self.variation), None)

    def _create_model(self, api) -> None:
        import tempfile

        meta = ModelMeta(owner_slug=self.owner, slug=self.name, title=self.name)
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "model-metadata.json").write_text(meta.to_json())
            result = api.model_create_new(tmp)
            if result.error and "already" not in result.error.lower():
                raise RuntimeError(f"create model {self.ref} failed: {result.error}")
            print(f"created model {self.ref}")

    def _create_instance(self, api, src_dir: str, quiet: bool) -> None:
        import tempfile

        meta = InstanceMeta(
            owner_slug=self.owner,
            model_slug=self.name,
            instance_slug=self.variation,
            framework=self.framework,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            (tmp_path / "model-instance-metadata.json").write_text(meta.to_json())
            for f in pathlib.Path(src_dir).iterdir():
                (tmp_path / f.name).symlink_to(f)
            result = api.model_instance_create(tmp, quiet=quiet)
            if result.error:
                raise RuntimeError(f"create instance {self.slug} failed: {result.error}")
            print(f"created instance {self.slug}")

    def _prepare_upload_target(self, api):
        import requests.exceptions

        try:
            return self._find_instance(api)
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {403, 404}:
                raise
            self._create_model(api)
            return None

    def upload(self, src_dir: str, notes: dict | None = None, quiet: bool = False):
        """Upload `src_dir` as the next version, creating model/instance if needed."""
        api = kaggle_api()
        notes_str = json.dumps(notes) if notes else "auto"
        instance = self._prepare_upload_target(api)
        if instance is None:
            self._create_instance(api, src_dir, quiet=quiet)
            self.version = 1
        else:
            result = api.model_instance_version_create(self.slug, src_dir, version_notes=notes_str, quiet=quiet)
            if result.error:
                raise RuntimeError(f"upload to {self.slug} failed: {result.error}")
            self.version = int(result.url.rsplit("/", 1)[-1])
        self.notes = notes or {}
        print(f"uploaded {self.slug} v{self.version} [{notes_str}]")


def _parse_notes(raw: str | None) -> dict:
    raw = raw or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
