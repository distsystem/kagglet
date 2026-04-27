"""Kagglet's facade over the kaggle SDK.

`Kaggle` wraps the upstream `KaggleApi` singleton, exposes the kernel and model
operations kagglet uses as methods, and ships the `parallel_uploads()` context
manager (monkey-patches `upload_files` to fan out per-file uploads through a
ThreadPoolExecutor — the upstream client serializes them).

Where the upstream high-level method is just "read a JSON metadata file from a
folder, build a kagglesdk request, send it" (push_kernel, create_model,
list kernel output), we bypass the temp-dir round-trip and call
`KaggleClient.<service>.<method>(request)` directly. File uploads still go
through `KaggleApi.upload_files` because the resumable + retry plumbing isn't
worth duplicating.

`kaggle()` returns a cached, authenticated instance.

`InstanceMeta` is the one Pydantic schema kept: `model_instance_create` insists
on a folder containing both `model-instance-metadata.json` and the artifact
files together, so `Kaggle.create_instance` still stages a temp dir.
"""

import os
import time
import pathlib
import contextlib
import concurrent.futures

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class InstanceMeta(BaseModel):
    """`model-instance-metadata.json` body for `kaggle models instances create`."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    owner_slug: str
    model_slug: str
    instance_slug: str
    framework: str
    license_name: str = "Apache 2.0"
    overview: str = ""
    usage: str = ""

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)


def _patch_resumable_upload() -> None:
    """Work around kaggle client 1.7.x bug where `ResumableFileUpload.from_dict`
    mis-reconstructs the nested `start_blob_upload_request` and invalidates
    previously-resumed uploads.
    """
    from kaggle.api import kaggle_api_extended

    def from_dict_fixed(other, context):
        request = kaggle_api_extended.ApiStartBlobUploadRequest.from_dict(other["start_blob_upload_request"])
        upload = kaggle_api_extended.ResumableFileUpload(other["path"], request, context)
        upload.timestamp = other.get("timestamp")

        response = other.get("start_blob_upload_response")
        if response is not None:
            upload.start_blob_upload_response = kaggle_api_extended.ApiStartBlobUploadResponse.from_dict(response)
            upload.upload_complete = False
        return upload

    def is_previous_valid_fixed(self, previous):
        return (
            previous.path == self.path
            and previous.start_blob_upload_request.to_dict() == self.start_blob_upload_request.to_dict()
            and previous.timestamp
            > time.time() - kaggle_api_extended.ResumableFileUpload.RESUMABLE_UPLOAD_EXPIRY_SECONDS
        )

    kaggle_api_extended.ResumableFileUpload.from_dict = staticmethod(from_dict_fixed)
    kaggle_api_extended.ResumableFileUpload._is_previous_valid = is_previous_valid_fixed


class Kaggle:
    """Authenticated facade over the kaggle SDK; methods are kagglet's working set."""

    def __init__(self) -> None:
        os.environ.setdefault("KAGGLE_ENABLE_OAUTH", "1")
        from kaggle.api.kaggle_api_extended import KaggleApi

        self._api = KaggleApi(enable_oauth=True)
        self._api.authenticate()
        _patch_resumable_upload()

    @property
    def raw(self):
        """The underlying upstream `KaggleApi` instance for ad-hoc calls."""
        return self._api

    @property
    def username(self) -> str:
        return self._api.config_values.get("username", "")

    @property
    def auth_method(self) -> str:
        return self._api.config_values.get("auth_method", "")

    # ── kernel ops ─────────────────────────────────────────────────────────

    def push_kernel(self, request):
        """Send a pre-built `ApiSaveKernelRequest` via `kernels_api_client.save_kernel`."""
        with self._api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.save_kernel(request)
        if response.error:
            raise RuntimeError(f"kernel push failed: {response.error}")
        return response

    def poll_kernel_terminal(self, slug: str, interval: int, on_tick=None):
        """Poll `kernels_status` until status hits a terminal state.

        Returns `(resp, status_lower)`. `on_tick(resp)` runs each non-terminal iteration.
        """
        while True:
            resp = self._api.kernels_status(slug)
            status = str(resp.status).lower()
            if "complete" in status or "error" in status or "cancel" in status:
                return resp, status
            if on_tick:
                on_tick(resp)
            time.sleep(interval)

    def fetch_kernel_logs(self, slug: str) -> list[tuple[str, str]]:
        """Return `[(name, content), ...]` pairs for the kernel session log + any `*.log` output files.

        Reads everything from `list_kernel_session_output` directly — no disk staging.
        """
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        owner_slug, kernel_slug = slug.split("/", 1)
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner_slug
        request.kernel_slug = kernel_slug

        with self._api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)

        results: list[tuple[str, str]] = []
        if response.log:
            results.append((f"{kernel_slug}.log", response.log))
        for item in response.files or []:
            if item and item.file_name.endswith(".log"):
                import requests

                results.append((item.file_name, requests.get(item.url).text))
        return sorted(results)

    # ── model ops ──────────────────────────────────────────────────────────

    def find_instance(self, ref: str, variation: str):
        """Return the instance matching `variation` under `ref`, or None."""
        resp = self._api.model_instances_list(ref)
        return next((inst for inst in resp.instances or [] if inst and inst.slug == variation), None)

    def create_model(self, owner: str, name: str) -> None:
        """Create `{owner}/{name}` via `model_api_client.create_model` (idempotent on 'already exists')."""
        from kagglesdk.models.types.model_api_service import ApiCreateModelRequest

        request = ApiCreateModelRequest()
        request.owner_slug = owner
        request.slug = name
        request.title = name
        request.is_private = True
        request.description = ""

        with self._api.build_kaggle_client() as client:
            response = client.models.model_api_client.create_model(request)
        if response.error and "already" not in response.error.lower():
            raise RuntimeError(f"create model {owner}/{name} failed: {response.error}")
        print(f"created model {owner}/{name}")

    def create_instance(self, owner: str, name: str, variation: str, framework: str, src_dir: str, quiet: bool = False) -> None:
        """`kaggle models instances create` — symlinks `src_dir` contents into a staging tmp dir."""
        import tempfile

        meta = InstanceMeta(owner_slug=owner, model_slug=name, instance_slug=variation, framework=framework)
        slug = f"{owner}/{name}/{framework}/{variation}"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            (tmp_path / "model-instance-metadata.json").write_text(meta.to_json())
            for f in pathlib.Path(src_dir).iterdir():
                (tmp_path / f.name).symlink_to(f)
            result = self._api.model_instance_create(tmp, quiet=quiet)
            if result.error:
                raise RuntimeError(f"create instance {slug} failed: {result.error}")
            print(f"created instance {slug}")

    def create_version(self, slug: str, src_dir: str, version_notes: str, quiet: bool = False) -> int:
        """Create a new instance version. Returns the new version number parsed from `result.url`."""
        result = self._api.model_instance_version_create(slug, src_dir, version_notes=version_notes, quiet=quiet)
        if result.error:
            raise RuntimeError(f"upload to {slug} failed: {result.error}")
        return int(result.url.rsplit("/", 1)[-1])

    def find_or_create_target(self, ref: str, variation: str, owner: str, name: str):
        """Find the variation instance; create the model on HTTP 403/404 and return None.

        None signals the caller that the instance still needs creating (no existing target).
        """
        import requests.exceptions

        try:
            return self.find_instance(ref, variation)
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in {403, 404}:
                raise
            self.create_model(owner, name)
            return None

    def poll_ready(self, ref: str, variation: str, target_version: int, timeout: int, on_tick=None):
        """Poll `find_instance` until the variation reaches READY at version >= target_version."""
        from kagglesdk.datasets.types.dataset_enums import DatabundleVersionStatus

        t0 = time.monotonic()
        while True:
            try:
                inst = self.find_instance(ref, variation)
            except Exception:
                inst = None
            if inst and inst.version_number >= target_version and inst.status == DatabundleVersionStatus.READY:
                return inst
            if time.monotonic() - t0 > timeout:
                return None
            if on_tick:
                on_tick()
            time.sleep(10)

    # ── uploads ────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def parallel_uploads(self, threads: int = 4):
        """Patch `upload_files` to upload non-metadata model files in parallel for the with-block."""
        if threads < 1:
            raise ValueError("threads must be >= 1")
        if threads == 1:
            yield
            return

        api = self._api
        metadata_files = {
            api.DATASET_METADATA_FILE,
            api.OLD_DATASET_METADATA_FILE,
            api.KERNEL_METADATA_FILE,
            api.MODEL_METADATA_FILE,
            api.MODEL_INSTANCE_METADATA_FILE,
        }
        original = api.upload_files

        def upload_files_parallel(request, resources, folder, blob_type, upload_context, quiet=False, dir_mode="skip"):
            file_names = [name for name in os.listdir(folder) if name not in metadata_files]
            if len(file_names) <= 1:
                return original(request, resources, folder, blob_type, upload_context, quiet, dir_mode)

            max_workers = min(threads, len(file_names))
            if not quiet:
                print(f"Uploading {len(file_names)} files with {max_workers} workers")

            def upload_one(file_name: str):
                return api._upload_file_or_folder(folder, file_name, blob_type, upload_context, dir_mode, quiet, resources)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                uploaded = list(executor.map(upload_one, file_names))

            if request.files is None:
                return None
            for upload_file in uploaded:
                if upload_file is not None:
                    request.files.append(api._new_file(upload_file))
            return None

        api.upload_files = upload_files_parallel
        try:
            yield
        finally:
            api.upload_files = original


_singleton: Kaggle | None = None


def kaggle() -> Kaggle:
    """Return a cached, authenticated `Kaggle` singleton."""
    global _singleton
    if _singleton is None:
        _singleton = Kaggle()
    return _singleton
