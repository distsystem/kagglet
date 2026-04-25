"""Operations against the Kaggle Models endpoint.

Free functions taking `api` (kaggle_api singleton) as the first argument; no
hidden state. The workflow layer in `kagglet.model` composes these.
"""

import json
import time
import pathlib

from kagglet.api.meta import ModelMeta, InstanceMeta


def find_instance(api, ref: str, variation: str):
    """Return the instance matching `variation` under `ref`, or None."""
    resp = api.model_instances_list(ref)
    return next((inst for inst in resp.instances or [] if inst and inst.slug == variation), None)


def create_model(api, owner: str, name: str) -> None:
    """`kaggle models create new` for `{owner}/{name}` (idempotent on 'already exists')."""
    import tempfile

    meta = ModelMeta(owner_slug=owner, slug=name, title=name)
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "model-metadata.json").write_text(meta.to_json())
        result = api.model_create_new(tmp)
        if result.error and "already" not in result.error.lower():
            raise RuntimeError(f"create model {owner}/{name} failed: {result.error}")
        print(f"created model {owner}/{name}")


def create_instance(api, owner: str, name: str, variation: str, framework: str, src_dir: str, quiet: bool) -> None:
    """`kaggle models instances create` — symlinks `src_dir` contents into a staging tmp dir."""
    import tempfile

    meta = InstanceMeta(
        owner_slug=owner,
        model_slug=name,
        instance_slug=variation,
        framework=framework,
    )
    slug = f"{owner}/{name}/{framework}/{variation}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        (tmp_path / "model-instance-metadata.json").write_text(meta.to_json())
        for f in pathlib.Path(src_dir).iterdir():
            (tmp_path / f.name).symlink_to(f)
        result = api.model_instance_create(tmp, quiet=quiet)
        if result.error:
            raise RuntimeError(f"create instance {slug} failed: {result.error}")
        print(f"created instance {slug}")


def create_version(api, slug: str, src_dir: str, version_notes: str, quiet: bool) -> int:
    """Create a new instance version. Returns the new version number parsed from `result.url`."""
    result = api.model_instance_version_create(slug, src_dir, version_notes=version_notes, quiet=quiet)
    if result.error:
        raise RuntimeError(f"upload to {slug} failed: {result.error}")
    return int(result.url.rsplit("/", 1)[-1])


def find_or_create_target(api, ref: str, variation: str, owner: str, name: str):
    """Find the variation instance; if the model doesn't exist (HTTP 403/404), create it and return None.

    None signals to the caller that the instance still needs to be created (no existing target).
    """
    import requests.exceptions

    try:
        return find_instance(api, ref, variation)
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code not in {403, 404}:
            raise
        create_model(api, owner, name)
        return None


def poll_ready(api, ref: str, variation: str, target_version: int, timeout: int, on_tick=None):
    """Poll `find_instance` until the variation reaches READY at version >= target_version.

    Returns the ready instance or None on timeout. `on_tick` runs each iteration before sleep.
    """
    from kagglesdk.datasets.types.dataset_enums import DatabundleVersionStatus

    t0 = time.monotonic()
    while True:
        try:
            inst = find_instance(api, ref, variation)
        except Exception:
            inst = None
        if inst and inst.version_number >= target_version and inst.status == DatabundleVersionStatus.READY:
            return inst
        if time.monotonic() - t0 > timeout:
            return None
        if on_tick:
            on_tick()
        time.sleep(10)


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
