"""Operations against the Kaggle Kernels endpoint."""

import time
import pathlib


def push_kernel(api, meta_json: str, notebook):
    """Stage `notebook.ipynb` + `kernel-metadata.json` in a temp dir and call `api.kernels_push`.

    Returns the API result. Raises on `result.error`.
    """
    import tempfile

    import nbformat

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        with (tmp_path / "notebook.ipynb").open("w", encoding="utf-8") as f:
            nbformat.write(notebook, f)
        (tmp_path / "kernel-metadata.json").write_text(meta_json + "\n")
        result = api.kernels_push(tmp)
    if result.error:
        raise RuntimeError(f"kernel push failed: {result.error}")
    return result


def poll_kernel_terminal(api, slug: str, interval: int, on_tick=None):
    """Poll `kernels_status` until status hits a terminal state (complete/error/cancel).

    Returns `(resp, status_lower)`. `on_tick(resp)` runs each non-terminal iteration before sleep.
    """
    while True:
        resp = api.kernels_status(slug)
        status = str(resp.status).lower()
        if "complete" in status or "error" in status or "cancel" in status:
            return resp, status
        if on_tick:
            on_tick(resp)
        time.sleep(interval)


def fetch_kernel_logs(api, slug: str) -> list[tuple[str, str]]:
    """Download kernel output and return `[(name, content), ...]` for `*.log` files (sorted)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        api.kernels_output(slug, tmp)
        return [(p.name, p.read_text()) for p in sorted(pathlib.Path(tmp).glob("*.log"))]
