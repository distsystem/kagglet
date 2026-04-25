"""Cached kaggle API client + parallel model-file upload monkey-patch.

The official kaggle client serializes file uploads. `parallel_kaggle_uploads()`
patches `api.upload_files` for the duration of a with-block, fanning out
per-file uploads through a ThreadPoolExecutor. Useful for model artifacts
with many medium-size files.
"""

import os
import contextlib
import concurrent.futures

_api = None


def kaggle_api():
    """Return a cached, authenticated `KaggleApi` singleton."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    global _api
    if _api is None:
        _api = KaggleApi()
        _api.authenticate()
    return _api


def _patch_resumable_upload() -> None:
    """Work around kaggle client 1.7.x bug where `ResumableFileUpload.from_dict`
    mis-reconstructs the nested `start_blob_upload_request` and invalidates
    previously-resumed uploads.
    """
    import time

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


@contextlib.contextmanager
def parallel_kaggle_uploads(api, upload_threads: int = 4):
    """Context manager: temporarily patch `api.upload_files` to upload in parallel."""
    if upload_threads < 1:
        raise ValueError("upload_threads must be >= 1")
    _patch_resumable_upload()
    if upload_threads == 1:
        yield
        return

    metadata_files = {
        api.DATASET_METADATA_FILE,
        api.OLD_DATASET_METADATA_FILE,
        api.KERNEL_METADATA_FILE,
        api.MODEL_METADATA_FILE,
        api.MODEL_INSTANCE_METADATA_FILE,
    }
    original_upload_files = api.upload_files

    def upload_files_parallel(request, resources, folder, blob_type, upload_context, quiet=False, dir_mode="skip"):
        file_names = [name for name in os.listdir(folder) if name not in metadata_files]
        if len(file_names) <= 1:
            return original_upload_files(request, resources, folder, blob_type, upload_context, quiet, dir_mode)

        max_workers = min(upload_threads, len(file_names))
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
        api.upload_files = original_upload_files
