"""Tar / tar.zst extraction strategies for Kaggle input mounts.

`/kaggle/input` is a slow NFS; extraction strategy matters. This module offers
four interchangeable strategies.
"""

import enum
import pathlib


class TarExtractor(enum.Enum):
    ZSTANDARD = "zstandard + tarfile"
    PLAIN = "plain tarfile"
    CLI = "tar --zstd"
    PIPE = "zstandard → tar pipe"

    @property
    def zstd(self) -> bool:
        return self != TarExtractor.PLAIN

    def extract(self, tar_path: pathlib.Path, dest: pathlib.Path) -> None:
        import tarfile

        match self:
            case TarExtractor.ZSTANDARD:
                import zstandard

                with open(tar_path, "rb") as fh:
                    reader = zstandard.ZstdDecompressor().stream_reader(fh)
                    with tarfile.open(fileobj=reader, mode="r|") as tf:
                        tf.extractall(dest, filter="tar")
            case TarExtractor.PLAIN:
                with tarfile.open(tar_path, "r:") as tf:
                    tf.extractall(dest, filter="tar")
            case TarExtractor.CLI:
                import subprocess

                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run(["tar", "-x", "--zstd", "-f", str(tar_path), "-C", str(dest)], check=True)
            case TarExtractor.PIPE:
                import subprocess

                import zstandard

                dest.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(["tar", "-xf", "-", "-C", str(dest)], stdin=subprocess.PIPE)
                with open(tar_path, "rb") as fh:
                    zstandard.ZstdDecompressor().copy_stream(fh, proc.stdin)
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"tar extraction failed: {tar_path}")
