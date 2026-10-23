"""Download CIFAR-10 (binary version) into data/raw, verify its MD5 and extract it.

    python -m scripts.download_data

Standard library only. We use the *binary* archive, not the "python version":
that one is a pickle, and unpickling a downloaded file can run arbitrary code.
The binary format is raw bytes, parsed with NumPy in resnet/data.py.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

RAW = Path("data/raw")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"


def md5(path: Path) -> str:
    """Hex MD5 of a file, read in 1 MB chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "cifar-10-binary.tar.gz"
    if not (tgz.exists() and md5(tgz) == MD5):
        print("fetching CIFAR-10 (~170 MB) ...", flush=True)
        urllib.request.urlretrieve(URL, tgz)
        if md5(tgz) != MD5:
            tgz.unlink()
            raise RuntimeError("checksum mismatch for cifar-10-binary.tar.gz")
    print("ok  cifar-10-binary.tar.gz")
    out = RAW / "cifar-10-batches-bin"
    if not out.exists():
        with tarfile.open(tgz) as tar:
            # filter="data" rejects absolute paths, "..", links and device files in the archive
            tar.extractall(RAW, filter="data")
    print("ok ", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
