#!/usr/bin/env python3
"""Extract the pinned NVIDIA C++ SDK and tools without changing system packages."""
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path,
                        default=Path.home() / 'Lib/TensorRT-10.13.3.9-cuda13.0')
    parser.add_argument('--downloads', type=Path,
                        default=repo / 'build/deps/tensorrt-download-10.13.3.9')
    args = parser.parse_args()
    if platform.machine() != 'x86_64':
        parser.error('This lock file is for Linux x86_64 only')
    prefix = args.prefix.resolve()
    if prefix.exists():
        parser.error(f'Prefix already exists; choose a new directory: {prefix}')
    lock = json.loads((repo / 'tools/tensorrt-packages.lock.json').read_text())
    downloads = args.downloads.resolve()
    downloads.mkdir(parents=True, exist_ok=True)
    for package in lock['packages']:
        path = downloads / Path(package['Filename']).name
        if not path.exists():
            subprocess.run(['apt-get', 'download', package['Package'] + '=' + package['Version']],
                           cwd=downloads, check=True)
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != int(package['Size']) or digest.hexdigest() != package['SHA256']:
            raise ValueError(f'Package integrity check failed: {path}')
        print(f'Verified {path.name}', flush=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.tensorrt-stage-', dir=prefix.parent) as temp:
        stage = Path(temp) / 'sdk'
        stage.mkdir()
        for package in lock['packages']:
            subprocess.run(['dpkg-deb', '-x', str(downloads / Path(package['Filename']).name),
                            str(stage)], check=True)
        # The large static development archives are unnecessary for this shared-library SDK.
        (stage / 'include').symlink_to('usr/include/x86_64-linux-gnu')
        (stage / 'lib').symlink_to('usr/lib/x86_64-linux-gnu')
        (stage / 'bin').symlink_to('usr/src/tensorrt/bin')
        (stage / 'lib/libnvinfer.so').symlink_to('libnvinfer.so.10')
        if not (stage / 'include/NvInfer.h').is_file() or not (stage / 'lib/libnvinfer.so').is_file():
            raise ValueError('Unexpected SDK layout')
        (stage / 'supervo-sdk-manifest.json').write_text(json.dumps(lock, indent=2) + '\n')
        stage.rename(prefix)
    print(f'Installed C++ shared-library SDK and trtexec at {prefix}')
    print(f'Configure with -DTENSORRT_ROOT={prefix}')


if __name__ == '__main__':
    main()
