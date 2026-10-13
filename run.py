"""Validate local inputs and launch an original anomaly experiment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    'full-memory': 'patchcore_full_memory.py',
    'dinov2': 'dinov2_patchcore_v4.py',
}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=SCRIPTS)
    args = parser.parse_args()
    for name in ('train_data', 'test_data'):
        folder = ROOT / name
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS] if folder.is_dir() else []
        if not images:
            parser.error(f'{name}/ contains no supported images. See docs/SETUP.md.')
    script = ROOT / 'experiments' / SCRIPTS[args.method]
    return subprocess.call([sys.executable, str(script)], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
