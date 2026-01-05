#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def _resolve_notebook(notebooks_dir: Path, pattern: str) -> Path:
    matches = sorted(notebooks_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 notebook for pattern '{pattern}', found {len(matches)}: {matches}"
        )
    return matches[0]


def _execute_with_nbclient(nb_path: Path, output_path: Path) -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, output_path)


def _execute_with_nbconvert(nb_path: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        nb_path.name,
        "--output-dir",
        str(output_dir),
        str(nb_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(nb_path.parent))


def run_notebooks() -> None:
    notebooks_dir = Path(__file__).resolve().parent
    output_dir = notebooks_dir / "_executed"
    output_dir.mkdir(exist_ok=True)

    notebooks = [
        _resolve_notebook(notebooks_dir, "P6_MANET_Stephane_notebook_exploration.ipynb"),
        _resolve_notebook(notebooks_dir, "P6_MANET_Stephane_notebook_compare_tuning_mlflow.ipynb"),
        _resolve_notebook(notebooks_dir, "P6_MANET_Stephane_notebook_mod*.ipynb"),
    ]

    total = len(notebooks)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run of {total} notebooks")
    for idx, nb_path in enumerate(notebooks, start=1):
        start = time.time()
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Launching notebook {idx}/{total}: {nb_path.name}"
        )
        output_path = output_dir / nb_path.name
        try:
            _execute_with_nbclient(nb_path, output_path)
        except ModuleNotFoundError as exc:
            if exc.name not in {"nbclient", "nbformat"}:
                raise
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                "nbclient not available; falling back to nbconvert."
            )
            _execute_with_nbconvert(nb_path, output_dir)
        elapsed = time.time() - start
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Finished {nb_path.name} in {elapsed:.1f}s"
        )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All notebooks completed")


if __name__ == "__main__":
    run_notebooks()
