"""Extract the most slide-worthy plots from the executed demo notebooks
and save them as PNG files for embedding in the slide deck.

We don't re-train anything — the notebooks were already executed with
matplotlib outputs cached as base64 PNGs. We just pull them out.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parent.parent.parent / "notebooks"
ASSETS = Path(__file__).parent

# Each entry: (notebook filename, code-cell index, output filename)
# Notebooks are 0-indexed cells. We pick the cell whose plot best previews the demo.
TARGETS = [
    # Damped oscillator — the killer plot is the PINN-vs-plain comparison.
    # Code-cell indices (skipping markdown cells):
    #   0 = imports          (no plot)
    #   1 = ground truth     (just the data plot)
    #   2 = plain NN train   (plain MLP comparison)
    #   3 = PINN train       (the side-by-side: plain vs PINN)
    #   4 = inverse PINN     (c-history converging)
    ("01_damped_oscillator.ipynb", 3, "demo1_damped_oscillator.png"),
    ("01_damped_oscillator.ipynb", 4, "demo1_inverse_problem.png"),
    # Burgers — the heatmap + snapshots cell is the headline.
    # Code-cell indices:
    #   0 = imports
    #   1 = sample regions plot
    #   2 = model def         (no plot)
    #   3 = training          (just text)
    #   4 = heatmap + snapshots  ← headline plot
    #   5 = loss curve
    ("02_burgers.ipynb", 4, "demo2_burgers_heatmap.png"),
]


def extract_plot(nb_path: Path, code_index: int, out_path: Path) -> None:
    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)

    # Walk the cells, counting only code cells.
    seen = 0
    target_cell = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if seen == code_index:
            target_cell = cell
            break
        seen += 1

    if target_cell is None:
        raise RuntimeError(f"{nb_path.name}: only {seen} code cells, need index {code_index}")

    # Find the last image/png output (skip text outputs and stderr)
    png_b64 = None
    for out in target_cell.get("outputs", []):
        if "data" in out and "image/png" in out["data"]:
            png_b64 = out["data"]["image/png"]
    if png_b64 is None:
        raise RuntimeError(f"{nb_path.name} code cell {code_index}: no image/png output")

    out_path.write_bytes(base64.b64decode(png_b64))
    print(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    for nb_name, code_index, out_name in TARGETS:
        extract_plot(NOTEBOOKS / nb_name, code_index, ASSETS / out_name)


if __name__ == "__main__":
    main()
