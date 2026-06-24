"""Render figure-bearing pages of main.pdf at high resolution, then crop to figures.

Run: `python _extract_figures.py` from inside slides/assets/.
Writes paper_fig1_spectrum.png, paper_fig2_correlation.png, paper_fig3_rollouts.png,
paper_fig4_roaster_schematic.png, paper_fig5_roast_profile.png in this directory.
"""
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

PAPER = Path(__file__).parents[2] / "main.pdf"
OUT_DIR = Path(__file__).parent

# Page index (0-based) → (output filename, crop_box in *pixel* coords on the 3x render).
# Render is 1836x2376 (standard US Letter at 3x scale).
# crop_box is (left, top, right, bottom) or None for full page.
TARGETS = {
    11: ("paper_fig1_spectrum.png", (60, 50, 1780, 1050)),       # Figure 1 plot only
    12: ("paper_fig2_correlation.png", (200, 20, 1700, 1140)),   # Figure 2 plot (incl. row 6)
    13: ("paper_fig3_rollouts.png", (130, 280, 1700, 1380)),     # Figure 3 plot only (tight)
    14: ("paper_fig4_roaster_schematic.png", (200, 280, 1700, 1750)),  # Figure 4 schematic
    15: ("paper_fig5_roast_profile.png", (60, 280, 1780, 1350)), # Figure 5 profile
}

# Extra: just Table 1 (placement spectrum results), separately for a clean slide.
TABLES = {
    11: ("paper_table1_results.png", (60, 1700, 1780, 2200)),
}


def render_and_crop(doc: fitz.Document, page_idx: int, filename: str, crop: tuple | None) -> None:
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    if crop:
        img = img.crop(crop)
    out = OUT_DIR / filename
    img.save(out)
    print(f"wrote {out}  ({img.width}x{img.height})")


def main() -> None:
    doc = fitz.open(PAPER)
    for page_idx, (filename, crop) in TARGETS.items():
        render_and_crop(doc, page_idx, filename, crop)
    for page_idx, (filename, crop) in TABLES.items():
        render_and_crop(doc, page_idx, filename, crop)
    doc.close()


if __name__ == "__main__":
    main()
