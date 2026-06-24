# Physics-Informed Machine Learning — guest lecture

Materials for a ~50-minute guest lecture in the MIT Professional Education course
*Advanced Data Analytics for IIOT and Smart Manufacturing*.

## What's here

```
piml_presentation/
├── main.pdf                       — the paper this lecture sidebars
├── slides/
│   ├── index.html                 — Reveal.js entry point
│   ├── slides.md                  — slide content (markdown + math + speaker notes)
│   └── assets/                    — paper figures + extraction script
└── notebooks/
    ├── 01_damped_oscillator.ipynb — forward PINN + inverse parameter ID
    ├── 02_burgers.ipynb           — canonical Raissi-2019 PDE PINN
    ├── requirements.txt
    └── _build_notebooks.py        — regenerates the two notebooks from source
```

## Running the slides

The deck loads `slides.md` over HTTP, so opening `index.html` directly via `file://`
won't work. Serve the `slides/` directory with anything:

```bash
cd slides
python -m http.server 8000
# open http://localhost:8000 in a browser
```

Keyboard: `→`/`←` to advance, `s` for speaker view (notes + timer), `f` for fullscreen,
`?` for the full shortcut list.

## Running the notebooks

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
jupyter notebook
```

Both notebooks run on a laptop CPU. `01_damped_oscillator.ipynb` is ~30 seconds end-to-end;
`02_burgers.ipynb` is ~2–3 minutes (5000 Adam steps on a ~17k-parameter network).

## Regenerating things

- **Notebooks** — edit cell content in `notebooks/_build_notebooks.py`, then
  `python notebooks/_build_notebooks.py`. The `.ipynb` files are build artifacts.
- **Figures** — `python slides/assets/_extract_figures.py` re-renders the paper figures
  from `main.pdf`.

## Lecture outline

| § | Section | Time |
|---|---------|------|
| 1 | Motivation — why physics-informed ML | 5 min |
| 2 | Neural networks in 10 minutes | 10 min |
| 3 | Seminal arc — Psichogios → Raissi → Karniadakis → Rackauckas | 10 min |
| 4 | Live demo — both notebooks | 15 min |
| 5 | Sidebar — placement matters more than architecture | 5 min |
| 6 | Practical guidance + Q&A | 5 min |

## License / attribution

Slides and notebooks for educational use. The paper figures in `slides/assets/` are from
Pronk & Anthony 2025 and may be reused under the same terms as the paper.
