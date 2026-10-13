# Report

- `report.md`: the technical report (reads on GitHub).
- `report.pdf`: the same report as a PDF, built with pandoc and Typst:

  ```bash
  cd report
  pixi exec -s pandoc -s typst -- pandoc report.md -o report.pdf --pdf-engine=typst \
      -V papersize=a4 -V margin.x=2.2cm -V margin.y=2.2cm -H pdf-style.typ
  ```
- `figures/`: the six figures, drawn by `scripts/R/report_figures.R` from `results/tables/`.
- `slides.pdf`: the final talk (12 October 2026).
