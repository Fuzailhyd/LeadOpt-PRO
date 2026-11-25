# LeadOpt PRO

Pfizer-grade translational CADD, lead optimization, safety profiling, and multi-parameter optimization (MPO) suite built with Python and Streamlit.

## Overview
LeadOpt PRO helps evaluate candidate molecules across structural stability, structure-activity trends, off-target/toxicity risks, pharmacokinetics, synthetic feasibility, and multi-parameter prioritization in a single interactive workflow.

## Modules
1. **MD & Pose Stability** — Evaluates MMFF94 conformer strain energy, ensemble RMSD, and binding pocket residence stability.
2. **SAR & MMP Optimization** — Explores structure-activity relationships and matched molecular pair transformations.
3. **hERG / CYP / DILI Safety** — Screens candidates for cardiac, metabolic, and hepatotoxicity risks along with structural alerts and medicinal chemistry mitigation guidance.
4. **PK / Dose & Formulation** — Supports oral pharmacokinetic, dose, and formulation assessment.
5. **Synthetic & Retrosynthesis** — Assists with synthetic accessibility and route planning.
6. **Pfizer MPO Prioritization** — Ranks lead candidates across balanced drug-like and safety criteria.

## Getting Started
```bash
pip install -r requirements.txt
streamlit run app.py
