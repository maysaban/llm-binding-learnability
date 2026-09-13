# Results

Raw surprisal scores, diagnostic tables, and paper figures.

```
results/
  {model}/experimentN/*.csv   surprisal at the reflexive (bits)
  tables/diagnostics.csv      planned contrast tests (Table 2)
  tables/cell_means.csv       condition means
  figures/                    Figures 1–4
```

Regenerate tables and figures from the surprisal CSVs:

```powershell
python analysis/run_analysis.py
```
