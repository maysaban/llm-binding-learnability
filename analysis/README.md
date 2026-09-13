# Analysis

`run_analysis.py` loads surprisal CSVs from `results/{model}/experimentN/` and tests planned item-wise contrasts with two-sided one-sample *t*-tests (df = 31).

```powershell
pip install -r analysis/requirements.txt
python analysis/run_analysis.py
```

In this balanced 32×4 within-item design, those tests match random-intercept LME simple effects for `(1 | item_id)`.

Outputs: `results/tables/diagnostics.csv`, `results/tables/cell_means.csv`, `results/figures/fig1_locality.*`–`fig4_diagnostics.*`.
