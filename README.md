# LLM Binding Learnability

Code and data for a study of **Principle A** (reflexive binding) in large language models (Saban, 2026).

The repository contains the 384-sentence factorial benchmarks, token surprisal scores from five language models, the Python analysis that produced Table 2 / Figure 4, and the paper figures.

## Layout

```
data/                 384 sentences with condition labels (3 × 128)
analysis/             run_analysis.py (by-item t-tests, df = 31)
results/{model}/      raw surprisal CSVs at the reflexive token (bits)
results/tables/       diagnostics.csv and cell means
results/figures/      Figures 1–4 (PNG and PDF)
```

## Reproduce the tables and figures

```powershell
pip install -r analysis/requirements.txt
python analysis/run_analysis.py
```

Outputs are written to `results/tables/` and `results/figures/`.

## Stimuli

Each experiment is a balanced 2×2 with 32 lexical items × 4 conditions:

| File | Experiment | Factors |
|---|---|---|
| `data/experiment1.csv` | Locality domains | Local × Matrix |
| `data/experiment2.csv` | Structural hierarchy | Head × Distractor |
| `data/experiment3.csv` | Logophoric diagnostic | Local × Context |
| `data/all_sentences.csv` | All 384 evaluation sentences | |

Stimulus-generation code lives in `src/`.

## Models

BabyLlama (100M), Qwen-2.5-7B/72B, Llama-3.1-8B/70B. Surprisal is measured in bits at the target reflexive (`himself` / `herself` / `themselves`).
