# LLM Binding Learnability

Code and data for a study of **Principle A** (reflexive binding) in large language models.

The repository contains the 384-sentence factorial benchmarks, Colab scoring notebooks, token surprisal scores from five language models, the Python analysis, and the paper figures.

## Layout

```
data/                 384 sentences with condition labels (3 × 128)
notebooks/            Colab scripts that scored the models
results/              raw surprisal CSVs at the reflexive (bits)
analysis/             run_analysis.py, diagnostics.csv, figures/
```

## Stimuli

| File | Experiment | Factors |
|---|---|---|
| `data/exp1_locality.csv` | Locality domains | Local × Matrix |
| `data/exp2_hierarchy.csv` | Structural hierarchy | Head × Distractor |
| `data/exp3_logophoric.csv` | Logophoric diagnostic | Local × Context |

## Scoring

BabyLlama and Qwen-2.5-7B were scored with `minicons` (`base_two=True`). Llama-3.1-8B, Llama-3.1-70B, and Qwen-2.5-72B used shifted cross-entropy in bits. Llama-3.1-70B was the official checkpoint in 4-bit NF4 via bitsandbytes; Qwen-2.5-72B was the Unsloth 4-bit NF4 checkpoint.

```
notebooks/score_babyllama_qwen7b.ipynb
notebooks/score_llama_qwen72b.ipynb
```

Gated Llama checkpoints need a Hugging Face login. In Colab, put the token in a secret named `HF_TOKEN`. Do not paste a token into the notebook.

## Analysis

```powershell
pip install -r requirements.txt
python analysis/run_analysis.py
```

This writes `analysis/diagnostics.csv` and `analysis/figures/` (by-item *t*-tests, df = 31).
