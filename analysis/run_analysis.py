"""Item-level contrast analysis for llm-binding-learnability.

Each model and experiment is analysed separately. Every lexical item
appears in all four cells of a 2×2, so diagnostics are item-wise
surprisal differences tested against zero with two-sided one-sample
t-tests (df = 31).

Contrasts (one set per model per experiment):
    Exp 1: local match × matrix match
    Exp 2: head match × distractor match
    Exp 3: local match × context type (co-argument vs picture-NP)

Two-level factors are coded ±0.5 (mismatch = -0.5, match = +0.5; for
context, coarg = -0.5, picture_np = +0.5). Coefficients are therefore
surprisal differences in bits. Negative = the +/match level is more
expected.

Run from the repo root:
    python analysis/run_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = Path(__file__).resolve().parent
TABLE_DIR = ANALYSIS_DIR
FIGURE_DIR = ANALYSIS_DIR / "figures"

MODEL_ORDER = [
    "babyllama",
    "qwen-2.5-7b",
    "llama-3.1-8b",
    "qwen-2.5-72b",
    "llama-3.1-70b",
]
MODEL_LABELS = {
    "babyllama": "BabyLlama",
    "qwen-2.5-7b": "Qwen-2.5-7B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen-2.5-72b": "Qwen-2.5-72B",
    "llama-3.1-70b": "Llama-3.1-70B",
}
EXPERIMENT_LABELS = {
    1: "Exp 1: Locality domains",
    2: "Exp 2: Structural hierarchy",
    3: "Exp 3: Logophoric diagnostic",
}

PALETTE_TWO = {"mismatch": "#D55E00", "match": "#0072B2"}
PALETTE_CONTEXT = {"coarg": "#D55E00", "picture_np": "#009E73"}


FILE_MODEL = {
    "babyllama": "babyllama",
    "qwen": "qwen-2.5-7b",
    "qwen7b": "qwen-2.5-7b",
    "qwen72b": "qwen-2.5-72b",
    "llama8b": "llama-3.1-8b",
    "llama70b": "llama-3.1-70b",
}


def discover_result_files() -> list[tuple[str, int, Path]]:
    found = []
    for path in RESULTS_DIR.glob("results_exp*.csv"):
        rest = path.stem.removeprefix("results_")  # exp1_babyllama
        experiment_part, _, model_part = rest.partition("_")
        if not experiment_part.startswith("exp") or not model_part:
            continue
        experiment = int("".join(ch for ch in experiment_part if ch.isdigit()))
        model = FILE_MODEL.get(model_part)
        if model is None or experiment not in EXPERIMENT_LABELS:
            continue
        found.append((model, experiment, path))
    found.sort(key=lambda row: (MODEL_ORDER.index(row[0]), row[1]))
    if not found:
        raise FileNotFoundError(f"No result CSVs found under {RESULTS_DIR}")
    return found


def annotate(df: pd.DataFrame, experiment: int) -> pd.DataFrame:
    df = df.copy()
    df["item_id"] = df["item_id"].astype(str)
    df["surprisal"] = pd.to_numeric(df["surprisal"])
    condition = df["condition"].astype(str)

    def plus_minus(flag: pd.Series) -> pd.Series:
        return np.where(flag, "match", "mismatch")

    if experiment == 1:
        df["local_match"] = plus_minus(condition.str.startswith("+Local"))
        df["matrix_match"] = plus_minus(condition.str.endswith("+Matrix"))
        df["local_c"] = df["local_match"].map({"mismatch": -0.5, "match": 0.5})
        df["matrix_c"] = df["matrix_match"].map({"mismatch": -0.5, "match": 0.5})
    elif experiment == 2:
        df["head_match"] = plus_minus(condition.str.startswith("+Head"))
        df["distractor_match"] = plus_minus(condition.str.endswith("+Distractor"))
        df["head_c"] = df["head_match"].map({"mismatch": -0.5, "match": 0.5})
        df["distractor_c"] = df["distractor_match"].map(
            {"mismatch": -0.5, "match": 0.5}
        )
    elif experiment == 3:
        df["local_match"] = plus_minus(condition.str.startswith("+Local"))
        df["context_type"] = np.where(condition.str.contains("Coarg"), "coarg", "picture_np")
        df["local_c"] = df["local_match"].map({"mismatch": -0.5, "match": 0.5})
        df["context_c"] = df["context_type"].map({"coarg": -0.5, "picture_np": 0.5})
    else:
        raise ValueError(f"Unknown experiment {experiment}")
    return df


def validate(df: pd.DataFrame, model: str, experiment: int) -> None:
    n, n_items, n_cond = len(df), df["item_id"].nunique(), df["condition"].nunique()
    if n != 128:
        print(f"Warning: {model} exp{experiment} has {n} rows (expected 128)")
    if n_items != 32:
        print(f"Warning: {model} exp{experiment} has {n_items} items (expected 32)")
    if n_cond != 4:
        print(f"Warning: {model} exp{experiment} has {n_cond} conditions (expected 4)")
    if df["surprisal"].isna().any():
        raise ValueError(f"{model} exp{experiment}: missing surprisal values")


def by_item_test(values: pd.Series) -> dict:
    """One-sample t-test of item-wise contrast scores (df = n_items - 1)."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(values))
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else math.nan
    t_val = estimate / se if se and se > 0 else math.nan
    p = float(2 * student_t.sf(abs(t_val), df=n - 1)) if se and se > 0 else math.nan
    return {"estimate": estimate, "se": se, "df": n - 1, "t": t_val, "p": p}


def wide_surprisal(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="item_id", columns="condition", values="surprisal")


def diagnostic_row(model, experiment, diagnostic, contrast, stats, note):
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "experiment": experiment,
        "diagnostic": diagnostic,
        "contrast": contrast,
        **stats,
        "note": note,
    }


def diagnostics_exp1(df: pd.DataFrame, model: str) -> list[dict]:
    wide = wide_surprisal(df)
    penalty = (
        wide[["-Local+Matrix", "-Local-Matrix"]].mean(axis=1)
        - wide[["+Local+Matrix", "+Local-Matrix"]].mean(axis=1)
    )
    rescue = wide["-Local+Matrix"] - wide["-Local-Matrix"]
    at_match = wide["+Local+Matrix"] - wide["+Local-Matrix"]
    return [
        diagnostic_row(
            model, 1, "local_mismatch_penalty", "mismatch - match (Local)",
            by_item_test(penalty),
            "Baseline: surprisal(-Local) - surprisal(+Local). Positive = local sensitivity.",
        ),
        diagnostic_row(
            model, 1, "matrix_rescue", "match - mismatch (Matrix | -Local)",
            by_item_test(rescue),
            "Critical: surprisal(+Matrix) - surprisal(-Matrix) at -Local. Negative = illicit rescue.",
        ),
        diagnostic_row(
            model, 1, "matrix_effect_at_local_match",
            "match - mismatch (Matrix | +Local)",
            by_item_test(at_match),
            "Control: matrix match effect when the local subject already matches.",
        ),
    ]


def diagnostics_exp2(df: pd.DataFrame, model: str) -> list[dict]:
    wide = wide_surprisal(df)
    penalty = (
        wide[["-Head+Distractor", "-Head-Distractor"]].mean(axis=1)
        - wide[["+Head+Distractor", "+Head-Distractor"]].mean(axis=1)
    )
    trap = wide["-Head+Distractor"] - wide["-Head-Distractor"]
    at_match = wide["+Head+Distractor"] - wide["+Head-Distractor"]
    return [
        diagnostic_row(
            model, 2, "head_mismatch_penalty", "mismatch - match (Head)",
            by_item_test(penalty),
            "Baseline: surprisal(-Head) - surprisal(+Head). Positive = structural-head sensitivity.",
        ),
        diagnostic_row(
            model, 2, "proximity_trap", "match - mismatch (Distractor | -Head)",
            by_item_test(trap),
            "Critical: surprisal(+Distractor) - surprisal(-Distractor) at -Head. Negative = linear trap.",
        ),
        diagnostic_row(
            model, 2, "distractor_effect_at_head_match",
            "match - mismatch (Distractor | +Head)",
            by_item_test(at_match),
            "Control: distractor match effect when the head already matches.",
        ),
    ]


def diagnostics_exp3(df: pd.DataFrame, model: str) -> list[dict]:
    wide = wide_surprisal(df)
    coarg = wide["-Local+Coarg"] - wide["+Local+Coarg"]
    picture = wide["-Local+Picture"] - wide["+Local+Picture"]
    return [
        diagnostic_row(
            model, 3, "mismatch_penalty_coarg", "mismatch - match (Local | coarg)",
            by_item_test(coarg),
            "surprisal(-Local) - surprisal(+Local) in co-argument contexts.",
        ),
        diagnostic_row(
            model, 3, "mismatch_penalty_picture",
            "mismatch - match (Local | picture-NP)",
            by_item_test(picture),
            "surprisal(-Local) - surprisal(+Local) in picture-NP contexts.",
        ),
        diagnostic_row(
            model, 3, "logophoric_exemption",
            "(mismatch-match | coarg) - (mismatch-match | picture-NP)",
            by_item_test(coarg - picture),
            "Positive = larger local-mismatch penalty in co-argument than picture-NP (exemption).",
        ),
    ]


def cell_means_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(frames, ignore_index=True)
    item_cells = (
        data.groupby(["model", "model_label", "experiment", "item_id", "condition"], as_index=False)
        .agg(surprisal=("surprisal", "mean"))
    )
    summary = (
        item_cells.groupby(["model", "model_label", "experiment", "condition"], as_index=False)
        .agg(n_items=("surprisal", "size"), mean=("surprisal", "mean"), sd=("surprisal", "std"))
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n_items"])
    summary["ci95_lo"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci95_hi"] = summary["mean"] + 1.96 * summary["se"]
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["experiment", "model", "condition"]).reset_index(drop=True)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def plot_interaction(means: pd.DataFrame, experiment: int, x_col: str, hue_col: str,
                     x_labels: dict, hue_labels: dict, palette: dict,
                     title: str, subtitle: str, xlabel: str, path_stem: str):
    models = [m for m in MODEL_ORDER if m in set(means["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    x_levels = list(x_labels.keys())
    x_pos = {level: i for i, level in enumerate(x_levels)}
    for ax, model in zip(axes, models):
        subset = means[means["model"] == model]
        for hue, colour in palette.items():
            rows = subset[subset[hue_col] == hue].copy()
            rows["x"] = rows[x_col].map(x_pos)
            rows = rows.sort_values("x")
            ax.plot(rows["x"], rows["mean"], color=colour, marker="o", linewidth=1.8)
            ax.errorbar(
                rows["x"], rows["mean"],
                yerr=1.96 * rows["se"],
                fmt="none", ecolor=colour, capsize=3, linewidth=1.1,
            )
        ax.set_xticks(range(len(x_levels)))
        ax.set_xticklabels([x_labels[level] for level in x_levels])
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        _style_axes(ax)
    axes[0].set_ylabel("Mean surprisal at reflexive")
    handles = [
        plt.Line2D([0], [0], color=colour, marker="o", label=hue_labels[key])
        for key, colour in palette.items()
        if key in hue_labels
    ]
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.08))
    fig.suptitle(title, y=0.98, fontsize=13)
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=9, color="#444444")
    fig.supxlabel(xlabel, y=0.01, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.14, 1, 0.88))
    fig.savefig(FIGURE_DIR / f"{path_stem}.png", dpi=150)
    fig.savefig(FIGURE_DIR / f"{path_stem}.pdf")
    plt.close(fig)


def plot_diagnostics(diagnostics: pd.DataFrame) -> None:
    # Forest plot of the five by-item t-tests reported in Table 2.
    # Each point is the mean of 32 item-wise surprisal differences (bits).
    key_order = [
        ("local_mismatch_penalty", "Exp 1: local mismatch penalty", "+"),
        ("matrix_rescue", "Exp 1: matrix rescue at −Local", "−"),
        ("head_mismatch_penalty", "Exp 2: head mismatch penalty", "+"),
        ("proximity_trap", "Exp 2: proximity trap at −Head", "−"),
        ("logophoric_exemption", "Exp 3: logophoric exemption", "+"),
    ]
    present = diagnostics[diagnostics["diagnostic"].isin({k for k, _, _ in key_order})]
    n = len(key_order)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 11), sharey=True)
    y_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
    y_pos = {label: i for i, label in enumerate(reversed(y_labels))}
    for ax, (diag, title, expected) in zip(axes, key_order):
        subset = present[present["diagnostic"] == diag]
        ax.axvline(0, color="#888888", linewidth=0.8)
        for _, row in subset.iterrows():
            y = y_pos[row["model_label"]]
            colour = "#0072B2" if row["p"] < 0.05 else "#888888"
            ax.errorbar(
                row["estimate"], y,
                xerr=1.96 * row["se"],
                fmt="o", color=colour, capsize=3,
            )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(list(reversed(y_labels)))
        ax.set_title(f"{title}  (expected {expected})", loc="left", fontsize=11)
        _style_axes(ax)
    axes[-1].set_xlabel("Mean item-wise surprisal difference (bits)")
    fig.suptitle("Diagnostic contrasts across models", fontsize=13)
    fig.text(
        0.5, 0.955,
        "Each point is the mean of 32 item-wise surprisal differences. "
        r"Error bars: 95% CI. Two-sided $t$-tests, df = 31. "
        r"Blue: $p < .05$; grey: n.s.",
        ha="center", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
    fig.savefig(FIGURE_DIR / "fig4_diagnostics.png", dpi=150)
    fig.savefig(FIGURE_DIR / "fig4_diagnostics.pdf")
    plt.close(fig)


def main() -> None:
    for path in (TABLE_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)

    frames = []
    diagnostic_parts = []

    for model, experiment, path in discover_result_files():
        df = pd.read_csv(path)
        required = {"item_id", "condition", "sentence", "target", "surprisal"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        df = annotate(df, experiment)
        df["model"] = model
        df["model_label"] = MODEL_LABELS[model]
        df["experiment"] = experiment
        validate(df, model, experiment)
        frames.append(df)

        print(f"Analysing {MODEL_LABELS[model]} / experiment {experiment}")
        if experiment == 1:
            diagnostic_parts.extend(diagnostics_exp1(df, model))
        elif experiment == 2:
            diagnostic_parts.extend(diagnostics_exp2(df, model))
        else:
            diagnostic_parts.extend(diagnostics_exp3(df, model))

    cell_means = cell_means_table(frames)

    diagnostics = pd.DataFrame(diagnostic_parts)
    diagnostics["significant"] = diagnostics["p"] < 0.05

    def support(row) -> bool | float:
        diag, est, p = row["diagnostic"], row["estimate"], row["p"]
        if p >= 0.05:
            return False if diag in {
                "matrix_rescue", "proximity_trap", "logophoric_exemption",
                "local_mismatch_penalty", "head_mismatch_penalty",
            } else math.nan
        if diag in {"matrix_rescue", "proximity_trap"}:
            return bool(est < 0)
        if diag in {"logophoric_exemption", "local_mismatch_penalty", "head_mismatch_penalty"}:
            return bool(est > 0)
        return math.nan

    diagnostics["supports_heuristic"] = diagnostics.apply(support, axis=1)
    diagnostics["model"] = pd.Categorical(diagnostics["model"], MODEL_ORDER, ordered=True)
    diagnostics = diagnostics.sort_values(["experiment", "diagnostic", "model"])
    diagnostics.to_csv(TABLE_DIR / "diagnostics.csv", index=False)

    exp1 = cell_means[cell_means["experiment"] == 1].copy()
    exp1["local_match"] = np.where(exp1["condition"].str.startswith("+Local"), "match", "mismatch")
    exp1["matrix_match"] = np.where(exp1["condition"].str.endswith("+Matrix"), "match", "mismatch")
    plot_interaction(
        exp1, 1, "local_match", "matrix_match",
        {"mismatch": "−Local", "match": "+Local"},
        {"mismatch": "−Matrix", "match": "+Matrix"},
        PALETTE_TWO,
        "Experiment 1: Locality domains",
        "Matrix rescue = lower surprisal for +Matrix when Local mismatches",
        "Local subject",
        "fig1_locality",
    )

    exp2 = cell_means[cell_means["experiment"] == 2].copy()
    exp2["head_match"] = np.where(exp2["condition"].str.startswith("+Head"), "match", "mismatch")
    exp2["distractor_match"] = np.where(
        exp2["condition"].str.endswith("+Distractor"), "match", "mismatch"
    )
    plot_interaction(
        exp2, 2, "head_match", "distractor_match",
        {"mismatch": "−Head", "match": "+Head"},
        {"mismatch": "−Distractor", "match": "+Distractor"},
        PALETTE_TWO,
        "Experiment 2: Structural hierarchy",
        "Proximity trap = lower surprisal for +Distractor when Head mismatches",
        "Head noun",
        "fig2_hierarchy",
    )

    exp3 = cell_means[cell_means["experiment"] == 3].copy()
    exp3["local_match"] = np.where(exp3["condition"].str.startswith("+Local"), "match", "mismatch")
    exp3["context_type"] = np.where(exp3["condition"].str.contains("Coarg"), "coarg", "picture_np")
    plot_interaction(
        exp3, 3, "local_match", "context_type",
        {"mismatch": "−Local", "match": "+Local"},
        {"coarg": "Co-argument", "picture_np": "Picture-NP"},
        PALETTE_CONTEXT,
        "Experiment 3: Logophoric diagnostic",
        "Exemption = smaller local-mismatch penalty in picture-NP than co-argument",
        "Local subject",
        "fig3_logophor",
    )
    plot_diagnostics(diagnostics)
    print(f"Wrote {TABLE_DIR / 'diagnostics.csv'}")
    print(f"Wrote figures to {FIGURE_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
