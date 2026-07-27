from __future__ import annotations

"""Canonical paper-reproduction pipeline for the OSD-914 liver-brain analysis."""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim


ROOT = Path(__file__).resolve().parent
RAW_MIRNA = ROOT / "GSE294046_miRNA_complete_quantification_raw.tsv.gz"
BRAIN_SERIES = ROOT / "GSE295428_series_matrix.txt.gz"

HIGH_CONFIDENCE_LIVER_MIRNAS = [
    "mmu-miR-122-5p",
    "mmu-miR-17-5p",
    "mmu-miR-18a-5p",
    "mmu-miR-19a-5p",
    "mmu-miR-20a-5p",
    "mmu-miR-92a-1-5p",
]

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
TOP_BRAIN_TARGETS = 5


class LiverBrainCrossTalkMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16, dropout_rate: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class Scaler:
    mean: pd.Series
    std: pd.Series


def log(message: str) -> None:
    print(f"[reproduce] {message}", flush=True)


def project_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def extract_subject_id(sample_name: object) -> str:
    sample_name = str(sample_name)
    tissue_match = re.match(r"^\d+_(?:Liver|Brain)_(.+?)_L1_1$", sample_name)
    if tissue_match:
        return tissue_match.group(1)

    mouse_match = re.search(r"(Mouse[_ ]\d+|M\d+)", sample_name)
    if mouse_match:
        return mouse_match.group(1).replace(" ", "_")

    return sample_name


def cohort(sample_id: str) -> str:
    return sample_id.split("-", 1)[0]


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fit_scaler(df: pd.DataFrame) -> Scaler:
    transformed = np.log1p(df)
    std = transformed.std(axis=0, ddof=0).replace(0, 1)
    return Scaler(mean=transformed.mean(axis=0), std=std)


def transform_with_scaler(df: pd.DataFrame, scaler: Scaler) -> pd.DataFrame:
    transformed = np.log1p(df)
    return (transformed - scaler.mean) / scaler.std


def rebuild_clean_matrices(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    missing_raw_files = [path for path in [RAW_MIRNA, BRAIN_SERIES] if not path.exists()]
    if missing_raw_files:
        missing = ", ".join(project_path(path) for path in missing_raw_files)
        raise FileNotFoundError(
            "Missing required raw input file(s): "
            f"{missing}. Place the gzipped source files in the repository root before running."
        )

    clean_dir = output_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    log("Rebuilding clean matrices from raw gz files.")
    mirna_raw = pd.read_csv(RAW_MIRNA, sep="\t", index_col=0)
    brain_raw = pd.read_csv(BRAIN_SERIES, sep="\t", comment="!", index_col=0)

    liver_columns = [col for col in mirna_raw.columns if re.match(r"^\d+_Liver_", col)]
    brain_columns = [col for col in mirna_raw.columns if re.match(r"^\d+_Brain_", col)]
    if not liver_columns:
        raise ValueError("No liver sample columns were found in the miRNA matrix.")
    if not brain_columns:
        raise ValueError("No brain sample columns were found in the miRNA matrix.")

    liver_raw = mirna_raw[liver_columns]
    min_samples_threshold = max(1, int(0.10 * liver_raw.shape[1]))
    liver_filtered = liver_raw[(liver_raw >= 5).sum(axis=1) >= min_samples_threshold]

    target_brain_genes = ["Sod1", "Sod2", "Cat", "Gpx1", "Gpx4", "Nfkb1"]
    available_genes = [gene for gene in target_brain_genes if gene in brain_raw.index]
    if available_genes:
        brain_filtered = brain_raw.loc[available_genes]
        brain_source = "GSE295428 target genes"
    else:
        brain_filtered = mirna_raw.loc[liver_filtered.index, brain_columns]
        brain_filtered = brain_filtered[(brain_filtered >= 5).sum(axis=1) >= min_samples_threshold]
        brain_source = "GSE294046 brain miRNA columns"

    liver_features = liver_filtered.T
    brain_targets = brain_filtered.T
    liver_features.index = liver_features.index.map(extract_subject_id)
    brain_targets.index = brain_targets.index.map(extract_subject_id)

    matching_subjects = liver_features.index.intersection(brain_targets.index)
    if matching_subjects.empty:
        raise ValueError("No matching liver/brain subjects were found after ID alignment.")

    liver_final = liver_features.loc[matching_subjects].sort_index()
    brain_final = brain_targets.loc[matching_subjects].sort_index()

    liver_path = clean_dir / "liver_features_clean.csv"
    brain_path = clean_dir / "brain_targets_clean.csv"
    liver_final.to_csv(liver_path)
    brain_final.to_csv(brain_path)

    metadata = {
        "raw_mirna_rows": int(mirna_raw.shape[0]),
        "raw_mirna_sample_columns": int(mirna_raw.shape[1]),
        "raw_liver_sample_columns": len(liver_columns),
        "raw_brain_sample_columns": len(brain_columns),
        "count_filter_min_samples": int(min_samples_threshold),
        "clean_liver_shape": list(liver_final.shape),
        "clean_brain_shape": list(brain_final.shape),
        "brain_source": brain_source,
        "clean_liver_sha256": sha256(liver_path),
        "clean_brain_sha256": sha256(brain_path),
    }
    return liver_final, brain_final, metadata


def select_model_matrices(liver_df: pd.DataFrame, brain_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    matching_subjects = liver_df.index.intersection(brain_df.index)
    liver_df = liver_df.loc[matching_subjects].sort_index()
    brain_df = brain_df.loc[matching_subjects].sort_index()

    selected_liver = [feature for feature in HIGH_CONFIDENCE_LIVER_MIRNAS if feature in liver_df.columns]
    missing = sorted(set(HIGH_CONFIDENCE_LIVER_MIRNAS) - set(selected_liver))
    if missing:
        raise ValueError(f"Missing expected liver miRNAs: {', '.join(missing)}")

    return liver_df[selected_liver], brain_df, selected_liver


def rank_brain_targets_by_training_variance(
    brain_df: pd.DataFrame,
    train_samples: Iterable[str],
) -> pd.Series:
    train_samples = list(train_samples)
    if len(train_samples) < 2:
        raise ValueError("At least two training samples are required to rank brain targets by variance.")
    return brain_df.loc[train_samples].var(axis=0).sort_values(ascending=False, kind="mergesort")


def select_fold_brain_targets(
    brain_df: pd.DataFrame,
    train_samples: Iterable[str],
    top_n: int = TOP_BRAIN_TARGETS,
) -> list[str]:
    ranked = rank_brain_targets_by_training_variance(brain_df, train_samples)
    if ranked.shape[0] < top_n:
        raise ValueError(f"Need at least {top_n} brain targets, found {ranked.shape[0]}.")
    return ranked.head(top_n).index.tolist()


def build_fold_target_selection(
    brain_df: pd.DataFrame,
    samples: Iterable[str],
    top_n: int = TOP_BRAIN_TARGETS,
) -> tuple[pd.DataFrame, dict]:
    samples = list(samples)
    full_cohort_reference = select_fold_brain_targets(brain_df, samples, top_n=top_n)
    fold_rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_index, sample_id in enumerate(samples, start=1):
        train_samples = [sample for sample in samples if sample != sample_id]
        ranked = rank_brain_targets_by_training_variance(brain_df, train_samples).head(top_n)
        selected_targets = ranked.index.tolist()
        target_set_key = ";".join(selected_targets)
        fold_summaries.append(
            {
                "fold": fold_index,
                "held_out_sample_id": sample_id,
                "train_sample_count": len(train_samples),
                "selected_brain_targets": selected_targets,
                "matches_full_cohort_reference": selected_targets == full_cohort_reference,
            }
        )
        for rank, (target, training_variance) in enumerate(ranked.items(), start=1):
            fold_rows.append(
                {
                    "fold": fold_index,
                    "held_out_sample_id": sample_id,
                    "cohort": cohort(sample_id),
                    "train_sample_count": len(train_samples),
                    "rank": rank,
                    "target": target,
                    "training_variance": float(training_variance),
                    "fold_target_set": target_set_key,
                    "matches_full_cohort_reference": bool(selected_targets == full_cohort_reference),
                }
            )

    ordered_sets = [summary["selected_brain_targets"] for summary in fold_summaries]
    unique_ordered_sets: list[list[str]] = []
    for target_set in ordered_sets:
        if target_set not in unique_ordered_sets:
            unique_ordered_sets.append(target_set)

    target_union = sorted({target for target_set in ordered_sets for target in target_set})
    same_ordered_targets = all(target_set == ordered_sets[0] for target_set in ordered_sets)
    same_target_set = all(set(target_set) == set(ordered_sets[0]) for target_set in ordered_sets)
    display_targets = ordered_sets[0] if same_ordered_targets else target_union
    summary = {
        "top_n": top_n,
        "selection_scope": "outer LOOCV training samples only",
        "train_sample_count_per_fold": len(samples) - 1,
        "full_cohort_reference_top_targets": full_cohort_reference,
        "same_ordered_targets_all_folds": bool(same_ordered_targets),
        "same_target_set_all_folds": bool(same_target_set),
        "unique_ordered_target_sets": unique_ordered_sets,
        "target_union": target_union,
        "display_brain_targets": display_targets,
        "folds": fold_summaries,
    }
    return pd.DataFrame(fold_rows), summary


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    input_dim: int,
    output_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout_rate: float,
) -> LiverBrainCrossTalkMLP:
    set_determinism(seed)
    model = LiverBrainCrossTalkMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    return model


def predict(model: LiverBrainCrossTalkMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()


def normalize_shap_values(values: object, feature_count: int, target_count: int) -> np.ndarray:
    if isinstance(values, list):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 2 and target_count == 1:
        arr = arr[:, :, None]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")

    if arr.shape[1] == feature_count and arr.shape[2] == target_count:
        return arr
    if arr.shape[0] == target_count and arr.shape[2] == feature_count:
        return np.moveaxis(arr, 0, 2)
    if arr.shape[1] == target_count and arr.shape[2] == feature_count:
        return np.swapaxes(arr, 1, 2)

    raise ValueError(f"Could not map SHAP output shape {arr.shape} to samples/features/targets.")


def compute_shap_for_fold(
    model: LiverBrainCrossTalkMLP,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    target_names: list[str],
    seed: int,
) -> np.ndarray:
    set_determinism(seed)

    def model_predict(data_numpy: np.ndarray) -> np.ndarray:
        return predict(model, data_numpy)

    explainer = shap.KernelExplainer(model_predict, X_background)
    values = explainer.shap_values(X_explain, nsamples=2 ** len(feature_names), silent=True)
    return normalize_shap_values(values, len(feature_names), len(target_names))


def run_seed(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    feature_names: list[str],
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    fold_metrics: list[dict] = []
    prediction_rows: list[dict] = []
    shap_rows: list[dict] = []

    samples = list(X_raw.index)
    for fold_index, sample_id in enumerate(samples, start=1):
        fold_seed = seed * 1000 + fold_index
        train_samples = [sample for sample in samples if sample != sample_id]
        target_names = select_fold_brain_targets(y_raw, train_samples)

        X_train_raw = X_raw.loc[train_samples]
        y_train_raw = y_raw.loc[train_samples, target_names]
        X_val_raw = X_raw.loc[[sample_id]]
        y_val_raw = y_raw.loc[[sample_id], target_names]

        X_scaler = fit_scaler(X_train_raw)
        y_scaler = fit_scaler(y_train_raw)
        X_train = transform_with_scaler(X_train_raw, X_scaler).to_numpy(dtype=np.float32)
        y_train = transform_with_scaler(y_train_raw, y_scaler).to_numpy(dtype=np.float32)
        X_val = transform_with_scaler(X_val_raw, X_scaler).to_numpy(dtype=np.float32)
        y_val = transform_with_scaler(y_val_raw, y_scaler).to_numpy(dtype=np.float32)

        model = train_model(
            X_train,
            y_train,
            seed=fold_seed,
            input_dim=len(feature_names),
            output_dim=len(target_names),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
        )

        y_pred = predict(model, X_val)
        fold_mse = float(np.mean((y_val - y_pred) ** 2))
        fold_metrics.append(
            {
                "seed": seed,
                "fold": fold_index,
                "sample_id": sample_id,
                "cohort": cohort(sample_id),
                "fold_mse": fold_mse,
                "selected_brain_targets": ";".join(target_names),
            }
        )

        for target_index, target_name in enumerate(target_names):
            prediction_rows.append(
                {
                    "seed": seed,
                    "fold": fold_index,
                    "sample_id": sample_id,
                    "cohort": cohort(sample_id),
                    "target": target_name,
                    "observed_scaled": float(y_val[0, target_index]),
                    "predicted_scaled": float(y_pred[0, target_index]),
                    "residual": float(y_val[0, target_index] - y_pred[0, target_index]),
                }
            )

        if not args.skip_shap:
            shap_values = compute_shap_for_fold(
                model,
                X_background=X_train,
                X_explain=X_val,
                feature_names=feature_names,
                target_names=target_names,
                seed=fold_seed + 500_000,
            )
            for feature_index, feature_name in enumerate(feature_names):
                for target_index, target_name in enumerate(target_names):
                    shap_rows.append(
                        {
                            "seed": seed,
                            "fold": fold_index,
                            "sample_id": sample_id,
                            "cohort": cohort(sample_id),
                            "feature": feature_name,
                            "target": target_name,
                            "shap_value": float(shap_values[0, feature_index, target_index]),
                            "abs_shap_value": float(abs(shap_values[0, feature_index, target_index])),
                        }
                    )

    seed_summary = {
        "seed": seed,
        "vmse": float(np.mean([row["fold_mse"] for row in fold_metrics])),
    }
    return seed_summary, fold_metrics, prediction_rows, shap_rows


def group_expression_summary(liver_df: pd.DataFrame, brain_df: pd.DataFrame, feature_names: list[str], target_names: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for tissue, df, names in [("liver", liver_df, feature_names), ("brain", brain_df, target_names)]:
        for name in names:
            for group_name, values in df[name].groupby([cohort(sample) for sample in df.index]):
                rows.append(
                    {
                        "tissue": tissue,
                        "feature": name,
                        "cohort": group_name,
                        "n": int(values.shape[0]),
                        "mean_raw_count": float(values.mean()),
                        "std_raw_count": float(values.std(ddof=1)),
                        "min_raw_count": float(values.min()),
                        "max_raw_count": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log(f"Wrote {project_path(path)}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {project_path(path)}")


def make_architecture_figure(metrics: dict, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title("Reproducible Liver-Brain miRNA Modeling Pipeline", fontsize=16, weight="bold", pad=18)

    boxes = [
        (0.05, 0.28, 0.23, 0.44, "Input liver miRNAs", "\n".join(metrics["selected_liver_features"])),
        (0.39, 0.28, 0.22, 0.44, "Fold-wise MLP", f"6 -> {metrics['model']['hidden_dim']} -> 5\nBatchNorm1d\nDropout p={metrics['model']['dropout_rate']}\nAdam, {metrics['model']['epochs']} epochs"),
        (0.73, 0.28, 0.23, 0.44, "Output brain targets", "\n".join(metrics["selected_brain_targets"])),
    ]
    colors = ["#fbe8e3", "#e7f3f2", "#e9f1fb"]
    edges = ["#c85a4a", "#2f6f73", "#527aa3"]
    for (x, y, w, h, title, body), fill, edge in zip(boxes, colors, edges):
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=fill, edgecolor=edge, linewidth=2))
        ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="center", fontsize=13, weight="bold", transform=ax.transAxes)
        ax.text(x + w / 2, y + h / 2 - 0.04, body, ha="center", va="center", fontsize=10, transform=ax.transAxes)

    for start, end in [(0.28, 0.39), (0.61, 0.73)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 2})

    ax.text(
        0.5,
        0.12,
        "Scaler fitting occurs inside each LOOCV training fold; held-out samples do not influence normalization.",
        ha="center",
        fontsize=10,
        color="#52616f",
        transform=ax.transAxes,
    )
    save_figure(fig, figure_dir / "figure_1_reproducible_architecture.png")


def make_residual_figure(predictions_df: pd.DataFrame, metrics: dict, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        predictions_df["observed_scaled"],
        predictions_df["predicted_scaled"],
        c=predictions_df["seed"],
        cmap="viridis",
        alpha=0.72,
        s=34,
        edgecolors="white",
        linewidths=0.4,
    )
    low = float(min(predictions_df["observed_scaled"].min(), predictions_df["predicted_scaled"].min()))
    high = float(max(predictions_df["observed_scaled"].max(), predictions_df["predicted_scaled"].max()))
    pad = (high - low) * 0.08
    ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#17212b", linewidth=1.5)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Observed brain target expression (fold-wise z-score)")
    ax.set_ylabel("Predicted brain target expression (fold-wise z-score)")
    ax.set_title("Leakage-Free LOOCV Predictions Across Fixed Seeds")
    summary = metrics["repeated_seed_summary"]
    ax.text(
        0.04,
        0.96,
        f"seeds = {len(metrics['seeds'])}\nmean VMSE = {summary['vmse_mean']:.4f}\nstd = {summary['vmse_std']:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c7d1dc"},
    )
    ax.grid(True, alpha=0.25)
    save_figure(fig, figure_dir / "figure_2_loocv_residuals.png")


def make_shap_figure(shap_summary_df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = shap_summary_df.sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], xerr=plot_df["std_abs_shap"], color="#2f6f73", alpha=0.9)
    ax.set_xlabel("Mean absolute SHAP value across folds and seeds")
    ax.set_ylabel("Liver miRNA")
    ax.set_title("Aggregated SHAP Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figure_dir / "figure_3_shap_importance.png")


def make_group_expression_figure(expression_df: pd.DataFrame, target_names: list[str], figure_dir: Path) -> None:
    brain_df = expression_df[(expression_df["tissue"] == "brain") & (expression_df["feature"].isin(target_names))]
    pivot = brain_df.pivot(index="feature", columns="cohort", values="mean_raw_count").reindex(target_names)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot[["FL", "HC", "VC"]].plot(kind="bar", ax=ax, color=["#c85a4a", "#527aa3", "#4b8a54"])
    ax.set_ylabel("Mean raw count")
    ax.set_xlabel("Brain target miRNA")
    ax.set_title("Brain Target Expression by Cohort")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_4_brain_targets_by_cohort.png")


def write_dashboard(metrics: dict, output_dir: Path) -> None:
    html_path = output_dir / "dashboard.html"
    summary = metrics["repeated_seed_summary"]
    feature_items = "\n".join(f"<li>{feature}</li>" for feature in metrics["selected_liver_features"])
    target_items = "\n".join(f"<li>{target}</li>" for target in metrics["selected_brain_targets"])
    seed_rows = "\n".join(f"<tr><td>{row['seed']}</td><td>{row['vmse']:.4f}</td></tr>" for row in metrics["seed_results"])
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NASA miRNA Reproducibility Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #eef2f5; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #d8dee7; border-radius: 8px; padding: 18px; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee7; border-radius: 6px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-bottom: 1px solid #d8dee7; padding: 6px 0; }}
    td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>NASA miRNA Reproducibility Dashboard</h1>
      <p>Leakage-free fold-wise LOOCV across fixed repeated seeds.</p>
      <div class="grid">
        <div><strong>VMSE mean:</strong> {summary['vmse_mean']:.4f}</div>
        <div><strong>VMSE std:</strong> {summary['vmse_std']:.4f}</div>
        <div><strong>VMSE range:</strong> {summary['vmse_min']:.4f}-{summary['vmse_max']:.4f}</div>
        <div><strong>Seeds:</strong> {', '.join(str(seed) for seed in metrics['seeds'])}</div>
      </div>
    </section>
    <section class="grid">
      <div>
        <h2>Seed VMSE</h2>
        <table>{seed_rows}</table>
      </div>
      <div>
        <h2>Liver Inputs</h2>
        <ul>{feature_items}</ul>
      </div>
      <div>
        <h2>Brain Targets</h2>
        <ul>{target_items}</ul>
      </div>
    </section>
    <section class="grid">
      <img src="figures/figure_2_loocv_residuals.png" alt="LOOCV residuals">
      <img src="figures/figure_3_shap_importance.png" alt="SHAP feature importance">
      <img src="figures/figure_4_brain_targets_by_cohort.png" alt="Brain targets by cohort">
    </section>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    log(f"Wrote {project_path(html_path)}")


def write_metrics_json(metrics: dict, output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"Wrote {project_path(metrics_path)}")


def run_reproducibility(args: argparse.Namespace) -> dict:
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"

    liver_clean, brain_clean, clean_metadata = rebuild_clean_matrices(output_dir)
    X_raw, y_raw, feature_names = select_model_matrices(liver_clean, brain_clean)
    seeds = parse_seeds(args.seeds)
    target_selection_df, target_selection_summary = build_fold_target_selection(y_raw, X_raw.index)
    display_target_names = target_selection_summary["display_brain_targets"]

    log(f"Selected liver inputs: {', '.join(feature_names)}")
    if target_selection_summary["same_ordered_targets_all_folds"]:
        log(f"Fold-wise brain target selection is identical across LOOCV folds: {', '.join(display_target_names)}")
    else:
        log(f"Fold-wise brain target selection varies across LOOCV folds; target union: {', '.join(display_target_names)}")
    log(f"Running leakage-free LOOCV for seeds: {', '.join(map(str, seeds))}")

    seed_results: list[dict] = []
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    shap_rows: list[dict] = []

    for seed in seeds:
        seed_summary, seed_fold_rows, seed_prediction_rows, seed_shap_rows = run_seed(
            X_raw,
            y_raw,
            feature_names,
            seed,
            args,
        )
        seed_results.append(seed_summary)
        fold_rows.extend(seed_fold_rows)
        prediction_rows.extend(seed_prediction_rows)
        shap_rows.extend(seed_shap_rows)
        log(f"Seed {seed} VMSE: {seed_summary['vmse']:.4f}")

    seed_df = pd.DataFrame(seed_results)
    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    expression_df = group_expression_summary(liver_clean, brain_clean, feature_names, display_target_names)

    save_csv(seed_df, table_dir / "metrics_by_seed.csv")
    save_csv(fold_df, table_dir / "fold_metrics.csv")
    save_csv(predictions_df, table_dir / "predictions.csv")
    save_csv(target_selection_df, table_dir / "fold_target_selection.csv")
    save_csv(expression_df, table_dir / "expression_group_summary.csv")

    if shap_rows:
        shap_df = pd.DataFrame(shap_rows)
        shap_by_seed = (
            shap_df.groupby(["seed", "feature"], as_index=False)["abs_shap_value"]
            .mean()
            .rename(columns={"abs_shap_value": "mean_abs_shap"})
        )
        shap_summary = (
            shap_by_seed.groupby("feature")["mean_abs_shap"]
            .agg(
                mean_abs_shap="mean",
                std_abs_shap="std",
                min_abs_shap="min",
                max_abs_shap="max",
            )
            .reset_index()
        )
        shap_summary["std_abs_shap"] = shap_summary["std_abs_shap"].fillna(0.0)
        shap_summary = shap_summary.sort_values("mean_abs_shap", ascending=False)
        shap_summary["rank"] = np.arange(1, len(shap_summary) + 1)
        save_csv(shap_df, table_dir / "shap_values.csv")
        save_csv(shap_by_seed, table_dir / "shap_importance_by_seed.csv")
        save_csv(shap_summary, table_dir / "shap_importance_summary.csv")
    else:
        shap_summary = pd.DataFrame(columns=["feature", "mean_abs_shap", "std_abs_shap", "min_abs_shap", "max_abs_shap", "rank"])

    vmse_values = seed_df["vmse"].to_numpy(dtype=float)
    metrics = {
        "pipeline_version": 2,
        "python": sys.version.split()[0],
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "raw_files": {
            "mirna_path": str(RAW_MIRNA.relative_to(ROOT)),
            "mirna_sha256": sha256(RAW_MIRNA),
            "brain_series_path": str(BRAIN_SERIES.relative_to(ROOT)),
            "brain_series_sha256": sha256(BRAIN_SERIES),
        },
        "clean_data": clean_metadata,
        "samples": [{"sample_id": sample, "cohort": cohort(sample)} for sample in X_raw.index],
        "cohort_counts": {group: int(count) for group, count in pd.Series([cohort(sample) for sample in X_raw.index]).value_counts().sort_index().items()},
        "selected_liver_features": feature_names,
        "selected_brain_targets": display_target_names,
        "target_selection": "top 5 brain miRNAs by variance within each outer LOOCV training fold",
        "target_selection_audit": target_selection_summary,
        "normalization": "log1p plus z-score fitted inside each training fold only",
        "leakage_free": True,
        "seeds": seeds,
        "model": {
            "input_dim": len(feature_names),
            "hidden_dim": args.hidden_dim,
            "output_dim": TOP_BRAIN_TARGETS,
            "dropout_rate": args.dropout_rate,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "Adam",
            "loss": "MSE",
            "batch_norm": "BatchNorm1d on hidden activations",
        },
        "seed_results": seed_results,
        "repeated_seed_summary": {
            "vmse_mean": float(np.mean(vmse_values)),
            "vmse_std": float(np.std(vmse_values, ddof=0)),
            "vmse_min": float(np.min(vmse_values)),
            "vmse_max": float(np.max(vmse_values)),
        },
        "outputs": {
            "metrics_json": project_path(output_dir / "metrics.json"),
            "metrics_by_seed_csv": project_path(table_dir / "metrics_by_seed.csv"),
            "fold_metrics_csv": project_path(table_dir / "fold_metrics.csv"),
            "predictions_csv": project_path(table_dir / "predictions.csv"),
            "fold_target_selection_csv": project_path(table_dir / "fold_target_selection.csv"),
            "expression_group_summary_csv": project_path(table_dir / "expression_group_summary.csv"),
            "shap_values_csv": project_path(table_dir / "shap_values.csv") if shap_rows else None,
            "shap_importance_summary_csv": project_path(table_dir / "shap_importance_summary.csv") if shap_rows else None,
        },
    }

    write_metrics_json(metrics, output_dir)
    make_architecture_figure(metrics, figure_dir)
    make_residual_figure(predictions_df, metrics, figure_dir)
    if not shap_summary.empty:
        make_shap_figure(shap_summary, figure_dir)
    make_group_expression_figure(expression_df, display_target_names, figure_dir)
    write_dashboard(metrics, output_dir)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the NASA liver-brain miRNA analysis from raw data to saved outputs.")
    parser.add_argument("--output-dir", default="reproducibility", help="Directory for reproducible outputs.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="Comma-separated fixed seeds.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per LOOCV fold.")
    parser.add_argument("--hidden-dim", type=int, default=16, help="MLP hidden layer dimension.")
    parser.add_argument("--dropout-rate", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Adam weight decay.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP calculations for a faster smoke test.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = run_reproducibility(args)
    summary = metrics["repeated_seed_summary"]
    log(
        "Completed reproducibility run: "
        f"VMSE mean={summary['vmse_mean']:.4f}, "
        f"std={summary['vmse_std']:.4f}, "
        f"range={summary['vmse_min']:.4f}-{summary['vmse_max']:.4f}"
    )


if __name__ == "__main__":
    main()
