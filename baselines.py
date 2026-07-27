from __future__ import annotations

"""Baseline and permutation comparisons for the OSD-914 reproduction package."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from reproduce import (
    DEFAULT_SEEDS,
    ROOT,
    TOP_BRAIN_TARGETS,
    build_fold_target_selection,
    cohort,
    fit_scaler,
    log,
    parse_seeds,
    predict,
    project_path,
    rebuild_clean_matrices,
    save_csv,
    save_figure,
    select_model_matrices,
    select_fold_brain_targets,
    train_model,
    transform_with_scaler,
)


DEFAULT_RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one numeric value is required.")
    return values


def calculate_vmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def scaled_fold_arrays(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    train_samples: Sequence[str],
    validation_samples: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_scaler = fit_scaler(X_raw.loc[list(train_samples)])
    y_scaler = fit_scaler(y_raw.loc[list(train_samples)])
    X_train = transform_with_scaler(X_raw.loc[list(train_samples)], X_scaler).to_numpy(dtype=np.float64)
    y_train = transform_with_scaler(y_raw.loc[list(train_samples)], y_scaler).to_numpy(dtype=np.float64)
    X_val = transform_with_scaler(X_raw.loc[list(validation_samples)], X_scaler).to_numpy(dtype=np.float64)
    y_val = transform_with_scaler(y_raw.loc[list(validation_samples)], y_scaler).to_numpy(dtype=np.float64)
    return X_train, y_train, X_val, y_val


def select_nested_ridge_alpha(
    X_outer_train_raw: pd.DataFrame,
    y_outer_train_raw: pd.DataFrame,
    alphas: Sequence[float],
    outer_fold: int,
    outer_sample_id: str,
) -> tuple[float, list[dict]]:
    inner_samples = list(X_outer_train_raw.index)
    alpha_rows: list[dict] = []

    for alpha in alphas:
        inner_errors: list[float] = []
        for inner_fold, inner_sample_id in enumerate(inner_samples, start=1):
            inner_train = [sample for sample in inner_samples if sample != inner_sample_id]
            X_train, y_train, X_val, y_val = scaled_fold_arrays(
                X_outer_train_raw,
                y_outer_train_raw,
                inner_train,
                [inner_sample_id],
            )
            model = Ridge(alpha=float(alpha))
            model.fit(X_train, y_train)
            inner_errors.append(calculate_vmse(y_val, model.predict(X_val)))

        alpha_rows.append(
            {
                "outer_fold": outer_fold,
                "outer_sample_id": outer_sample_id,
                "alpha": float(alpha),
                "inner_vmse": float(np.mean(inner_errors)),
            }
        )

    best_row = min(alpha_rows, key=lambda row: (row["inner_vmse"], row["alpha"]))
    for row in alpha_rows:
        row["selected"] = bool(row["alpha"] == best_row["alpha"])
    return float(best_row["alpha"]), alpha_rows


def run_deterministic_baselines(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = list(X_raw.index)
    observed_rows: list[np.ndarray] = []
    prediction_arrays: dict[str, list[np.ndarray]] = {
        "training_fold_mean": [],
        "linear_regression": [],
        "ridge_regression": [],
    }
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    ridge_alpha_rows: list[dict] = []

    for fold_index, sample_id in enumerate(samples, start=1):
        train_samples = [sample for sample in samples if sample != sample_id]
        target_names = select_fold_brain_targets(y_raw, train_samples)
        y_fold_raw = y_raw.loc[:, target_names]
        X_train, y_train, X_val, y_val = scaled_fold_arrays(X_raw, y_fold_raw, train_samples, [sample_id])
        observed_rows.append(y_val)

        fold_predictions: dict[str, np.ndarray] = {}
        fold_predictions["training_fold_mean"] = np.mean(y_train, axis=0, keepdims=True)

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        fold_predictions["linear_regression"] = linear_model.predict(X_val)

        selected_alpha, alpha_rows = select_nested_ridge_alpha(
            X_raw.loc[train_samples],
            y_fold_raw.loc[train_samples],
            alphas,
            outer_fold=fold_index,
            outer_sample_id=sample_id,
        )
        ridge_alpha_rows.extend(alpha_rows)
        ridge_model = Ridge(alpha=selected_alpha)
        ridge_model.fit(X_train, y_train)
        fold_predictions["ridge_regression"] = ridge_model.predict(X_val)

        for model_id, y_pred in fold_predictions.items():
            prediction_arrays[model_id].append(y_pred)
            fold_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold_index,
                    "sample_id": sample_id,
                    "cohort": cohort(sample_id),
                    "fold_mse": calculate_vmse(y_val, y_pred),
                    "selected_alpha": selected_alpha if model_id == "ridge_regression" else np.nan,
                    "selected_brain_targets": ";".join(target_names),
                }
            )
            for target_index, target_name in enumerate(target_names):
                observed = float(y_val[0, target_index])
                predicted = float(y_pred[0, target_index])
                prediction_rows.append(
                    {
                        "model_id": model_id,
                        "fold": fold_index,
                        "sample_id": sample_id,
                        "cohort": cohort(sample_id),
                        "target": target_name,
                        "observed_scaled": observed,
                        "predicted_scaled": predicted,
                        "residual": observed - predicted,
                        "squared_error": (observed - predicted) ** 2,
                    }
                )

    y_observed = np.vstack(observed_rows)
    summary_rows = [
        {
            "model_id": "training_fold_mean",
            "model": "Training-fold mean",
            "loocv_vmse": calculate_vmse(y_observed, np.vstack(prediction_arrays["training_fold_mean"])),
            "interpretation": "Naive average-expression baseline",
        },
        {
            "model_id": "linear_regression",
            "model": "Linear regression",
            "loocv_vmse": calculate_vmse(y_observed, np.vstack(prediction_arrays["linear_regression"])),
            "interpretation": "Unregularized multivariate linear model",
        },
        {
            "model_id": "ridge_regression",
            "model": "Nested ridge regression",
            "loocv_vmse": calculate_vmse(y_observed, np.vstack(prediction_arrays["ridge_regression"])),
            "interpretation": "Regularized linear model with alpha selected inside each outer fold",
        },
    ]
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(ridge_alpha_rows),
    )


def run_mlp_seed_vmse(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    feature_names: Sequence[str],
    seed: int,
    args: argparse.Namespace,
) -> float:
    samples = list(X_raw.index)
    fold_mses: list[float] = []
    for fold_index, sample_id in enumerate(samples, start=1):
        fold_seed = seed * 1000 + fold_index
        train_samples = [sample for sample in samples if sample != sample_id]
        target_names = select_fold_brain_targets(y_raw, train_samples)
        y_fold_raw = y_raw.loc[:, target_names]
        X_train, y_train, X_val, y_val = scaled_fold_arrays(X_raw, y_fold_raw, train_samples, [sample_id])
        model = train_model(
            X_train.astype(np.float32),
            y_train.astype(np.float32),
            seed=fold_seed,
            input_dim=len(feature_names),
            output_dim=len(target_names),
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
        )
        fold_mses.append(calculate_vmse(y_val, predict(model, X_val.astype(np.float32))))
    return float(np.mean(fold_mses))


def run_permutation_null(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    feature_names: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.permutation_seed)
    samples = np.array(list(X_raw.index), dtype=object)
    permutation_rows: list[dict] = []
    seed_rows: list[dict] = []
    pairing_rows: list[dict] = []

    for permutation_index in range(1, args.permutations + 1):
        permuted_source_samples = rng.permutation(samples)
        y_permuted = y_raw.loc[list(permuted_source_samples)].copy()
        y_permuted.index = samples

        for input_sample, brain_sample in zip(samples, permuted_source_samples):
            pairing_rows.append(
                {
                    "permutation": permutation_index,
                    "input_sample_id": input_sample,
                    "brain_sample_id": brain_sample,
                    "kept_original_pair": bool(input_sample == brain_sample),
                }
            )

        vmse_values: list[float] = []
        for seed in seeds:
            vmse = run_mlp_seed_vmse(X_raw, y_permuted, feature_names, seed, args)
            vmse_values.append(vmse)
            seed_rows.append(
                {
                    "permutation": permutation_index,
                    "seed": seed,
                    "vmse": vmse,
                }
            )

        permutation_rows.append(
            {
                "permutation": permutation_index,
                "vmse": float(np.mean(vmse_values)),
                "vmse_std_across_seeds": float(np.std(vmse_values, ddof=0)),
                "n_seeds": len(seeds),
            }
        )
        if permutation_index == 1 or permutation_index % max(1, args.permutation_log_every) == 0:
            log(f"Permutation {permutation_index}/{args.permutations} VMSE={permutation_rows[-1]['vmse']:.4f}")

    return pd.DataFrame(permutation_rows), pd.DataFrame(seed_rows), pd.DataFrame(pairing_rows)


def run_observed_mlp_with_seeds(
    X_raw: pd.DataFrame,
    y_raw: pd.DataFrame,
    feature_names: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    for seed in seeds:
        vmse = run_mlp_seed_vmse(X_raw, y_raw, feature_names, seed, args)
        rows.append({"seed": seed, "vmse": vmse})

    values = np.array([row["vmse"] for row in rows], dtype=float)
    summary = {
        "vmse_mean": float(np.mean(values)),
        "vmse_std": float(np.std(values, ddof=0)),
        "vmse_min": float(np.min(values)),
        "vmse_max": float(np.max(values)),
        "n_seeds": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "source": "recomputed by baselines.py using the same MLP seed procedure as the permutation null",
    }
    return summary, pd.DataFrame(rows)


def load_observed_mlp_summary(metrics_path: Path, observed_mlp_vmse: Optional[float]) -> Optional[dict]:
    if observed_mlp_vmse is not None:
        return {
            "vmse_mean": float(observed_mlp_vmse),
            "vmse_std": np.nan,
            "vmse_min": np.nan,
            "vmse_max": np.nan,
            "n_seeds": np.nan,
            "source": "command-line argument",
        }
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = metrics.get("repeated_seed_summary", {})
    seeds = metrics.get("seeds", [])
    return {
        "vmse_mean": float(summary["vmse_mean"]),
        "vmse_std": float(summary.get("vmse_std", np.nan)),
        "vmse_min": float(summary.get("vmse_min", np.nan)),
        "vmse_max": float(summary.get("vmse_max", np.nan)),
        "n_seeds": len(seeds),
        "source": project_path(metrics_path),
    }


def load_observed_mlp_seed_results(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = metrics.get("seed_results", [])
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "seed": int(row["seed"]),
                "vmse": float(row["vmse"]),
            }
            for row in rows
        ]
    )


def summarize_permutations(permutation_df: pd.DataFrame, observed_mlp_vmse: Optional[float]) -> Optional[dict]:
    if permutation_df.empty:
        return None
    values = permutation_df["vmse"].to_numpy(dtype=float)
    summary = {
        "n_permutations": int(values.shape[0]),
        "vmse_mean": float(np.mean(values)),
        "vmse_std": float(np.std(values, ddof=0)),
        "vmse_median": float(np.median(values)),
        "vmse_min": float(np.min(values)),
        "vmse_max": float(np.max(values)),
        "vmse_percentile_2_5": float(np.percentile(values, 2.5)),
        "vmse_percentile_97_5": float(np.percentile(values, 97.5)),
    }
    if observed_mlp_vmse is not None:
        count_as_good_or_better = int(np.sum(values <= observed_mlp_vmse))
        summary.update(
            {
                "observed_mlp_vmse": float(observed_mlp_vmse),
                "count_permuted_less_or_equal_observed": count_as_good_or_better,
                "permutation_p_value": float((1 + count_as_good_or_better) / (1 + values.shape[0])),
            }
        )
    return summary


def build_comparison_table(
    deterministic_summary: pd.DataFrame,
    observed_mlp: Optional[dict],
    permutation_summary: Optional[dict],
) -> pd.DataFrame:
    rows = deterministic_summary.copy()
    if observed_mlp is not None:
        seed_label = f"{observed_mlp['n_seeds']}-seed mean" if not pd.isna(observed_mlp["n_seeds"]) else "observed"
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "model_id": "mlp",
                            "model": f"MLP ({seed_label})",
                            "loocv_vmse": observed_mlp["vmse_mean"],
                            "vmse_sd": observed_mlp["vmse_std"],
                            "interpretation": "Proposed nonlinear model",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if permutation_summary is not None:
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "model_id": "permuted_mlp_median",
                            "model": "Permuted MLP median",
                            "loocv_vmse": permutation_summary["vmse_median"],
                            "vmse_sd": permutation_summary["vmse_std"],
                            "interpretation": "Subject-pairing null distribution",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    if "vmse_sd" not in rows.columns:
        rows["vmse_sd"] = np.nan
    mean_vmse = float(rows.loc[rows["model_id"] == "training_fold_mean", "loocv_vmse"].iloc[0])
    rows["relative_improvement_vs_mean_percent"] = (mean_vmse - rows["loocv_vmse"]) / mean_vmse * 100.0
    preferred_order = {
        "training_fold_mean": 1,
        "linear_regression": 2,
        "ridge_regression": 3,
        "mlp": 4,
        "permuted_mlp_median": 5,
    }
    rows["sort_order"] = rows["model_id"].map(preferred_order).fillna(99)
    rows = rows.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)
    return rows


def make_baseline_comparison_figure(comparison_df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = comparison_df[comparison_df["model_id"] != "permuted_mlp_median"].copy()
    colors = {
        "training_fold_mean": "#6b7280",
        "linear_regression": "#2f6f73",
        "ridge_regression": "#527aa3",
        "mlp": "#c85a4a",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        plot_df["model"],
        plot_df["loocv_vmse"],
        color=[colors.get(model_id, "#8f7a4f") for model_id in plot_df["model_id"]],
    )
    ax.set_ylabel("LOOCV VMSE")
    ax.set_title("Baseline Model Comparison")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for index, row in plot_df.iterrows():
        ax.text(index, row["loocv_vmse"] + 0.04, f"{row['loocv_vmse']:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_5_baseline_comparison.png")


def make_permutation_figure(permutation_df: pd.DataFrame, observed_mlp_vmse: Optional[float], figure_dir: Path) -> None:
    if permutation_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(permutation_df["vmse"], bins=min(30, max(5, len(permutation_df) // 10)), color="#527aa3", alpha=0.82)
    if observed_mlp_vmse is not None:
        ax.axvline(observed_mlp_vmse, color="#c85a4a", linewidth=2, label=f"Observed MLP = {observed_mlp_vmse:.3f}")
        ax.legend(frameon=False)
    ax.set_xlabel("Permuted MLP VMSE")
    ax.set_ylabel("Permutation count")
    ax.set_title("Subject-Pairing Permutation Null")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_6_permutation_null.png")


def write_baseline_metrics(
    path: Path,
    comparison_df: pd.DataFrame,
    observed_mlp: Optional[dict],
    permutation_summary: Optional[dict],
    feature_names: Sequence[str],
    target_selection_summary: dict,
    alphas: Sequence[float],
    args: argparse.Namespace,
) -> None:
    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    payload = {
        "selected_liver_features": list(feature_names),
        "selected_brain_targets": target_selection_summary["display_brain_targets"],
        "target_selection": "top 5 brain miRNAs by variance within each outer LOOCV training fold",
        "target_selection_audit": target_selection_summary,
        "normalization": "log1p plus z-score fitted inside each training fold only",
        "deterministic_baselines": comparison_df[
            comparison_df["model_id"].isin(["training_fold_mean", "linear_regression", "ridge_regression"])
        ].to_dict(orient="records"),
        "observed_mlp": observed_mlp,
        "ridge_alphas": [float(alpha) for alpha in alphas],
        "permutation": {
            "requested_permutations": args.permutations,
            "permutation_seed": args.permutation_seed,
            "mlp_seeds": parse_seeds(args.mlp_seeds),
            "summary": permutation_summary,
        },
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"Wrote {project_path(path)}")


def run_baseline_analysis(args: argparse.Namespace) -> dict:
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    liver_clean, brain_clean, _clean_metadata = rebuild_clean_matrices(output_dir)
    X_raw, y_raw, feature_names = select_model_matrices(liver_clean, brain_clean)
    target_selection_df, target_selection_summary = build_fold_target_selection(y_raw, X_raw.index, top_n=TOP_BRAIN_TARGETS)
    alphas = parse_float_list(args.ridge_alphas)
    observed_mlp = load_observed_mlp_summary(Path(args.observed_metrics), args.observed_mlp_vmse)

    if target_selection_summary["same_ordered_targets_all_folds"]:
        log(
            "Fold-wise brain target selection is identical across LOOCV folds: "
            f"{', '.join(target_selection_summary['display_brain_targets'])}"
        )
    else:
        log(
            "Fold-wise brain target selection varies across LOOCV folds; target union: "
            f"{', '.join(target_selection_summary['display_brain_targets'])}"
        )
    log("Running mean, ordinary linear, and nested ridge baselines.")
    deterministic_summary, fold_df, prediction_df, ridge_alpha_df = run_deterministic_baselines(
        X_raw,
        y_raw,
        alphas,
    )

    permutation_df = pd.DataFrame()
    permutation_seed_df = pd.DataFrame()
    permutation_pairing_df = pd.DataFrame()
    observed_seed_df = load_observed_mlp_seed_results(Path(args.observed_metrics))
    permutation_summary = None
    if args.permutations > 0:
        permutation_seeds = parse_seeds(args.mlp_seeds)
        log("Recomputing observed MLP with the same seed procedure used for the permutation null.")
        observed_mlp, observed_seed_df = run_observed_mlp_with_seeds(
            X_raw,
            y_raw,
            feature_names,
            permutation_seeds,
            args,
        )
        log(
            "Running subject-level MLP permutation null: "
            f"{args.permutations} permutations x {len(permutation_seeds)} seed(s)."
        )
        permutation_df, permutation_seed_df, permutation_pairing_df = run_permutation_null(
            X_raw,
            y_raw,
            feature_names,
            permutation_seeds,
            args,
        )
        observed_value = observed_mlp["vmse_mean"] if observed_mlp is not None else None
        permutation_summary = summarize_permutations(permutation_df, observed_value)

    comparison_df = build_comparison_table(deterministic_summary, observed_mlp, permutation_summary)

    save_csv(deterministic_summary, table_dir / "baseline_model_summary.csv")
    save_csv(fold_df, table_dir / "baseline_fold_metrics.csv")
    save_csv(prediction_df, table_dir / "baseline_predictions.csv")
    save_csv(ridge_alpha_df, table_dir / "ridge_alpha_selection.csv")
    save_csv(target_selection_df, table_dir / "baseline_fold_target_selection.csv")
    save_csv(comparison_df, table_dir / "model_comparison.csv")
    if not observed_seed_df.empty:
        save_csv(observed_seed_df, table_dir / "observed_mlp_seed_vmse.csv")
    if not permutation_df.empty:
        save_csv(permutation_df, table_dir / "permutation_mlp_vmse.csv")
        save_csv(permutation_seed_df, table_dir / "permutation_mlp_seed_vmse.csv")
        save_csv(permutation_pairing_df, table_dir / "permutation_pairings.csv")

    make_baseline_comparison_figure(comparison_df, figure_dir)
    make_permutation_figure(permutation_df, observed_mlp["vmse_mean"] if observed_mlp is not None else None, figure_dir)
    write_baseline_metrics(
        output_dir / "baseline_metrics.json",
        comparison_df,
        observed_mlp,
        permutation_summary,
        feature_names,
        target_selection_summary,
        alphas,
        args,
    )

    log("Baseline comparison:")
    print(
        comparison_df[["model", "loocv_vmse", "relative_improvement_vs_mean_percent", "interpretation"]].to_string(
            index=False,
            formatters={
                "loocv_vmse": "{:.4f}".format,
                "relative_improvement_vs_mean_percent": "{:.1f}".format,
            },
        )
    )
    return {
        "comparison": comparison_df.to_dict(orient="records"),
        "permutation_summary": permutation_summary,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-free baseline and optional permutation comparisons for the NASA liver-brain MLP."
    )
    parser.add_argument("--output-dir", default="reproducibility", help="Directory for baseline outputs.")
    parser.add_argument("--observed-metrics", default=str(ROOT / "reproducibility" / "metrics.json"))
    parser.add_argument("--observed-mlp-vmse", type=float, default=None, help="Override observed MLP VMSE.")
    parser.add_argument("--ridge-alphas", default=",".join(str(alpha) for alpha in DEFAULT_RIDGE_ALPHAS))
    parser.add_argument("--permutations", type=int, default=0, help="Number of subject-pairing permutations to run.")
    parser.add_argument("--permutation-seed", type=int, default=914)
    parser.add_argument("--permutation-log-every", type=int, default=25)
    parser.add_argument("--mlp-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="MLP seeds used per permutation.")
    parser.add_argument("--epochs", type=int, default=100, help="MLP epochs per LOOCV fold for permutation runs.")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.permutations < 0:
        raise ValueError("--permutations must be nonnegative.")
    if args.permutation_log_every <= 0:
        raise ValueError("--permutation-log-every must be positive.")
    run_baseline_analysis(args)


if __name__ == "__main__":
    main()
