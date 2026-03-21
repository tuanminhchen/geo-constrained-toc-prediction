#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Egypt_Western_Desert - Minimal, self-contained training runner
Outputs (per setting, per seed):
  outputs/artifacts/predictions/predsTEST_{setting}_seed{seed}.csv
  outputs/artifacts/metrics/metrics_{setting}_seed{seed}.json
  outputs/artifacts/metrics/metrics_summary_ablation.csv
  outputs/artifacts/metrics/delta_r2_by_seed.csv (long: seed, setting, r2_base, r2_full, delta_r2)

Design goals:
- Robust auto-discovery of input CSV under ./data
- Robust column inference for TOC / S2 and common well logs
- No target leakage: HI/OI/PI and TOC-derived variables excluded from features
- Deterministic per-seed split, train on TRAIN only, constraint fit on TRAIN only
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# -----------------------------
# Utilities
# -----------------------------
def clip_nonneg(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, None)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def summary_stats(a: List[float]) -> Dict[str, float]:
    arr = np.asarray(a, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "q1": float(np.quantile(arr, 0.25)) if arr.size else float("nan"),
        "q3": float(np.quantile(arr, 0.75)) if arr.size else float("nan"),
        "min": float(np.min(arr)) if arr.size else float("nan"),
        "max": float(np.max(arr)) if arr.size else float("nan"),
    }


def _normalize_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())


# Prevent target leakage: remove TOC-derived variables (HI/OI/PI)
def is_leaky_feature(col: str) -> bool:
    """
    Return True if the column should NEVER be used as a feature (target leakage).
    HI = 100*S2/TOC, OI = 100*S3/TOC, PI = S1/(S1+S2) - these contain TOC.
    """
    n = _normalize_colname(col)
    leaky_substrings = ["toc", "hi", "oi", "pi", "gtoc", "per_toc"]
    return any(sub in n for sub in leaky_substrings)


def find_input_csv(data_dir: Path) -> Path:
    """
    Heuristic: pick the "most dataset-like" CSV under ./data (recursive).
    Prefer names containing: merged/clean/processed/dataset/train/all
    """
    cands = list(data_dir.rglob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"No CSV found under: {data_dir}")

    def score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        key = 0
        for w in ["merged", "clean", "processed", "dataset", "train", "all", "final"]:
            if w in name:
                key += 10
        # prefer larger files
        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0
        return key, sz

    cands.sort(key=score, reverse=True)
    return cands[0]


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        cand_n = _normalize_colname(cand)
        if cand_n in norm_map:
            return norm_map[cand_n]
    return None


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer TOC / S2 / HI columns using common naming patterns.
    """
    toc = pick_col(df, ["TOC", "toc_wt", "toc_wt_pct", "toc_wtpercent", "tocwt", "toc_%", "toc_percent"])
    s2  = pick_col(df, ["S2", "s2_mghc_grock", "s2_mgHC_gRock", "s2_mg_hc_g_rock"])
    hi  = pick_col(df, ["HI", "hi_mghc_gtoc", "hi_mgHC_gTOC", "hydrogen_index"])

    return {"TOC": toc, "S2": s2, "HI": hi}


def infer_feature_sets(df: pd.DataFrame, toc_col: str) -> Dict[str, List[str]]:
    """
    Build 3 settings with SAFE features only (no target leakage).
    First priority: well logs (GR, RT, RD, etc.)
    Second priority: Rock-Eval S1, S2, S3, Tmax (never HI/OI/PI).
    Removed HI/OI/PI to prevent target leakage.
    """
    # First priority: well log style features
    log_cands = [
        "GR", "CGR", "SGR",
        "RES", "RT", "RD", "RS", "RILD", "RILM",
        "RHOB", "DEN",
        "NPHI", "CNL",
        "DT", "DTC", "AC",
        "PEF",
        "SP",
        "CAL",
    ]
    # Second priority: Rock-Eval variables that do NOT contain TOC
    rock_eval_cands = [
        "S1", "S1_mgHC_gRock", "S1_mghc_grock",
        "S2", "S2_mgHC_gRock", "S2_mghc_grock",
        "S3", "S3_mgCO2_gRock", "S3_mgco2_grock",
        "Tmax", "Tmax_degC", "TMAX",
    ]

    def existing_safe(cols: List[str]) -> List[str]:
        out = []
        for c in cols:
            cc = pick_col(df, [c])
            if cc is not None and cc != toc_col and cc not in out and not is_leaky_feature(cc):
                out.append(cc)
        return out

    logs = existing_safe(log_cands)
    rock = existing_safe(rock_eval_cands)
    # Fusion: logs first, then rock-eval (safe priority order)
    fusion = logs + [c for c in rock if c not in logs]

    # if fusion is empty, fall back to "all numeric except target and leaky"
    if not fusion:
        numeric_cols = [
            c for c in df.columns
            if c != toc_col and pd.api.types.is_numeric_dtype(df[c]) and not is_leaky_feature(c)
        ]
        fusion = numeric_cols[:]

    settings = {
        "RockEval": rock,
        "Logs": logs,
        "Fusion": fusion,
    }

    # drop empty settings except Fusion (keep Fusion always)
    settings = {k: v for k, v in settings.items() if (k == "Fusion") or (len(v) > 0)}
    return settings


# -----------------------------
# Constraint (train-only fit) - No HI-based constraints (target leakage)
# -----------------------------
@dataclass
class GeochemConstraint:
    # Linear geological consistency: TOC ≈ a*S2 + b (fitted on train only)
    a: float
    b: float
    # Range constraint: 0 <= TOC <= toc_max (from train only)
    toc_max: float
    # softness for S2 linear projection
    lambda_s2: float
    eps: float


def fit_geochem_constraint(
    y_train_true: np.ndarray,
    s2_train: np.ndarray,
    lambda_s2: float = 0.15,
    eps: float = 1e-6,
) -> GeochemConstraint:
    """
    Fit safe geological constraints using TRAIN data ONLY:
    - TOC ≈ a*S2 + b (least squares, nonnegative slope)
    - TOC_max = max(train_TOC) * 1.1
    Removed HI-based constraints to prevent target leakage.
    """
    y = np.asarray(y_train_true, float)
    s2 = np.asarray(s2_train, float)

    m = np.isfinite(y) & np.isfinite(s2)
    y = y[m]
    s2 = s2[m]

    if y.size < 10:
        a, b = 0.0, float(np.nanmedian(y) if y.size else 0.0)
    else:
        X = np.vstack([s2, np.ones_like(s2)]).T
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a, b = float(coef[0]), float(coef[1])
        except Exception:
            a, b = 0.0, float(np.nanmedian(y))
        if not np.isfinite(a):
            a = 0.0
        if a < 0:
            a = 0.0

    toc_max = float(np.max(y) * 1.1) if y.size else 30.0
    toc_max = max(toc_max, 1.0)

    return GeochemConstraint(a=a, b=b, toc_max=toc_max, lambda_s2=lambda_s2, eps=eps)


def project_to_s2_band(y_pred: np.ndarray, s2: np.ndarray, c: GeochemConstraint) -> np.ndarray:
    """
    Soft projection: y_new = (1-λ)*y_pred + λ*(a*S2+b)
    Linear geological consistency constraint.
    """
    y = np.asarray(y_pred, float)
    s2 = np.asarray(s2, float)
    target = c.a * s2 + c.b
    return (1.0 - c.lambda_s2) * y + c.lambda_s2 * target


def apply_range_constraint(y_pred: np.ndarray, c: GeochemConstraint) -> np.ndarray:
    """Clip TOC to [0, toc_max] (estimated from train only)."""
    return np.clip(np.asarray(y_pred, float), 0.0, c.toc_max)


def apply_constraint_ablation(
    y_pred_base: np.ndarray,
    s2_test: Optional[np.ndarray],
    constraint: Optional[GeochemConstraint],
    mode: str,
) -> np.ndarray:
    """
    mode:
      - "base": raw model output (clipped nonneg, range [0,toc_max])
      - "nn": nonnegative clip (same as base)
      - "s2": apply S2-TOC linear blend only
      - "full": apply range clip + S2-TOC linear blend (no HI)
    """
    y = clip_nonneg(np.asarray(y_pred_base, float))

    if mode in {"base", "nn"}:
        if constraint is not None:
            return apply_range_constraint(y, constraint)
        return y

    if s2_test is None or constraint is None:
        if constraint is not None:
            return apply_range_constraint(y, constraint)
        return y

    if mode == "s2":
        y1 = project_to_s2_band(y, s2_test, constraint)
        return clip_nonneg(apply_range_constraint(y1, constraint))

    if mode == "full":
        y1 = project_to_s2_band(y, s2_test, constraint)
        return clip_nonneg(apply_range_constraint(y1, constraint))

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------
# Refinement MLP (Stage 2: constraint loss in training)
# -----------------------------
def fit_toc_s2_linear(s2_train: np.ndarray, toc_train: np.ndarray) -> Tuple[float, float]:
    """
    Fit TOC ≈ a*S2 + b on train data (least squares).
    Used for L_linear: penalize deviation from a*S2 + b.
    Returns (a, b). If degenerate, returns (0.1, 0.0).
    """
    s2 = np.asarray(s2_train, float)
    toc = np.asarray(toc_train, float)
    m = np.isfinite(s2) & np.isfinite(toc)
    if m.sum() < 10:
        return 0.1, 0.0
    X = np.column_stack([s2[m], np.ones(m.sum())])
    y = toc[m]
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if not np.isfinite(a) or a < 0:
            return 0.1, 0.0
        return a, b
    except Exception:
        return 0.1, 0.0


def _loss_range_torch(y_pred: "torch.Tensor", y_min: float, y_max: float) -> "torch.Tensor":
    """L_range = E[max(0, y - y_max) + max(0, y_min - y)]"""
    over = torch.relu(y_pred - y_max)
    under = torch.relu(y_min - y_pred)
    return (over.mean() + under.mean())


def _loss_lin_torch(
    y_pred: "torch.Tensor",
    s2: "torch.Tensor",
    a: float,
    b: float,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """L_linear = E[(y_pred - (a*S2 + b))^2], TOC ≈ a*S2 + b from train fit"""
    toc_from_s2 = a * s2 + b
    return ((y_pred - toc_from_s2) ** 2).mean()


class RefinementMLP(nn.Module):
    """Small MLP: input (x, y_base) -> output y_refined."""

    def __init__(self, n_features: int, hidden: Tuple[int, ...] = (32, 16)):
        super().__init__()
        dims = [n_features + 1] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor", y_base: "torch.Tensor") -> "torch.Tensor":
        inp = torch.cat([x, y_base.unsqueeze(1)], dim=1)
        return self.net(inp).squeeze(-1)


def train_refinement_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_base_train: np.ndarray,
    s2_train: np.ndarray,
    toc_max: float,
    a_lin: float,
    b_lin: float,
    n_epochs: int = 150,
    lr: float = 1e-2,
    lambda_range: float = 0.2,
    lambda_lin: float = 0.2,
    seed: int = 0,
) -> RefinementMLP:
    """Train refinement MLP with L_mse + L_range + L_lin."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_base = torch.tensor(y_base_train, dtype=torch.float32, device=device)
    s2 = torch.tensor(s2_train, dtype=torch.float32, device=device)

    model = RefinementMLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        y_pred = model(X, y_base)
        L_mse = ((y_pred - y_true) ** 2).mean()
        L_range = _loss_range_torch(y_pred, 0.0, toc_max)
        L_lin = _loss_lin_torch(y_pred, s2, a_lin, b_lin)
        loss = L_mse + lambda_range * L_range + lambda_lin * L_lin
        loss.backward()
        opt.step()

    return model


def predict_refinement_mlp(
    model: RefinementMLP,
    X: np.ndarray,
    y_base: np.ndarray,
) -> np.ndarray:
    """Predict y_refined from (X, y_base)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        yb_t = torch.tensor(y_base, dtype=torch.float32, device=device)
        out = model(x_t, yb_t)
    return out.cpu().numpy().astype(float)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", type=str, default=None, help="Default: script parent dir")
    ap.add_argument("--data_csv", type=str, default=None, help="Optional explicit data CSV path. If omitted, auto-search under ./data")
    ap.add_argument("--n_repeats", type=int, default=500)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state_offset", type=int, default=0, help="Seed offset (useful to resume different blocks)")
    ap.add_argument("--max_rows", type=int, default=0, help="Debug: if >0, subsample first N rows after cleaning")
    ap.add_argument("--use_refinement", action="store_true", default=True, help="Use refinement MLP with constraint loss (Stage 2)")
    ap.add_argument("--no_refinement", action="store_false", dest="use_refinement", help="Disable refinement MLP, use post-hoc constraints only")
    ap.add_argument("--refinement_setting", type=str, default="fusion", choices=["all", "fusion"], help="Apply refinement to: 'all' settings or 'fusion' only (faster)")
    ap.add_argument("--clean_outputs", action="store_true", help="Delete existing predictions and metrics JSON before run (avoids stale artifacts)")
    args = ap.parse_args()

    if args.project_root:
        project_root = Path(args.project_root).expanduser().resolve()
    else:
        this_dir = Path(__file__).resolve().parent
        # If ./data is not under src/ but under the project root, auto-hop one level up
        if not (this_dir / "data").exists() and (this_dir.parent / "data").exists():
            project_root = this_dir.parent
        else:
            project_root = this_dir

    data_dir = project_root / "data"
    out_root = project_root / "outputs" / "artifacts"
    metrics_dir = out_root / "metrics"
    preds_dir = out_root / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Safety cleanup: remove stale artifacts before run
    if args.clean_outputs:
        for fp in preds_dir.glob("*.csv"):
            fp.unlink()
            print(f"  removed {fp.name}")
        for fp in metrics_dir.glob("metrics_*_seed*.json"):
            fp.unlink()
            print(f"  removed {fp.name}")
        print("Cleaned outputs/artifacts (predictions + metrics JSON)")

    # load data
    if args.data_csv:
        data_csv = Path(args.data_csv).expanduser().resolve()
    else:
        data_csv = find_input_csv(data_dir)

    print("=== Egypt Western Desert TRAIN (self-contained) ===")
    print("project_root:", project_root)
    print("use_refinement:", args.use_refinement, "(Stage 2 MLP with L_range + L_lin)" if args.use_refinement else "")
    if args.use_refinement and not TORCH_AVAILABLE:
        print("  [WARN] PyTorch not found, falling back to post-hoc constraints only.")
    print("data_csv     :", data_csv)
    print("metrics_dir  :", metrics_dir)
    print("preds_dir    :", preds_dir)

    df = pd.read_csv(data_csv, encoding="utf-8-sig")
    # basic cleanup: drop fully-empty cols, strip col names
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")

    cols = infer_columns(df)
    toc_col, s2_col = cols["TOC"], cols["S2"]
    if toc_col is None:
        raise ValueError(f"Cannot infer TOC column. Available columns={list(df.columns)}")
    if s2_col is None:
        raise ValueError(f"Cannot infer S2 column (required for constraint). inferred={cols}")

    # keep only rows with valid target
    df = df[np.isfinite(pd.to_numeric(df[toc_col], errors="coerce"))].copy()
    df[toc_col] = pd.to_numeric(df[toc_col], errors="coerce").astype(float)

    # Ensure S2 numeric (constraint needs it; HI not used - removed to prevent leakage)
    df[s2_col] = pd.to_numeric(df[s2_col], errors="coerce").astype(float)

    # Drop rows lacking S2 (constraint needs S2)
    df = df[np.isfinite(df[s2_col])].copy()

    # optional cap
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    print(f"rows after cleaning: {len(df)}")
    print(f"TOC={toc_col}  S2={s2_col}")

    # feature sets: only RockEval (exclude Fusion/Logs - Fusion refinement overfits, invalid r2~0.98)
    settings_all = infer_feature_sets(df, toc_col=toc_col)
    settings = {"RockEval": settings_all["RockEval"]} if "RockEval" in settings_all else settings_all
    if "RockEval" not in settings:
        raise ValueError("RockEval setting not found. Need S1/S2/Tmax features.")
    print("settings (RockEval only):")
    for k, v in settings.items():
        print(f"  - {k}: {len(v)} features")

    # model hyperparams (stable, fast)
    model_cfg = dict(
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        l2_regularization=0.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
    )

    # constraint params (no HI - removed to prevent target leakage)
    constraint_cfg = dict(
        lambda_s2=0.15,
        eps=1e-6,
    )

    # modes: base vs full (你论文主线就这俩；nn/s2保留扩展但不强制用)
    constraint_modes = ["base", "full"]

    # storage
    metrics_store = {(s, m): {"r2": [], "rmse": [], "mae": []} for s in settings for m in constraint_modes}

    # per-seed long-format rows: one per (seed, setting)
    per_seed_rows = []  # seed, setting, r2_base, r2_full, delta_r2

    # ----------------------------
    # run seeds
    # Order: load -> select features -> split -> fit constraints on TRAIN -> train -> evaluate
    # ----------------------------
    y_all = df[toc_col].astype(float).values
    s2_all = df[s2_col].astype(float).values

    # prebuild X per setting for speed (features already filtered for leakage)
    X_by_setting = {}
    for setting_name, feat_cols in settings.items():
        X_by_setting[setting_name] = df[feat_cols].astype(float).values

    for i in range(args.n_repeats):
        seed = int(args.random_state_offset + i)

        idx_all = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            idx_all,
            test_size=args.test_size,
            random_state=seed,
        )

        y_train = y_all[idx_train]
        y_test = y_all[idx_test]
        s2_train = s2_all[idx_train]
        s2_test = s2_all[idx_test]

        # fit constraint on TRAIN ONLY (no full-dataset statistics)
        constraint = fit_geochem_constraint(
            y_train_true=y_train,
            s2_train=s2_train,
            **constraint_cfg,
        )

        for setting_name in settings.keys():
            X = X_by_setting[setting_name]
            X_train = X[idx_train]
            X_test = X[idx_test]

            model = HistGradientBoostingRegressor(random_state=seed, **model_cfg)
            model.fit(X_train, np.log1p(clip_nonneg(y_train)))

            y_pred_log = model.predict(X_test)
            y_pred_base_raw = np.expm1(y_pred_log)

            y_base = apply_constraint_ablation(y_pred_base_raw, s2_test=s2_test, constraint=constraint, mode="base")
            y_full = apply_constraint_ablation(y_pred_base_raw, s2_test=s2_test, constraint=constraint, mode="full")

            # Stage 2: Refinement MLP (optional, per-setting; guard by Fusion-only for speed)
            apply_refinement = (
                args.use_refinement
                and TORCH_AVAILABLE
                and (args.refinement_setting == "all" or setting_name == "Fusion")
            )
            if apply_refinement:
                y_base_train = np.expm1(model.predict(X_train))
                a_lin, b_lin = fit_toc_s2_linear(s2_train, y_train)
                toc_max = constraint.toc_max
                mlp = train_refinement_mlp(
                    X_train, y_train, y_base_train, s2_train,
                    toc_max=toc_max, a_lin=a_lin, b_lin=b_lin, seed=seed,
                )
                y_refined = predict_refinement_mlp(mlp, X_test, y_base.copy())
                y_full = np.clip(y_refined, 0.0, None)

            r2_base, rmse_base, mae_base = compute_metrics(y_test, y_base)
            r2_full, rmse_full, mae_full = compute_metrics(y_test, y_full)

            metrics_store[(setting_name, "base")]["r2"].append(r2_base)
            metrics_store[(setting_name, "base")]["rmse"].append(rmse_base)
            metrics_store[(setting_name, "base")]["mae"].append(mae_base)

            metrics_store[(setting_name, "full")]["r2"].append(r2_full)
            metrics_store[(setting_name, "full")]["rmse"].append(rmse_full)
            metrics_store[(setting_name, "full")]["mae"].append(mae_full)

            # Persist predictions for this setting immediately
            preds_df = pd.DataFrame({
                "index": idx_test.astype(int),
                "y_true": y_test.astype(float),
                "y_pred_base": y_base.astype(float),
                "y_pred_full": y_full.astype(float),
            })
            preds_path = preds_dir / f"predsTEST_{setting_name}_seed{seed}.csv"
            preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")

            # Persist metrics JSON for this setting
            refinement_used = apply_refinement
            metric_obj = {
                "seed": int(seed),
                "setting": setting_name,
                "n_train": int(len(idx_train)),
                "n_test": int(len(idx_test)),
                "target": toc_col,
                "s2_col": s2_col,
                "model": "HistGradientBoostingRegressor(log1p target)",
                "model_cfg": model_cfg,
                "constraint_cfg": constraint_cfg,
                "constraint_fitted_on": "train_only",
                "refinement_mlp": refinement_used,
                "metrics": {
                    "base": {"r2_test": r2_base, "rmse_test": rmse_base, "mae_test": mae_base},
                    "full": {"r2_test": r2_full, "rmse_test": rmse_full, "mae_test": mae_full},
                },
            }
            metrics_path = metrics_dir / f"metrics_{setting_name}_seed{seed}.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metric_obj, f, indent=2, ensure_ascii=False)

            # Append to per-seed long-format table (one row per seed, setting)
            per_seed_rows.append({
                "seed": int(seed),
                "setting": setting_name,
                "r2_base": float(r2_base),
                "r2_full": float(r2_full),
                "delta_r2": float(r2_full - r2_base),
            })

        if seed % 20 == 0:
            print(f"[seed={seed}] done ({len(settings)} settings)...")

    # ----------------------------
    # summaries
    # ----------------------------
    rows = []
    for (setting_name, mode), store in metrics_store.items():
        s_r2 = summary_stats(store["r2"])
        s_rmse = summary_stats(store["rmse"])
        s_mae = summary_stats(store["mae"])
        rows.append({
            "setting": setting_name,
            "mode": mode,
            "r2_mean": s_r2["mean"], "r2_std": s_r2["std"], "r2_median": s_r2["median"], "r2_q1": s_r2["q1"], "r2_q3": s_r2["q3"],
            "rmse_mean": s_rmse["mean"], "rmse_std": s_rmse["std"], "rmse_median": s_rmse["median"], "rmse_q1": s_rmse["q1"], "rmse_q3": s_rmse["q3"],
            "mae_mean": s_mae["mean"], "mae_std": s_mae["std"], "mae_median": s_mae["median"], "mae_q1": s_mae["q1"], "mae_q3": s_mae["q3"],
            "n_repeats": int(s_r2["n"]),
        })

    summary_df = pd.DataFrame(rows).sort_values(["setting", "mode"]).reset_index(drop=True)
    summary_path = metrics_dir / "metrics_summary_ablation.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # per-seed distribution file for AllPlot
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["seed", "setting"]).reset_index(drop=True)
    delta_path = metrics_dir / "delta_r2_by_seed.csv"
    per_seed_df.to_csv(delta_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Saved:")
    print(" -", summary_path)
    print(" -", delta_path)
    print("Predictions in:", preds_dir)
    print("Metrics in     :", metrics_dir)
    print("=" * 70)
    print(summary_df[["setting", "mode", "r2_median", "r2_q1", "r2_q3", "n_repeats"]].to_string(index=False))


if __name__ == "__main__":
    main()
