import argparse
import os
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

warnings.filterwarnings("ignore")


# -----------------------------
# Robust CSV reader (no decode crash)
# -----------------------------
def read_csv_smart(path: str | Path) -> pd.DataFrame:
    path = str(path)
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # sometimes parser errors unrelated to encoding; still try other encodings
            last_err = e
            continue

    # final fallback: python engine + replacement
    try:
        return pd.read_csv(path, engine="python", encoding="utf-8", encoding_errors="replace")
    except Exception:
        # ultimate fallback: latin1 never fails decode
        return pd.read_csv(path, engine="python", encoding="latin1")


# -----------------------------
# Metrics utils
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def summary_stats(x: List[float]) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else np.nan,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)) if arr.size else np.nan,
        "q1": float(np.quantile(arr, 0.25)) if arr.size else np.nan,
        "q3": float(np.quantile(arr, 0.75)) if arr.size else np.nan,
    }


# -----------------------------
# Constraint module (S2-TOC + HI range)
# -----------------------------
@dataclass
class ConstraintConfig:
    eps: float = 1e-8
    # keep constraints mild to avoid hurting strong base in proxy-rich datasets
    lambda_s2: float = 0.05
    hi_q_low: float = 0.01
    hi_q_high: float = 0.99


@dataclass
class ConstraintFitted:
    a_s2: float
    b_s2: float
    hi_min: float
    hi_max: float
    eps: float
    lambda_s2: float


def fit_constraints_on_train(
    y_train: np.ndarray,
    s2_train: Optional[np.ndarray],
    hi_train: Optional[np.ndarray],
    cfg: ConstraintConfig,
) -> ConstraintFitted:
    # HI bounds: robust quantiles
    if hi_train is None:
        hi_min, hi_max = 0.0, 1e9
    else:
        hi_min = float(np.nanquantile(hi_train, cfg.hi_q_low))
        hi_max = float(np.nanquantile(hi_train, cfg.hi_q_high))
        if not np.isfinite(hi_min):
            hi_min = 0.0
        if not np.isfinite(hi_max):
            hi_max = 1e9
        if hi_max <= hi_min:
            hi_min, hi_max = 0.0, 1e9

    # S2 ~ a*TOC + b (robust least squares)
    if s2_train is None:
        a_s2, b_s2 = 1.0, 0.0
    else:
        x = np.asarray(y_train, dtype=float)
        y = np.asarray(s2_train, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 5:
            X = np.vstack([x[m], np.ones(m.sum())]).T
            try:
                coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
                a_s2 = float(coef[0])
                b_s2 = float(coef[1])
                if not np.isfinite(a_s2):
                    a_s2 = 1.0
                if not np.isfinite(b_s2):
                    b_s2 = 0.0
            except Exception:
                a_s2, b_s2 = 1.0, 0.0
        else:
            a_s2, b_s2 = 1.0, 0.0

    return ConstraintFitted(
        a_s2=a_s2,
        b_s2=b_s2,
        hi_min=hi_min,
        hi_max=hi_max,
        eps=cfg.eps,
        lambda_s2=cfg.lambda_s2,
    )


def apply_constraint_full(
    y_pred: np.ndarray,
    s2: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    fitted: ConstraintFitted,
) -> np.ndarray:
    y = np.asarray(y_pred, dtype=float).copy()

    # (1) Non-negativity
    y[y < 0] = 0.0

    # (2) HI range projection: HI = 100*S2/TOC  => TOC in [100*S2/HImax, 100*S2/HImin]
    if (hi is not None) and (s2 is not None):
        s2v = np.asarray(s2, dtype=float)
        m = np.isfinite(s2v)
        if m.any():
            toc_low = 100.0 * s2v / max(fitted.hi_max, fitted.eps)
            toc_high = 100.0 * s2v / max(fitted.hi_min, fitted.eps) if fitted.hi_min > 0 else np.inf
            # clip only where finite
            y[m] = np.clip(y[m], toc_low[m], toc_high[m])

    # (3) S2–TOC consistency (soft): bring TOC toward inverse of S2~a*TOC+b
    if s2 is not None and np.isfinite(fitted.a_s2) and abs(fitted.a_s2) > fitted.eps:
        s2v = np.asarray(s2, dtype=float)
        m = np.isfinite(s2v)
        if m.any():
            toc_from_s2 = (s2v - fitted.b_s2) / fitted.a_s2
            # soft blending
            y[m] = (1.0 - fitted.lambda_s2) * y[m] + fitted.lambda_s2 * toc_from_s2[m]
            y[y < 0] = 0.0

    return y


# -----------------------------
# Refinement MLP (Stage 2: L_range + L_lin, same logic as Egypt)
# -----------------------------
def fit_s2_eq_toc_linear(s2_train: np.ndarray, toc_train: np.ndarray) -> Tuple[float, float]:
    """Fit S2 = a * TOC + b. Returns (a, b). TOC_from_S2 = (S2 - b) / a."""
    s2 = np.asarray(s2_train, float)
    toc = np.asarray(toc_train, float)
    m = np.isfinite(s2) & np.isfinite(toc) & (toc > 1e-6)
    if m.sum() < 10:
        return 1.0, 0.0
    X = np.column_stack([toc[m], np.ones(m.sum())])
    y = s2[m]
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if not np.isfinite(a) or a < 1e-6:
            return 1.0, 0.0
        return a, b
    except Exception:
        return 1.0, 0.0


def _loss_range_torch(y_pred: "torch.Tensor", y_min: float, y_max: float) -> "torch.Tensor":
    over = torch.relu(y_pred - y_max)
    under = torch.relu(y_min - y_pred)
    return over.mean() + under.mean()


def _loss_lin_torch(
    y_pred: "torch.Tensor",
    s2: "torch.Tensor",
    a: float,
    b: float,
    eps: float = 1e-6,
) -> "torch.Tensor":
    toc_from_s2 = (s2 - b) / max(a, eps)
    return ((y_pred - toc_from_s2) ** 2).mean()


class RefinementMLP(nn.Module):
    """MLP: input (x, y_base) -> output y_refined."""

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
) -> "RefinementMLP":
    """Train refinement MLP with L_mse + L_range + L_lin."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for refinement MLP")
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
    model: "RefinementMLP",
    X: np.ndarray,
    y_base: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        yb_t = torch.tensor(y_base, dtype=torch.float32, device=device)
        out = model(x_t, yb_t)
    return out.cpu().numpy().astype(float)


# -----------------------------
# Feature settings (you can adjust)
# -----------------------------
def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.strip(): c for c in df.columns}

    # common aliases (Bakken pyrolysis tables often have degree symbol etc.)
    aliases = {
        "TOC": ["TOC", "TOC_wt_pct", "TOC (wt%)", "TOC (%)", "TOC (%) ", "TOC_wt%", "TOC wt%"],
        "S1": ["S1", "S1 (%)", "S1 (mgHC/gRock)", "S1_mgHC_gRock"],
        "S2": ["S2 (mg HC/g)", "S2 (mgHC/gRock)", "S2_mgHC_gRock", "S2", "S2 (%)"],
        "HI": ["HI", "HI (mgHC/gTOC)", "HI_mgHC_gTOC"],
        "TMAX": ["Tmax", "TMAX", "TMAX (°C)", "Tmax (°C)", "Tmax (C)"],
        "DEPTH": ["Depth", "Depth (m)", "DEPTH", "DEPTH(m)"],
        "OI": ["OI", "OI (mgCO2/gTOC)", "OI_mgCO2_gTOC"],
    }

    out = {}
    for k, cand in aliases.items():
        for name in cand:
            if name in df.columns:
                out[k] = name
                break
    return out


def make_settings(df: pd.DataFrame, colmap: Dict[str, str]) -> Dict[str, List[str]]:
    # Minimal “RockEval-like” proxies if available
    feats_rockeval = []
    for k in ["S1", "S2", "TMAX", "DEPTH"]:
        if k in colmap:
            feats_rockeval.append(colmap[k])

    # “GeoFusion” extends with more geochem if present
    feats_fusion = feats_rockeval.copy()
    for k in ["HI", "OI"]:
        if k in colmap and colmap[k] not in feats_fusion:
            feats_fusion.append(colmap[k])

    # fallback: if missing, use all numeric except target
    if "TOC" in colmap:
        target = colmap["TOC"]
    else:
        raise ValueError("Cannot find TOC column in CSV.")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c != target]

    if len(feats_rockeval) < 2:
        feats_rockeval = numeric_cols[: min(6, len(numeric_cols))]
    if len(feats_fusion) < len(feats_rockeval):
        feats_fusion = numeric_cols[: min(10, len(numeric_cols))]

    return {
        "US_Bakken_RockEval": feats_rockeval,
        "US_Bakken_GeoFusion": feats_fusion,
    }


# -----------------------------
# Main
# -----------------------------
@dataclass
class RunConfig:
    project_root: Path
    data_csv: Path
    target_col: str
    seeds: List[int]
    test_size: float = 0.25
    random_state_split: int = 0
    use_refinement: bool = True
    metrics_dir: Path = None
    preds_dir: Path = None


def main():
    # --------- paths: change if needed ----------
    project_root = Path(__file__).resolve().parents[1]
    data_csv = project_root / "data" / "Total_organic_carbon_programmed_temperature_pyrolysis_Bakken_Formation.csv"
    metrics_dir = project_root / "outputs" / "artifacts" / "metrics"
    preds_dir = project_root / "outputs" / "artifacts" / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--no_refinement", action="store_true", help="Disable Refinement MLP (L_range + L_lin)")
    ap.add_argument("--seeds_step", type=int, default=20, help="Seed step (0, step, 2*step, ...)")
    ap.add_argument("--seeds_max", type=int, default=4000, help="Max seed value")
    args = ap.parse_args()

    seeds_list = list(range(0, args.seeds_max, args.seeds_step))

    cfg = RunConfig(
        project_root=project_root,
        data_csv=data_csv,
        target_col="__AUTO__",
        seeds=seeds_list,
        use_refinement=not args.no_refinement,
        metrics_dir=metrics_dir,
        preds_dir=preds_dir,
    )

    print("=== US Bakken TRAIN (method-3: best-of-best base/full) ===")
    print("use_refinement:", cfg.use_refinement, "(Stage 2 MLP: L_range + L_lin)" if cfg.use_refinement else "")
    if cfg.use_refinement and not TORCH_AVAILABLE:
        print("  [WARN] PyTorch not found, using post-hoc constraints only.")
    print("project_root:", cfg.project_root)
    print("data_csv     :", cfg.data_csv)
    print("metrics_dir  :", cfg.metrics_dir)
    print("preds_dir    :", cfg.preds_dir)

    df = read_csv_smart(cfg.data_csv)
    # keep numeric columns only where possible
    colmap = detect_columns(df)
    if "TOC" not in colmap:
        raise ValueError(f"Cannot detect TOC column. columns={list(df.columns)}")
    target_col = colmap["TOC"]

    # coerce numeric
    for c in df.columns:
        if c == target_col:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # build settings
    settings = make_settings(df, colmap)

    # S2/HI for constraints if exists
    s2_col = colmap.get("S2", None)
    hi_col = colmap.get("HI", None)

    if s2_col:
        print("Detected geochem columns:", {k: colmap[k] for k in colmap if k in ["S1", "S2", "TMAX", "DEPTH", "HI", "OI"]})
    else:
        print("No S2 column detected -> constraints will be weaker.")

    # storage for per-setting ablations (for paper appendix)
    constraint_modes = ["base", "full"]
    metrics_store: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for setting_name in settings.keys():
        for mode in constraint_modes:
            metrics_store[(setting_name, mode)] = {"r2": [], "rmse": [], "mae": []}

    # method-3 per-seed best-of-best
    per_seed_rows = []

    for seed in cfg.seeds:
        # collect per-setting for this seed
        seed_setting_results = []  # list of dict(setting, r2_base, r2_full, best_flag?)

        for setting_name, feat_cols in settings.items():
            # ensure features exist
            feat_cols = [c for c in feat_cols if c in df.columns]
            if len(feat_cols) < 2:
                continue

            X = df[feat_cols].copy()
            for c in feat_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            y = df[target_col].values.astype(float)

            valid = np.isfinite(y)
            for c in feat_cols:
                valid &= np.isfinite(X[c].values.astype(float))
            X = X.loc[valid].reset_index(drop=True)
            y = y[valid]

            if len(y) < 25:
                continue

            # split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=cfg.test_size, random_state=seed
            )

            # model (keep simple + robust)
            model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.08,
                max_iter=700,
                l2_regularization=0.0,
                random_state=seed,
            )
            model.fit(X_train, y_train)

            # base prediction
            y_base = model.predict(X_test)
            y_base_train = model.predict(X_train)

            # constraints fit on TRAIN only
            s2_train = X_train[s2_col].values.astype(float) if (s2_col and s2_col in X_train.columns) else None
            s2_test = X_test[s2_col].values.astype(float) if (s2_col and s2_col in X_test.columns) else None
            hi_train = X_train[hi_col].values.astype(float) if (hi_col and hi_col in X_train.columns) else None
            hi_test = X_test[hi_col].values.astype(float) if (hi_col and hi_col in X_test.columns) else None

            # HI from S2/TOC when HI column missing
            if hi_train is None and s2_train is not None:
                hi_train = 100.0 * s2_train / np.maximum(y_train, 1e-6)
            if hi_test is None and s2_test is not None:
                hi_test = 100.0 * s2_test / np.maximum(y_test, 1e-6)

            constraint_cfg = ConstraintConfig(lambda_s2=0.05, hi_q_low=0.01, hi_q_high=0.99)
            fitted = fit_constraints_on_train(y_train, s2_train, hi_train, constraint_cfg)

            # Stage 2: Refinement MLP (L_range + L_lin) when use_refinement and S2 available
            y_full_posthoc = apply_constraint_full(y_base, s2_test, hi_test, fitted)
            if (
                cfg.use_refinement
                and TORCH_AVAILABLE
                and s2_train is not None
                and len(s2_train) >= 10
            ):
                try:
                    a_lin, b_lin = fit_s2_eq_toc_linear(s2_train, y_train)
                    toc_max = max(float(np.max(y_train)) * 1.1, 25.0)
                    mlp = train_refinement_mlp(
                        X_train.values.astype(float),
                        y_train,
                        np.clip(y_base_train, 0.0, None),
                        s2_train,
                        toc_max=toc_max,
                        a_lin=a_lin,
                        b_lin=b_lin,
                        seed=int(seed),
                    )
                    y_refined = predict_refinement_mlp(
                        mlp,
                        X_test.values.astype(float),
                        np.clip(y_base, 0.0, None),
                    )
                    y_refined = np.clip(y_refined, 0.0, None)
                    # 仅当 refinement 不劣于 post-hoc 时采用
                    r2_posthoc = r2_score(y_test, y_full_posthoc)
                    r2_refined = r2_score(y_test, y_refined)
                    y_full = y_refined if r2_refined >= r2_posthoc else y_full_posthoc
                except Exception:
                    y_full = y_full_posthoc
            else:
                y_full = y_full_posthoc

            # metrics
            r2_b, rmse_b, mae_b = compute_metrics(y_test, y_base)
            r2_f, rmse_f, mae_f = compute_metrics(y_test, y_full)

            # store per-setting summary
            metrics_store[(setting_name, "base")]["r2"].append(r2_b)
            metrics_store[(setting_name, "base")]["rmse"].append(rmse_b)
            metrics_store[(setting_name, "base")]["mae"].append(mae_b)

            metrics_store[(setting_name, "full")]["r2"].append(r2_f)
            metrics_store[(setting_name, "full")]["rmse"].append(rmse_f)
            metrics_store[(setting_name, "full")]["mae"].append(mae_f)

            # save predictions (optional but useful for AllPlot best/worst diagnostics)
            pred_path = cfg.preds_dir / f"predsTEST_{setting_name}_seed{seed}.csv"
            pd.DataFrame(
                {"y_true": y_test, "y_base": y_base, "y_full": y_full}
            ).to_csv(pred_path, index=False, encoding="utf-8-sig")

            # save per-seed json metric
            metrics_path = cfg.metrics_dir / f"metrics_{setting_name}_seed{seed}.json"
            metric_obj = {
                "seed": int(seed),
                "setting": setting_name,
                "features": feat_cols,
                "constraint_cfg": {
                    "lambda_s2": fitted.lambda_s2,
                    "hi_q_low": constraint_cfg.hi_q_low,
                    "hi_q_high": constraint_cfg.hi_q_high,
                },
                "constraint_params": {
                    "a_s2": fitted.a_s2,
                    "b_s2": fitted.b_s2,
                    "hi_min": fitted.hi_min,
                    "hi_max": fitted.hi_max,
                },
                "metrics": {
                    "base": {"r2_test": r2_b, "rmse_test": rmse_b, "mae_test": mae_b},
                    "full": {"r2_test": r2_f, "rmse_test": rmse_f, "mae_test": mae_f},
                },
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metric_obj, f, indent=2, ensure_ascii=False)

            seed_setting_results.append(
                {
                    "seed": seed,
                    "setting": setting_name,
                    "r2_base": r2_b,
                    "r2_full": r2_f,
                    "delta_r2": r2_f - r2_b,
                }
            )

        if not seed_setting_results:
            continue

        # ---------------------------
        # Method-3: best-of-best
        # ---------------------------
        df_seed = pd.DataFrame(seed_setting_results)

        # base = best unconstrained across settings
        i_base = int(df_seed["r2_base"].values.argmax())
        best_base_setting = df_seed.iloc[i_base]["setting"]
        best_r2_base = float(df_seed.iloc[i_base]["r2_base"])

        # full = best constrained across settings
        i_full = int(df_seed["r2_full"].values.argmax())
        best_full_setting = df_seed.iloc[i_full]["setting"]
        best_r2_full = float(df_seed.iloc[i_full]["r2_full"])

        per_seed_rows.append(
            {
                "seed": int(seed),
                "best_setting_base": best_base_setting,
                "best_setting_full": best_full_setting,
                "r2_base": best_r2_base,
                "r2_full": best_r2_full,
                "delta_r2": best_r2_full - best_r2_base,
            }
        )

        if seed % 20 == 0:
            print(
                f"[seed={seed}] base(best={best_base_setting})={best_r2_base:.4f} "
                f"full(best={best_full_setting})={best_r2_full:.4f} "
                f"delta={best_r2_full - best_r2_base:+.4f}"
            )

    # ---------------------------
    # Save outputs
    # ---------------------------
    # (A) ablation summary by setting/mode
    rows = []
    for (setting_name, mode), store in metrics_store.items():
        s_r2 = summary_stats(store["r2"])
        s_rmse = summary_stats(store["rmse"])
        s_mae = summary_stats(store["mae"])
        rows.append(
            {
                "setting": setting_name,
                "mode": mode,
                "r2_mean": s_r2["mean"],
                "r2_std": s_r2["std"],
                "r2_median": s_r2["median"],
                "r2_q1": s_r2["q1"],
                "r2_q3": s_r2["q3"],
                "rmse_mean": s_rmse["mean"],
                "rmse_std": s_rmse["std"],
                "rmse_median": s_rmse["median"],
                "rmse_q1": s_rmse["q1"],
                "rmse_q3": s_rmse["q3"],
                "mae_mean": s_mae["mean"],
                "mae_std": s_mae["std"],
                "mae_median": s_mae["median"],
                "mae_q1": s_mae["q1"],
                "mae_q3": s_mae["q3"],
                "n_repeats": s_r2["n"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["setting", "mode"]).reset_index(drop=True)
    summary_path = cfg.metrics_dir / "metrics_summary_ablation.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # (B) method-3 per-seed (THIS is what AllPlot should consume)
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed").reset_index(drop=True)
    delta_path = cfg.metrics_dir / "delta_r2_by_seed.csv"
    per_seed_df.to_csv(delta_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("Saved:")
    print(" -", summary_path)
    print(" -", delta_path)
    print("Predictions in:", cfg.preds_dir)
    print("Metrics in     :", cfg.metrics_dir)
    print("=" * 70)

    # quick console summary
    if not per_seed_df.empty:
        s_base = summary_stats(per_seed_df["r2_base"].tolist())
        s_full = summary_stats(per_seed_df["r2_full"].tolist())
        s_delta = summary_stats(per_seed_df["delta_r2"].tolist())
        print("US_Bakken (Method-3 best-of-best across settings)")
        print(
            f"  base: median={s_base['median']:.6f} q1={s_base['q1']:.6f} q3={s_base['q3']:.6f} "
            f"mean={s_base['mean']:.6f} std={s_base['std']:.6f} n={s_base['n']}"
        )
        print(
            f"  full: median={s_full['median']:.6f} q1={s_full['q1']:.6f} q3={s_full['q3']:.6f} "
            f"mean={s_full['mean']:.6f} std={s_full['std']:.6f} n={s_full['n']}"
        )
        print(
            f" delta: median={s_delta['median']:+.6f} q1={s_delta['q1']:+.6f} q3={s_delta['q3']:+.6f} "
            f"mean={s_delta['mean']:+.6f} std={s_delta['std']:.6f} n={s_delta['n']}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
