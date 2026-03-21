#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Energies 数据集 TOC 回归训练脚本（XGBoost / HistGradientBoosting，含电阻率消融实验）

运行方式（默认路径）:
    python train_toc_xgb.py

可选参数:
    --data_path   默认: <project_root>/data/energies-2133761-supplementary.xlsx
    --out_dir     默认: <project_root>/outputs
    --n_repeats   默认: 5  (对应 seeds = 0..4)
    --test_size   默认: 0.15
    --val_size    默认: 0.15
    --use_xgb     默认: True  (可传 False/0/true/1 等字符串)
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ==================== 全局随机种子与绘图风格 ====================

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# 与现有项目风格保持一致（中文、顶刊风格、高清）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 320
plt.rcParams["savefig.dpi"] = 320
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {v}")


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def try_import_xgb():
    """尝试导入 xgboost, 若失败则返回 (False, None) 并提示使用 HistGradientBoosting。"""
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # noqa: BLE001
        print("=" * 60)
        print("警告: 无法导入 xgboost，已自动回退到 HistGradientBoostingRegressor。")
        print("错误信息:", repr(e))
        print("如需获得更强性能，建议安装 xgboost:")
        print("  pip install xgboost")
        print("=" * 60)
        return False, None
    return True, xgb


def detect_columns(df: pd.DataFrame):
    """自动识别特征列与标签列，兼容不同命名方式。"""
    print("=" * 60)
    print("数据列名如下：")
    print(list(df.columns))
    print("=" * 60)

    # 统一去掉两侧空格并转为小写以便匹配
    normalized_cols = {c: c.strip().lower() for c in df.columns}

    def find_col(candidates):
        for cand in candidates:
            cand_norm = cand.strip().lower()
            # 精确匹配
            for raw, norm in normalized_cols.items():
                if norm == cand_norm:
                    return raw
            # 部分匹配（如 "TOC (%)" vs "toc(%)"）
            for raw, norm in normalized_cols.items():
                if cand_norm in norm or norm in cand_norm:
                    return raw
        return None

    feature_aliases = {
        "AC": ["AC", "ac", "AC_US", "sonic"],
        "GR": ["GR", "gr", "Gamma", "gamma_ray", "gammaray"],
        "K": ["K", "k", "K (%)", "Potassium"],
        "TH": ["TH", "th", "Th", "Thorium"],
        "U": ["U", "u", "Uranium"],
        "RD": ["RD", "rd", "Rdeep", "R_deep", "ResDeep"],
        "RS": ["RS", "rs", "Rshallow", "R_shallow", "ResShallow"],
    }

    label_candidates = [
        "TOC",
        "TOC (%)",
        "TOC(%)",
        "toc",
        "Total organic carbon",
        "TOC_wt_pct",
    ]

    feature_cols = {}
    for std_name, aliases in feature_aliases.items():
        col = find_col(aliases)
        if col is None:
            raise ValueError(f"未找到特征列 {std_name}，尝试过的别名: {aliases}")
        feature_cols[std_name] = col

    label_col = find_col(label_candidates)
    if label_col is None:
        raise ValueError(
            f"未找到 TOC 标签列，请检查 Excel。尝试过的候选列名: {label_candidates}"
        )

    print("列名自动匹配结果：")
    for k, v in feature_cols.items():
        print(f"  Feature {k} -> {v}")
    print(f"  Label TOC   -> {label_col}")
    print("=" * 60)

    return feature_cols, label_col


def load_and_clean_data(xlsx_path: Path, out_clean_path: Path):
    """读取 Excel, 自动识别列名, 清洗缺失值, 构造派生特征, 保存 cleaned_data.csv。"""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {xlsx_path}")

    print("=" * 60)
    print(f"读取 Excel: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Training and testing data")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"读取 Excel 失败: {e}") from e

    print(f"原始数据形状: {df.shape}")
    feature_cols_map, label_col = detect_columns(df)

    # 只保留需要的列（原始 AC/GR/K/TH/U/RD/RS + TOC）
    used_cols = list(feature_cols_map.values()) + [label_col]
    df_used = df[used_cols].copy()

    # 所有原始特征和标签统一转为 float
    for col in used_cols:
        df_used[col] = pd.to_numeric(df_used[col], errors="coerce")

    # 丢弃缺失（在构造派生特征之前完成）
    before = len(df_used)
    df_used = df_used.dropna()
    after = len(df_used)
    print(f"清洗后数据形状: {df_used.shape} (删除 NaN 行: {before - after})")

    # ==================== Step A: 增加 domain-informed 派生特征 ====================
    print("开始构造派生特征 (log_RD/log_RS/dlogR/sumlogR 及谱比值 U_TH/U_K/TH_K)...")
    eps = 1e-6

    rd_col = feature_cols_map["RD"]
    rs_col = feature_cols_map["RS"]
    k_col = feature_cols_map["K"]
    th_col = feature_cols_map["TH"]
    u_col = feature_cols_map["U"]

    # 电阻率相关对数特征
    df_used["log_RD"] = np.log1p(df_used[rd_col])
    df_used["log_RS"] = np.log1p(df_used[rs_col])
    df_used["dlogR"] = df_used["log_RD"] - df_used["log_RS"]
    df_used["sumlogR"] = df_used["log_RD"] + df_used["log_RS"]

    # 谱比值特征
    df_used["U_TH"] = df_used[u_col] / (df_used[th_col] + eps)
    df_used["U_K"] = df_used[u_col] / (df_used[k_col] + eps)
    df_used["TH_K"] = df_used[th_col] / (df_used[k_col] + eps)

    # 处理可能出现的极端 inf / nan：用该列有限值的中位数填充
    ratio_cols = ["U_TH", "U_K", "TH_K"]
    for col in ratio_cols:
        values = df_used[col].values.astype(float)
        finite_mask = np.isfinite(values)
        if not finite_mask.all():
            finite_vals = values[finite_mask]
            if len(finite_vals) > 0:
                median_val = float(np.median(finite_vals))
            else:
                median_val = 0.0
            values[~finite_mask] = median_val
            df_used[col] = values
            print(f"  列 {col}: 发现非有限值，已用中位数 {median_val:.4f} 填充")

    # 保存带派生特征的清洗数据，方便复现实验
    df_used.to_csv(out_clean_path, index=False, encoding="utf-8-sig")
    print(f"清洗 + 派生特征后的数据已保存至: {out_clean_path}")
    print(f"最终可用样本数: {len(df_used)}")
    print("=" * 60)

    return df_used, feature_cols_map, label_col


def _write_cache_for_allplot(
    project_root: Path,
    out_dir: Path,
    data_path: Path,
) -> None:
    """
    固化聚合数据：读取 delta_r2_by_seed.csv，写入 AllPlot/data/CN_Multi_Basins.npz 与 .meta.json。
    仅在训练结束后调用，不改变任何训练逻辑。
    """
    metrics_dir = out_dir / "artifacts" / "metrics"
    preds_dir = out_dir / "artifacts" / "predictions"
    delta_path = metrics_dir / "delta_r2_by_seed.csv"
    if not delta_path.exists():
        print(f"[cache] 跳过: {delta_path} 不存在")
        return

    sim_root = project_root.parent
    cache_dir = sim_root / "AllPlot" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(delta_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    base_col = "r2_base" if "r2_base" in df.columns else ("r2_nores" if "r2_nores" in df.columns else None)
    full_col = "r2_full" if "r2_full" in df.columns else None
    if base_col is None or full_col is None or base_col not in df.columns or full_col not in df.columns:
        print(f"[cache] 跳过: delta csv 缺少 r2_base/r2_nores 或 r2_full")
        return

    seeds = df["seed"].astype(int).to_numpy()
    r2_base = df[base_col].astype(float).to_numpy()
    r2_full = df[full_col].astype(float).to_numpy()
    delta_r2 = r2_full - r2_base if "delta_r2" not in df.columns else df["delta_r2"].astype(float).to_numpy()

    best_idx = int(np.argmax(delta_r2))
    worst_idx = int(np.argmin(delta_r2))
    med_val = float(np.median(delta_r2))
    median_idx = int(np.argmin(np.abs(delta_r2 - med_val)))
    best_seed = int(seeds[best_idx])
    worst_seed = int(seeds[worst_idx])
    median_seed = int(seeds[median_idx])

    def _load_diag(seed: int):
        nores_f = preds_dir / f"predsTEST_NoRes_seed{seed}.csv"
        full_f = preds_dir / f"predsTEST_Full_seed{seed}.csv"
        if not nores_f.exists() or not full_f.exists():
            return None
        try:
            df_n = pd.read_csv(nores_f, encoding="utf-8-sig")
            df_f = pd.read_csv(full_f, encoding="utf-8-sig")
            if "index" not in df_n.columns or "index" not in df_f.columns:
                return None
            m = pd.merge(
                df_n[["index", "y_true", "y_pred"]].rename(columns={"y_pred": "y_base"}),
                df_f[["index", "y_pred"]].rename(columns={"y_pred": "y_full"}),
                on="index",
                how="inner",
            )
            if "y_true" not in m.columns or "y_base" not in m.columns or "y_full" not in m.columns:
                return None
            return (
                m["y_true"].to_numpy(float),
                m["y_base"].to_numpy(float),
                m["y_full"].to_numpy(float),
            )
        except Exception:
            return None

    cache = {
        "seeds": seeds,
        "r2_base": r2_base.astype(float),
        "r2_full": r2_full.astype(float),
        "delta_r2": delta_r2.astype(float),
        "best_seed": best_seed,
        "worst_seed": worst_seed,
        "median_seed": median_seed,
        "key": np.array("CN_Multi_Basins", dtype=object),
        "label": np.array("CN Multi-basins (Yanchang, Shahejie, Longmaxi, Shanxi/Taiyuan)", dtype=object),
        "source": np.array(str(delta_path.resolve()), dtype=object),
    }

    for tag, s in [("best", best_seed), ("worst", worst_seed), ("median", median_seed)]:
        diag = _load_diag(s)
        if diag is not None:
            cache[f"{tag}_y_true"] = diag[0]
            cache[f"{tag}_y_base"] = diag[1]
            cache[f"{tag}_y_full"] = diag[2]

    npz_path = cache_dir / "CN_Multi_Basins.npz"
    meta_path = cache_dir / "CN_Multi_Basins.meta.json"
    np.savez_compressed(npz_path, **cache)
    meta = {
        "key": "CN_Multi_Basins",
        "label": "CN Multi-basins (Yanchang, Shahejie, Longmaxi, Shanxi/Taiyuan)",
        "project_root": str(project_root.resolve()),
        "data_file": str(data_path.resolve()),
        "metrics_dir": str(metrics_dir.resolve()),
        "preds_dir": str(preds_dir.resolve()),
        "seed_list": seeds.tolist(),
        "best_seed": best_seed,
        "worst_seed": worst_seed,
        "median_seed": median_seed,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": "cache for plotting; training logic unchanged",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cache] 固化聚合 -> {npz_path}")
    print(f"[cache] 元信息   -> {meta_path}")


def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}


# ==================== Refinement MLP with geological constraints (L_range + L_mono) ====================
# CN 数据无 S2，仅使用 L_range 与 L_mono（电阻率/GR/U 与 TOC 单调性）


def _loss_range_torch(y_pred, y_min: float, y_max: float):
    """L_range = E[max(0, y - y_max) + max(0, y_min - y)]"""
    over = torch.relu(y_pred - y_max)
    under = torch.relu(y_min - y_pred)
    return over.mean() + under.mean()


def _loss_mono_torch(model, x: torch.Tensor, y_base: torch.Tensor, monotonic_indices: list, monotonic_signs: list):
    """L_mono: 惩罚违反单调性的梯度。RefinementMLP 输入为 (x, y_base)。"""
    x = x.clone().detach().requires_grad_(True)
    y_pred = model(x, y_base)
    if y_pred.dim() > 1:
        y_pred = y_pred.squeeze(-1)
    loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    for j, sign in zip(monotonic_indices, monotonic_signs):
        (dy_dxj,) = torch.autograd.grad(y_pred.sum(), x, retain_graph=True, create_graph=False)
        grad_j = dy_dxj[:, j]
        violation = torch.relu(-sign * grad_j)
        loss = loss + violation.mean()
    return loss / max(len(monotonic_indices), 1)


class RefinementMLP(nn.Module):
    """小 MLP: 输入 (x, y_base) -> 输出 y_refined。"""

    def __init__(self, n_features: int, hidden=(32, 16)):
        super().__init__()
        dims = [n_features + 1] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x, y_base):
        inp = torch.cat([x, y_base.unsqueeze(1)], dim=1)
        return self.net(inp).squeeze(-1)


def train_refinement_mlp_cn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_base_train: np.ndarray,
    toc_max: float,
    monotonic_indices: list,
    monotonic_signs: list,
    n_epochs: int = 150,
    lr: float = 1e-2,
    lambda_range: float = 0.1,
    lambda_mono: float = 0.02,
    seed: int = 0,
):
    """训练 Refinement MLP：L_mse + L_range + L_mono（无 S2，无 L_lin）。"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for refinement MLP")
    torch.manual_seed(seed)
    device = torch.device("cpu")
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_true = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_base = torch.tensor(y_base_train, dtype=torch.float32, device=device)

    model = RefinementMLP(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epochs):
        opt.zero_grad()
        y_pred = model(X, y_base)
        L_mse = ((y_pred - y_true) ** 2).mean()
        L_range = _loss_range_torch(y_pred, 0.0, toc_max)
        L_mono = _loss_mono_torch(model, X, y_base, monotonic_indices, monotonic_signs)
        loss = L_mse + lambda_range * L_range + lambda_mono * L_mono
        loss.backward()
        opt.step()

    return model


def predict_refinement_mlp(model, X: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    """从 (X, y_base) 预测 y_refined。"""
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        yb_t = torch.tensor(y_base, dtype=torch.float32, device=device)
        out = model(x_t, yb_t)
    return out.cpu().numpy().astype(float)


def get_monotonic_indices_for_full(feature_keys: list) -> tuple:
    """
    Full 特征顺序: AC, GR, K, TH, U, log_RD, log_RS, dlogR, sumlogR, U_TH, U_K, TH_K
    单调正相关 (∂TOC/∂x >= 0): GR, U, log_RD, log_RS, U_TH
    """
    name_to_idx = {k: i for i, k in enumerate(feature_keys)}
    mono_names = ["GR", "U", "log_RD", "log_RS", "U_TH"]
    indices = []
    signs = []
    for n in mono_names:
        if n in name_to_idx:
            indices.append(name_to_idx[n])
            signs.append(1)
    return indices, signs


def plot_scatter(y_true, y_pred, setting, out_path, metrics=None):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.xlabel("True TOC")
    plt.ylabel("Predicted TOC")
    plt.title(f"{setting} - True vs Predicted TOC")
    # 在右上角添加指标标注（若提供）
    if metrics is not None:
        txt = (
            f"R2={metrics['r2']:.2f}\n"
            f"RMSE={metrics['rmse']:.2f}\n"
            f"MAE={metrics['mae']:.2f}"
        )
        ax = plt.gca()
        ax.text(
            0.98,
            0.02,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"保存散点图: {out_path}")


def plot_residual_hist(y_true, y_pred, setting, out_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(0.0, color="r", linestyle="--", linewidth=2)
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{setting} - Residual Distribution")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"保存残差直方图: {out_path}")


def plot_feature_importance(model, feature_names, setting, out_path):
    if not hasattr(model, "feature_importances_"):
        print(f"警告: 模型 {setting} 不支持 feature_importances_，跳过特征重要性绘图。")
        return
    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_vals = importances[order]

    plt.figure(figsize=(6, 4.5))
    y_pos = np.arange(len(sorted_names))
    plt.barh(y_pos, sorted_vals)
    plt.yticks(y_pos, sorted_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Feature Importance")
    plt.title(f"{setting} - Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"保存特征重要性图: {out_path}")


def run_experiment(
    df: pd.DataFrame,
    feature_cols_map: dict,
    label_col: str,
    out_dir: Path,
    n_repeats: int,
    test_size: float,
    val_size: float,
    use_xgb_flag: bool,
    use_refinement: bool = True,
):
    # 目录结构
    logs_dir = out_dir / "logs"
    figs_dir = out_dir / "figures"
    models_dir = out_dir / "models"
    tables_dir = out_dir / "tables"
    artifacts_metrics_dir = out_dir / "artifacts" / "metrics"
    preds_dir = out_dir / "artifacts" / "predictions"
    for d in [out_dir, logs_dir, figs_dir, models_dir, tables_dir, artifacts_metrics_dir, preds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    xgb_available, xgb_module = try_import_xgb()
    use_xgb_effective = use_xgb_flag and xgb_available

    print(f"use_refinement = {use_refinement} (Full 阶段 L_range + L_mono)" + (
        "" if TORCH_AVAILABLE else " [PyTorch 未安装，跳过]"
    ))

    # 设置特征组合（包含派生特征）
    setting_defs = {
        # Full: 原始 AC/GR/K/TH/U + 电阻率对数特征 + dlogR/sumlogR + 谱比值
        "Full": [
            "AC",
            "GR",
            "K",
            "TH",
            "U",
            "log_RD",
            "log_RS",
            "dlogR",
            "sumlogR",
            "U_TH",
            "U_K",
            "TH_K",
        ],
        # NoRes: 不使用 RD/RS 及其对数与差分，但保留谱比值
        "NoRes": [
            "AC",
            "GR",
            "K",
            "TH",
            "U",
            "U_TH",
            "U_K",
            "TH_K",
        ],
    }

    # 准备全体数据（标签保持原始 TOC，用于指标与绘图；训练用 log1p(TOC)）
    label_series = df[label_col].astype(float)
    n_samples_total = len(label_series)
    print(f"用于建模的样本总数: {n_samples_total}")
    toc_all = label_series.values.astype(float)
    # Step B: log1p 变换作为训练目标，确保非负
    y_all_log = np.log1p(np.clip(toc_all, 0.0, None))

    results_summary = {}
    best_models = {}
    all_r2 = {}  # per-seed r2 for delta_r2_by_seed.csv (output only)

    # 对两个 setting 分别做实验
    for setting_name, feature_keys in setting_defs.items():
        print("=" * 60)
        print(f"开始实验: {setting_name}")
        print(f"使用特征（标准名）: {feature_keys}")

        # 映射到实际列名；派生特征直接使用列名本身
        used_feature_cols = [
            feature_cols_map.get(k, k) for k in feature_keys
        ]
        print(f"使用特征（实际列名）: {used_feature_cols}")

        X_all = df[used_feature_cols].values.astype(float)
        y_all = toc_all  # 原始 TOC（评估与绘图使用）

        r2_list = []
        rmse_list = []
        mae_list = []

        best_r2 = -1e9
        best_record = None  # 保存用于绘图与模型导出的信息

        for i in range(n_repeats):
            seed = i  # seeds = [0,1,2,...]
            print("-" * 60)
            print(f"[{setting_name}] 第 {i + 1}/{n_repeats} 次重复, seed = {seed}")

            # 基于索引的可复现划分, 两个 setting 共享同一划分逻辑
            idx = np.arange(n_samples_total)
            idx_train_val, idx_test = train_test_split(
                idx, test_size=test_size, random_state=seed
            )
            # 相对于 train_val 部分的验证集比例
            val_rel = val_size / (1.0 - test_size)
            idx_train, idx_val = train_test_split(
                idx_train_val, test_size=val_rel, random_state=seed
            )

            X_train = X_all[idx_train]
            y_train = y_all[idx_train]
            X_val = X_all[idx_val]
            y_val = y_all[idx_val]
            X_test = X_all[idx_test]
            y_test = y_all[idx_test]

            # log-space 训练标签
            y_train_log = y_all_log[idx_train]
            y_val_log = y_all_log[idx_val]
            y_test_log = y_all_log[idx_test]  # 仅用于检查，如有需要

            if use_xgb_effective:
                XGBRegressor = xgb_module.XGBRegressor  # type: ignore[attr-defined]
                model = XGBRegressor(
                    n_estimators=5000,
                    learning_rate=0.02,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=1,
                    objective="reg:squarederror",
                    tree_method="hist",
                    random_state=seed,
                )
                # 注意: 当前环境中的 xgboost sklearn 接口不支持 early_stopping_rounds / eval_metric
                # 这里退化为固定 n_estimators 训练，以保持兼容性。
                model.fit(X_train, y_train_log)
                best_iteration = getattr(model, "best_iteration", None)
                if best_iteration is None:
                    # 某些版本字段名不同
                    best_iteration = getattr(model, "best_ntree_limit", None)
            else:
                model = HistGradientBoostingRegressor(
                    max_depth=4,
                    learning_rate=0.1,
                    max_iter=500,
                    l2_regularization=1.0,
                    min_samples_leaf=20,
                    random_state=seed,
                )
                model.fit(X_train, y_train_log)
                best_iteration = None

            # 评估：从 log-space 预测还原到 TOC 空间，并在 TOC 空间计算指标
            y_val_pred_log = model.predict(X_val)
            y_test_pred_log = model.predict(X_test)
            y_val_pred = np.expm1(y_val_pred_log)
            y_test_pred = np.expm1(y_test_pred_log)
            # 截断到非负
            y_val_pred = np.clip(y_val_pred, 0.0, None)
            y_test_pred = np.clip(y_test_pred, 0.0, None)

            # Stage 2: Refinement MLP（仅 Full 且 use_refinement 时，L_range + L_mono）
            if (
                setting_name == "Full"
                and use_refinement
                and TORCH_AVAILABLE
            ):
                y_base_train = np.expm1(model.predict(X_train))
                y_base_train = np.clip(y_base_train, 0.0, None)
                toc_max = max(float(np.max(y_train)) * 1.1, 1.0)
                if toc_max < 5.0:
                    toc_max = 15.0  # 陆相页岩 TOC 上界
                mono_idx, mono_sign = get_monotonic_indices_for_full(feature_keys)
                try:
                    mlp = train_refinement_mlp_cn(
                        X_train, y_train, y_base_train,
                        toc_max=toc_max,
                        monotonic_indices=mono_idx if mono_idx else [],
                        monotonic_signs=mono_sign if mono_sign else [],
                        seed=seed,
                    )
                    y_test_refined = predict_refinement_mlp(mlp, X_test, y_test_pred)
                    y_test_refined = np.clip(y_test_refined, 0.0, None)
                    y_val_refined = predict_refinement_mlp(mlp, X_val, y_val_pred)
                    y_val_refined = np.clip(y_val_refined, 0.0, None)
                    # 仅当 refinement 提升 R2 时采用，否则保留 base
                    r2_base_val = r2_score(y_val, y_val_pred)
                    r2_refined_val = r2_score(y_val, y_val_refined)
                    if r2_refined_val >= r2_base_val:
                        y_test_pred = y_test_refined
                        y_val_pred = y_val_refined
                except Exception as e:  # noqa: BLE001
                    print(f"  [refinement] 跳过 seed={seed}: {e}")

            val_metrics = compute_metrics(y_val, y_val_pred)
            test_metrics = compute_metrics(y_test, y_test_pred)

            print(
                f"[{setting_name}] seed={seed} | "
                f"VAL -> R2={val_metrics['r2']:.4f}, "
                f"RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}"
            )
            print(
                f"[{setting_name}] seed={seed} | "
                f"TEST -> R2={test_metrics['r2']:.4f}, "
                f"RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}"
            )

            r2_list.append(test_metrics["r2"])
            rmse_list.append(test_metrics["rmse"])
            mae_list.append(test_metrics["mae"])

            # 保存单次日志
            run_log = {
                "setting": setting_name,
                "seed": seed,
                "use_xgb": bool(use_xgb_effective),
                "best_iteration": best_iteration,
                "n_samples_total": int(n_samples_total),
                "n_train": int(len(idx_train)),
                "n_val": int(len(idx_val)),
                "n_test": int(len(idx_test)),
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            log_path = logs_dir / f"run_{setting_name}_seed{seed}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(run_log, f, indent=2, ensure_ascii=False)
            print(f"保存日志: {log_path}")

            # 追加记录：每 seed 完成后保存 preds（供后续 cache 聚合用，不改变训练）
            pred_df = pd.DataFrame({"index": idx_test, "y_true": y_test, "y_pred": y_test_pred})
            pred_df.to_csv(preds_dir / f"predsTEST_{setting_name}_seed{seed}.csv", index=False, encoding="utf-8-sig")

            # 更新最佳模型 (按 TEST R2)
            if test_metrics["r2"] > best_r2:
                best_r2 = test_metrics["r2"]
                best_record = {
                    "model": model,
                    "setting": setting_name,
                    "seed": seed,
                    "test_metrics": test_metrics,
                    "X_test": X_test,
                    "y_test": y_test,
                    "y_test_pred": y_test_pred,
                    "feature_names": used_feature_cols,
                }

        # setting 的统计
        r2_arr = np.asarray(r2_list, dtype=float)
        rmse_arr = np.asarray(rmse_list, dtype=float)
        mae_arr = np.asarray(mae_list, dtype=float)

        setting_summary = {
            "setting": setting_name,
            "r2_mean": float(r2_arr.mean()),
            "r2_std": float(r2_arr.std(ddof=0)),
            "rmse_mean": float(rmse_arr.mean()),
            "rmse_std": float(rmse_arr.std(ddof=0)),
            "mae_mean": float(mae_arr.mean()),
            "mae_std": float(mae_arr.std(ddof=0)),
            "n_samples": int(n_samples_total),
        }
        results_summary[setting_name] = setting_summary
        best_models[setting_name] = best_record
        all_r2[setting_name] = r2_list.copy()

        print("=" * 60)
        print(
            f"{setting_name} - TEST 汇总: "
            f"R2 = {setting_summary['r2_mean']:.4f} ± {setting_summary['r2_std']:.4f}, "
            f"RMSE = {setting_summary['rmse_mean']:.4f} ± {setting_summary['rmse_std']:.4f}, "
            f"MAE = {setting_summary['mae_mean']:.4f} ± {setting_summary['mae_std']:.4f}"
        )

    # 固化聚合：delta_r2_by_seed.csv（训练结束后输出，不改变训练逻辑）
    if "NoRes" in all_r2 and "Full" in all_r2:
        seeds_arr = list(range(n_repeats))
        r2_base_arr = all_r2["NoRes"]
        r2_full_arr = all_r2["Full"]
        delta_arr = [f - b for f, b in zip(r2_full_arr, r2_base_arr)]
        df_delta = pd.DataFrame({
            "seed": seeds_arr,
            "r2_base": r2_base_arr,
            "r2_full": r2_full_arr,
            "delta_r2": delta_arr,
        })
        df_delta.to_csv(artifacts_metrics_dir / "delta_r2_by_seed.csv", index=False, encoding="utf-8-sig")
        print(f"固化: {artifacts_metrics_dir / 'delta_r2_by_seed.csv'}")

    # 保存指标汇总表
    summary_rows = [
        [
            v["setting"],
            v["r2_mean"],
            v["r2_std"],
            v["rmse_mean"],
            v["rmse_std"],
            v["mae_mean"],
            v["mae_std"],
            v["n_samples"],
        ]
        for v in results_summary.values()
    ]
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "setting",
            "r2_mean",
            "r2_std",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "n_samples",
        ],
    )
    summary_path = tables_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"汇总结果已保存: {summary_path}")

    # 为每个 setting 生成基于最佳模型的图表 & 保存模型
    for setting_name, record in best_models.items():
        if record is None:
            print(f"警告: setting={setting_name} 未找到可用模型，跳过绘图与保存。")
            continue
        model = record["model"]
        y_test = record["y_test"]
        y_test_pred = record["y_test_pred"]
        feature_names = record["feature_names"]
        test_metrics = record["test_metrics"]

        scatter_path = figs_dir / f"scatter_{setting_name}.png"
        residual_path = figs_dir / f"residual_hist_{setting_name}.png"
        feat_imp_path = figs_dir / f"feat_imp_{setting_name}.png"

        plot_scatter(
            y_test,
            y_test_pred,
            setting_name,
            scatter_path,
            metrics=test_metrics,
        )
        plot_residual_hist(y_test, y_test_pred, setting_name, residual_path)
        plot_feature_importance(model, feature_names, setting_name, feat_imp_path)

        # 保存最佳模型
        model_path = models_dir / f"best_{setting_name}.pkl"
        joblib.dump(model, model_path)
        print(f"保存最佳模型 ({setting_name}, seed={record['seed']}): {model_path}")

    return results_summary


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data" / "energies-2133761-supplementary.xlsx"
    default_out = project_root / "outputs"

    parser = argparse.ArgumentParser(
        description="Train TOC regression models on Energies dataset (Full vs NoRes)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(default_data),
        help="Excel 数据路径（包含 sheet 'Training and testing data'）。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(default_out),
        help="输出目录（将自动创建 logs/figures/models/tables 等子目录）。",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=300,
        help="不同随机种子重复次数（默认 100，对应 seeds=0..4）。",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="测试集比例（相对于总样本，默认 0.15）。",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="验证集比例（相对于总样本，默认 0.15）。",
    )
    parser.add_argument(
        "--use_xgb",
        type=str2bool,
        default=True,
        help="是否首选使用 XGBoost（默认 True，若环境无 xgboost 会自动回退）。",
    )
    parser.add_argument(
        "--use_refinement",
        type=str2bool,
        default=True,
        help="Full 阶段是否使用 Refinement MLP（L_range + L_mono 约束，默认 True）。",
    )
    parser.add_argument(
        "--no_refinement",
        action="store_true",
        help="禁用 Refinement MLP（等价于 --use_refinement False）。",
    )

    args = parser.parse_args()
    if getattr(args, "no_refinement", False):
        args.use_refinement = False

    data_path = Path(args.data_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("运行配置：")
    print(f"  data_path = {data_path}")
    print(f"  out_dir   = {out_dir}")
    print(f"  n_repeats = {args.n_repeats}")
    print(f"  test_size = {args.test_size}")
    print(f"  val_size  = {args.val_size}")
    print(f"  use_xgb   = {args.use_xgb}")
    print(f"  use_refinement = {args.use_refinement}")
    print("=" * 60)

    try:
        cleaned_path = out_dir / "cleaned_data.csv"
        df_clean, feature_cols_map, label_col = load_and_clean_data(
            data_path, cleaned_path
        )
        results_summary = run_experiment(
            df=df_clean,
            feature_cols_map=feature_cols_map,
            label_col=label_col,
            out_dir=out_dir,
            n_repeats=args.n_repeats,
            test_size=args.test_size,
            val_size=args.val_size,
            use_xgb_flag=args.use_xgb,
            use_refinement=args.use_refinement,
        )
        # 固化聚合：训练结束后写入 AllPlot/data 的 npz + meta.json（不改变训练逻辑）
        _write_cache_for_allplot(project_root, out_dir, data_path)
    except Exception as e:  # noqa: BLE001
        print("=" * 60)
        print("训练过程中发生错误：")
        print(repr(e))
        print("请检查数据路径、列名和依赖库是否正确。")
        print("=" * 60)
        sys.exit(1)

    # 结论输出：Full vs NoRes 的 R2 差异
    if "Full" in results_summary and "NoRes" in results_summary:
        full_r2 = results_summary["Full"]["r2_mean"]
        nores_r2 = results_summary["NoRes"]["r2_mean"]
        diff = full_r2 - nores_r2
        print("=" * 60)
        print(
            f"Full vs NoRes 的 R2 差异 = {diff:.4f} (Full: {full_r2:.4f}, NoRes: {nores_r2:.4f})"
        )
        print("=" * 60)
    else:
        print("警告: 无法计算 Full vs NoRes 的 R2 差异，因为某个 setting 缺失结果。")


if __name__ == "__main__":
    main()

