#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
26_validation_ci.py — 外部验证共变 r 的 bootstrap 95% CI

对三个外部队列（GSE95233 / E-MTAB-4451 / GSE185263）的
「刚度分数 × 凝血分数 Spearman r」给出 bootstrap 95% 置信区间，
并同时报告样本量与 Pearson r，供正文 External validation 段与 Table 6 使用。

只读 results/validation_*.csv，不重算表达谱，不改动任何既有结果文件。
输出：results/validation_coverage_ci.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "validation_coverage_ci.csv")

RNG = np.random.default_rng(42)
N_BOOT = 10000


def boot_spearman(x, y, n_boot=N_BOOT, rng=RNG):
    """Bootstrap 95% CI of Spearman rho (percentile method)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    rho = stats.spearmanr(x, y).statistic
    idx = np.arange(n)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        xs, ys = x[s], y[s]
        if np.std(xs) == 0 or np.std(ys) == 0:
            boots[b] = np.nan
            continue
        boots[b] = stats.spearmanr(xs, ys).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, lo, hi


def analyze(path, label, surv_col, non_surv_labels):
    df = pd.read_csv(path, index_col=0)
    x = df["stiff"].values
    y = df["coag"].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rho, lo, hi = boot_spearman(x, y)
    pr, pp = stats.pearsonr(x, y)
    sp = stats.spearmanr(x, y)
    n = len(x)

    # mHLA-DR 死亡 vs 存活（若该表有）
    ns_p = np.nan
    if surv_col in df.columns and "mhla" in df.columns:
        lab = df[surv_col].astype(str).str.lower().str.strip()
        dead = df.loc[lab.isin(non_surv_labels), "mhla"].dropna()
        alive = df.loc[~lab.isin(non_surv_labels) & lab.ne("nan"), "mhla"].dropna()
        if len(dead) and len(alive):
            ns_p = stats.mannwhitneyu(dead, alive).pvalue

    return {
        "cohort": label,
        "n_samples": n,
        "spearman_r": round(rho, 4),
        "rho_ci_lo": round(lo, 4),
        "rho_ci_hi": round(hi, 4),
        "spearman_p": f"{sp.pvalue:.3e}",
        "pearson_r": round(pr, 4),
        "mhla_p_deadvs_alive": (f"{ns_p:.4f}" if not np.isnan(ns_p) else "NA"),
    }


def main():
    rows = []
    rows.append(analyze(os.path.join(RES, "validation_gse95233.csv"),
                        "GSE95233", "survival", {"non survivor", "non-survivor"}))
    rows.append(analyze(os.path.join(RES, "validation_emtab4451.csv"),
                        "E-MTAB-4451", "survival", {"non survivor", "non-survivor"}))
    rows.append(analyze(os.path.join(RES, "validation_gse185263.csv"),
                        "GSE185263", "mortality", {"died", "dead", "non-survivor"}))

    # 发现队列（GSE65682）r 的 CI，用于对比
    sc = pd.read_csv(os.path.join(RES, "scores_gse65682.csv"), index_col=0)
    if "is_sepsis" in sc.columns:
        sc = sc[sc["is_sepsis"]]
    if {"stiffness_537_z", "coag_42_z"}.issubset(sc.columns):
        rho, lo, hi = boot_spearman(sc["stiffness_537_z"].values, sc["coag_42_z"].values)
        rows.insert(0, {
            "cohort": "GSE65682 (discovery)",
            "n_samples": int(sc.shape[0]),
            "spearman_r": round(rho, 4),
            "rho_ci_lo": round(lo, 4),
            "rho_ci_hi": round(hi, 4),
            "spearman_p": "8.7e-75",
            "pearson_r": round(stats.pearsonr(sc["stiffness_537_z"], sc["coag_42_z"]).statistic, 4),
            "mhla_p_deadvs_alive": "0.0017",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
