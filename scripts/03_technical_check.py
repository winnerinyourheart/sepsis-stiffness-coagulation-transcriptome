#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_technical_check.py — 头号技术伪影排查

问题：死亡样本是否因"全局表达下移 / RNA 质量差 / 白细胞低"而系统性压低所有通路分数，
     制造"低分=死亡"的假象？

检验：
  1. 全局表达中位数：死亡 vs 存活
  2. 管家基因表达（GAPDH, ACTB, B2M, RPLP0 等）：死亡 vs 存活
  3. 表达"检出率"（高于某阈值的基因数）作为 RNA 完整性代理
  4. 白细胞相关基因（CD3D/CD19/CD14/CD8A 等）作为细胞组成代理
"""
import numpy as np
import pandas as pd
from scipy import stats
import os as _os
# ---------------------------------------------------------------------------
# Path configuration.
# This script was written against a local working tree. Set the following
# environment variables before running:
#   STS_DATA_DIR    - directory holding the raw/derived expression inputs
#   STS_RESULT_DIR  - directory for the output result tables
#   STS_EXT_DIR     - (optional) directory holding shared external resources
#                     (deCODE pQTL clean tables, GSE65682 stiffness matrix)
# ---------------------------------------------------------------------------
DATA_DIR = _os.environ.get("STS_DATA_DIR", "data").rstrip("/\\")
RESULT_DIR = _os.environ.get("STS_RESULT_DIR", "results").rstrip("/\\")
EXT_DIR = _os.environ.get("STS_EXT_DIR", "external").rstrip("/\\")
_os.makedirs(RESULT_DIR, exist_ok=True)
# ---------------------------------------------------------------------------


DATA_DIR = r"" + EXT_DIR + "/sepsis_bioinfo/data"
OUT_DIR  = r"" + RESULT_DIR + ""

HOUSEKEEPING = ["GAPDH", "ACTB", "B2M", "RPLP0", "GUSB", "HPRT1", "TBP", "PGK1"]
LEUKOCYTE = ["PTPRC", "CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1", "CD14",
             "FCGR3A", "NCAM1", "CD68", "LYZ"]

def load_gene_expr():
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    expr = expr.groupby(expr.index).mean()
    return expr

def main():
    expr = load_gene_expr()
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()
    dead = seps[seps["mortality_28d"] == 1].index
    alive = seps[seps["mortality_28d"] == 0].index

    # 只保留脓毒症样本的列
    expr_seps = expr[seps.index]

    print("=== 1. 全局表达中位数（每样本）死亡 vs 存活 ===")
    med = expr_seps.median(axis=0)
    dm = med[dead].mean(); am = med[alive].mean()
    U, p = stats.mannwhitneyu(med[dead], med[alive], alternative="two-sided")
    print(f"  死亡样本全局中位数均值 = {dm:.4f}")
    print(f"  存活样本全局中位数均值 = {am:.4f}")
    print(f"  差异 {dm-am:+.4f}, Mann-Whitney p = {p:.4f}")
    print(f"  -> {'⚠️ 死亡样本全局表达显著更低（伪影风险！）' if p<0.05 else '✅ 无显著全局表达差异'}")

    print("\n=== 2. 管家基因表达 死亡 vs 存活 ===")
    for g in HOUSEKEEPING:
        if g in expr_seps.index:
            d = expr_seps.loc[g, dead].mean(); a = expr_seps.loc[g, alive].mean()
            U, p = stats.mannwhitneyu(expr_seps.loc[g, dead], expr_seps.loc[g, alive])
            flag = "⚠️" if p < 0.05 else "  "
            print(f"  {g:8s} 死={d:7.3f} 活={a:7.3f} 差={d-a:+7.3f} p={p:.4f} {flag}")

    print("\n=== 3. 表达检出率（>7 的基因数，log2 尺度）===")
    thr = 7.0
    n_det = (expr_seps > thr).sum(axis=0)
    d = n_det[dead].mean(); a = n_det[alive].mean()
    U, p = stats.mannwhitneyu(n_det[dead], n_det[alive])
    print(f"  死亡检出基因数均值 = {d:.1f}, 存活 = {a:.1f}, p = {p:.4f}")
    print(f"  -> {'⚠️ 死亡样本检出基因显著更少（质量差）' if p<0.05 else '✅ 检出率无显著差异'}")

    print("\n=== 4. 白细胞相关基因（细胞组成代理）===")
    for g in LEUKOCYTE:
        if g in expr_seps.index:
            d = expr_seps.loc[g, dead].mean(); a = expr_seps.loc[g, alive].mean()
            U, p = stats.mannwhitneyu(expr_seps.loc[g, dead], expr_seps.loc[g, alive])
            flag = "⚠️" if p < 0.05 else "  "
            print(f"  {g:8s} 死={d:7.3f} 活={a:7.3f} 差={d-a:+7.3f} p={p:.4f} {flag}")

if __name__ == "__main__":
    main()
