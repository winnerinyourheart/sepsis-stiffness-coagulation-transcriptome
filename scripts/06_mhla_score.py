#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_mhla_score.py — mHLA-DR 联合分数 + GSE95233 方向补强

1. 单核抗原呈递分数：HLA-DRA/DRB1/DQA1 三基因 z-score 均值（mHLA-DR 联合代理）
2. 验证：mHLA-DR 分数与 刚度/凝血分数 的关系（GSE65682）
3. 补强：GSE95233 里 mHLA-DR 分数 的 Non-Survivor vs Survivor 方向
   （若"免疫麻痹"叙事成立，Non-Survivor 的 mHLA-DR 分数应更低）
"""
import json
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

MHLA_GENES = ["HLA-DRA", "HLA-DRB1", "HLA-DQA1"]

def load_gene_expr(dataset):
    expr = pd.read_csv(f"{DATA_DIR}/{dataset}_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/{dataset}_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    expr = expr.loc[expr.index.intersection(annot["ID"])]
    expr.index = expr.index.map(id2sym)
    return expr.groupby(expr.index).mean()

def mhla_score(expr, samples):
    """单核抗原呈递联合分数：三基因 z-score 均值。"""
    avail = [g for g in MHLA_GENES if g in expr.index]
    sub = expr.loc[avail, samples]
    sub_z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1), axis=0)
    return sub_z.mean(axis=0), avail

def main():
    # ---------- GSE65682 ----------
    expr = load_gene_expr("GSE65682")
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()

    mhla, avail = mhla_score(expr, seps.index)
    print(f"mHLA-DR 联合分数可测基因: {avail}")
    seps["mhla_dr"] = mhla

    print("\n========== GSE65682：mHLA-DR 分数 vs 刚度/凝血 ==========")
    for col in ["stiffness_537_z", "coag_42_z"]:
        r,p = stats.spearmanr(seps["mhla_dr"], seps[col])
        print(f"  mHLA-DR × {col}: r={r:+.3f}, p={p:.2e}")

    # 死亡 vs 存活
    dead = seps[seps["mortality_28d"]==1]["mhla_dr"]; alive = seps[seps["mortality_28d"]==0]["mhla_dr"]
    U,p = stats.mannwhitneyu(dead, alive)
    print(f"  mHLA-DR 死亡={dead.mean():+.3f} vs 存活={alive.mean():+.3f}, p={p:.4f}")

    # Mars1 vs 其他
    mars1 = seps[seps["endotype_class"]=="Mars1"]["mhla_dr"]; others = seps[seps["endotype_class"]!="Mars1"]["mhla_dr"]
    U,p = stats.mannwhitneyu(mars1, others)
    print(f"  mHLA-DR Mars1={mars1.mean():+.3f} vs 其他={others.mean():+.3f}, p={p:.2e}")

    # LL vs HH 象限
    st_med = seps["stiffness_537_z"].median(); co_med = seps["coag_42_z"].median()
    ll = seps[(seps["stiffness_537_z"]<st_med)&(seps["coag_42_z"]<co_med)]
    hh = seps[(seps["stiffness_537_z"]>=st_med)&(seps["coag_42_z"]>=co_med)]
    print(f"  mHLA-DR LL象限={ll['mhla_dr'].mean():+.3f} vs HH象限={hh['mhla_dr'].mean():+.3f}")

    # 保存
    seps["mhla_dr"].to_csv(f"{OUT_DIR}/mhla_dr_gse65682.csv")

    # ---------- GSE95233 补强 ----------
    print("\n========== GSE95233：mHLA-DR 分数方向补强 ==========")
    expr2 = load_gene_expr("GSE95233")
    pheno2 = pd.read_csv(f"{DATA_DIR}/GSE95233_pheno.csv", low_memory=False)
    p2 = pheno2.set_index("Sample_geo_accession")

    mhla2, avail2 = mhla_score(expr2, expr2.columns)
    print(f"GSE95233 mHLA-DR 可测基因: {avail2}")
    val = pd.DataFrame({"mhla_dr": mhla2})
    val["survival"] = val.index.map(lambda s: p2.loc[s,"survival"] if s in p2.index else np.nan)

    ns = val[val["survival"]=="Non Survivor"]; sv = val[val["survival"]=="Survivor"]
    U,p = stats.mannwhitneyu(ns["mhla_dr"], sv["mhla_dr"])
    print(f"  mHLA-DR Non-Survivor={ns['mhla_dr'].mean():+.3f} vs Survivor={sv['mhla_dr'].mean():+.3f}, p={p:.4f}")
    print(f"  -> {'✅ 方向一致：重症/死亡 mHLA-DR 更低（免疫麻痹）' if ns['mhla_dr'].mean()<sv['mhla_dr'].mean() else '⚠️ 方向相反'}")

if __name__ == "__main__":
    main()
