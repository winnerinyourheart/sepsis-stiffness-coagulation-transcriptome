#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_immunoparalysis.py — 支撑「刚度-凝血协同下调标记免疫耗竭」叙事

1. Mars1 vs 其他内型的 刚度/凝血分数差异（验证低分=Mars1）
2. 刚度/凝血分数 vs 免疫耗竭标志(HLA-DRA下调+PDCD1/CTLA4/LAG3等) 的关系
3. 刚度/凝血分数 vs 免疫激活标志(CD69/CD38/GZMB/PRF1等) 的关系
4. 关键：mHLA-DR（单核HLA-DRA）作为免疫麻痹的经典标志，是否与分数正相关

预期（若叙事成立）：
  - 低分样本 = Mars1 富集 + HLA-DRA 低（单核免疫麻痹）+ 耗竭标志高
  - 分数与 HLA-DRA 正相关，与耗竭标志负相关
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

EXHAUSTION = ['PDCD1','CTLA4','LAG3','HAVCR2','TIGIT','CD244','ENTPD1','LILRB1','EOMES']
ACTIVATION = ['HLA-DRA','HLA-DRB1','HLA-DQA1','CD69','CD38','IL2RA','GZMB','PRF1','TBX21','CCR7','SELL']

def load_gene_expr():
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    expr = expr.loc[expr.index.intersection(annot["ID"])]
    expr.index = expr.index.map(id2sym)
    return expr.groupby(expr.index).mean()

def main():
    expr = load_gene_expr()
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()

    print("========== 1. Mars1 vs 其他内型 分数差异 ==========")
    mars1 = seps[seps["endotype_class"]=="Mars1"]
    others = seps[seps["endotype_class"]!="Mars1"]
    for col in ["stiffness_537_z", "coag_42_z"]:
        U,p = stats.mannwhitneyu(mars1[col], others[col])
        print(f"  {col}: Mars1={mars1[col].mean():+.3f} vs 其他={others[col].mean():+.3f}, p={p:.2e}")

    print("\n========== 2. 分数 vs 免疫耗竭/激活标志（Spearman） ==========")
    rows = []
    for col in ["stiffness_537_z", "coag_42_z"]:
        for g in EXHAUSTION + ACTIVATION:
            if g in expr.index:
                r,p = stats.spearmanr(seps[col], expr.loc[g, seps.index])
                rows.append({"score": col, "gene": g, "r": r, "p": p})
    df = pd.DataFrame(rows)
    # 分类标记
    df["cat"] = df["gene"].apply(lambda g: "exhaustion" if g in EXHAUSTION else "activation")
    df.to_csv(f"{OUT_DIR}/immunoparalysis_corr.csv", index=False)

    for col in ["stiffness_537_z", "coag_42_z"]:
        print(f"\n  --- {col} ---")
        ex = df[(df["score"]==col)&(df["cat"]=="exhaustion")]
        ac = df[(df["score"]==col)&(df["cat"]=="activation")]
        print(f"  耗竭标志: 均值 r = {ex['r'].mean():+.3f} (范围 {ex['r'].min():+.3f}~{ex['r'].max():+.3f})")
        print(f"  激活标志: 均值 r = {ac['r'].mean():+.3f} (范围 {ac['r'].min():+.3f}~{ac['r'].max():+.3f})")
        # 关键基因
        for g in ["HLA-DRA","PDCD1","CTLA4","LAG3","HAVCR2","CD69","GZMB"]:
            row = df[(df["score"]==col)&(df["gene"]==g)]
            if len(row):
                r,p = row["r"].iloc[0], row["p"].iloc[0]
                print(f"    {g:8s} r={r:+.3f} p={p:.2e}")

    print("\n========== 3. 关键：HLA-DRA（单核免疫麻痹标志 mHLA-DR 代理）==========")
    # mHLA-DR 低 = 免疫麻痹，是脓毒症公认的免疫抑制标志
    hla = expr.loc["HLA-DRA", seps.index]
    for col in ["stiffness_537_z", "coag_42_z"]:
        r,p = stats.spearmanr(seps[col], hla)
        print(f"  {col} vs HLA-DRA: r={r:+.3f}, p={p:.2e}")

    # HLA-DRA 在死亡 vs 存活
    dead = seps[seps["mortality_28d"]==1].index; alive = seps[seps["mortality_28d"]==0].index
    print(f"\n  HLA-DRA 死亡={hla[dead].mean():.3f} vs 存活={hla[alive].mean():.3f}")
    U,p = stats.mannwhitneyu(hla[dead], hla[alive])
    print(f"  (死亡样本 HLA-DRA 更低? p={p:.4f})")

    # 低分样本是否 Mars1 富集 + HLA-DRA 低
    print("\n========== 4. 低分样本的特征（低刚度低凝血 LL 象限） ==========")
    st_med = seps["stiffness_537_z"].median(); co_med = seps["coag_42_z"].median()
    ll = seps[(seps["stiffness_537_z"]<st_med)&(seps["coag_42_z"]<co_med)]
    hh = seps[(seps["stiffness_537_z"]>=st_med)&(seps["coag_42_z"]>=co_med)]
    print(f"  LL象限 Mars1 占比: {(ll['endotype_class']=='Mars1').mean():.1%} vs HH象限 {(hh['endotype_class']=='Mars1').mean():.1%}")
    print(f"  LL象限 HLA-DRA 均值: {hla[ll.index].mean():.3f} vs HH象限 {hla[hh.index].mean():.3f}")
    print(f"  LL象限 死亡率: {(ll['mortality_28d']==1).mean():.1%} vs HH象限 {(hh['mortality_28d']==1).mean():.1%}")

if __name__ == "__main__":
    main()
