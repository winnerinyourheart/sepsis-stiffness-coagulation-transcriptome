#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_covariation_network_enrichment.py — 共变驱动基因 + 富集

生信文章主体模块 2：
  1. 全基因组筛选"共变驱动基因"：与刚度分数和凝血分数都显著相关的基因
     （这是"刚度-凝血共变"的分子网络，比 6 个交集基因更全面）
  2. 共变驱动基因的 GO/KEGG 富集（gseapy enrichr）
  3. 正相关 vs 负相关基因分别富集

输出：
  results/covariation_driver_genes.csv
  results/enrichment_covariation_*.csv
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
import gseapy as gp
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

    st = seps["stiffness_537_z"].values
    co = seps["coag_42_z"].values

    print("[1] 全基因组筛选共变驱动基因（与刚度、凝血分数都显著相关）...")
    rows = []
    genes = expr.index.tolist()
    for g in genes:
        gv = expr.loc[g, seps.index].values
        r_st, p_st = stats.spearmanr(gv, st)
        r_co, p_co = stats.spearmanr(gv, co)
        rows.append({"gene": g, "r_stiff": r_st, "p_stiff": p_st,
                     "r_coag": r_co, "p_coag": p_co})
    df = pd.DataFrame(rows)
    # 多重校正（BH）
    from statsmodels.stats.multitest import multipletests
    df["p_stiff_bh"] = multipletests(df["p_stiff"], method="fdr_bh")[1]
    df["p_coag_bh"] = multipletests(df["p_coag"], method="fdr_bh")[1]

    # 共变驱动 = 两者都显著（BH<0.05）且同向
    both = df[(df["p_stiff_bh"]<0.05) & (df["p_coag_bh"]<0.05)].copy()
    both["direction"] = np.where(both["r_stiff"]>0, "up", "down")
    print(f"    共变驱动基因（双显著）: {len(both)} 个，其中上调 {sum(both['direction']=='up')}、下调 {sum(both['direction']=='down')}")
    both.to_csv(f"{OUT_DIR}/covariation_driver_genes.csv", index=False)

    # 保存基因列表
    up_genes = both[both["direction"]=="up"]["gene"].tolist()
    down_genes = both[both["direction"]=="down"]["gene"].tolist()
    print(f"    上调 {len(up_genes)} 个，下调 {len(down_genes)} 个")

    print("\n[2] 富集分析（上调共变基因）...")
    if len(up_genes) > 5:
        try:
            enr_up = gp.enrichr(gene_list=up_genes, gene_sets=["KEGG_2021_Human","GO_Biological_Process_2023","Reactome_2022"],
                                organism="human", outdir=None)
            res_up = enr_up.results
            res_up.to_csv(f"{OUT_DIR}/enrichment_covariation_up.csv", index=False)
            print("    KEGG 前10:")
            kegg = res_up[res_up["Gene_set"].str.contains("KEGG")].head(10)
            print(kegg[["Term","Overlap","Adjusted P-value"]].to_string(index=False))
        except Exception as e:
            print("    上调富集失败:", e)

    print("\n[3] 富集分析（下调共变基因）...")
    if len(down_genes) > 5:
        try:
            enr_dn = gp.enrichr(gene_list=down_genes, gene_sets=["KEGG_2021_Human","GO_Biological_Process_2023","Reactome_2022"],
                                organism="human", outdir=None)
            res_dn = enr_dn.results
            res_dn.to_csv(f"{OUT_DIR}/enrichment_covariation_down.csv", index=False)
            print("    KEGG 前10:")
            kegg = res_dn[res_dn["Gene_set"].str.contains("KEGG")].head(10)
            print(kegg[["Term","Overlap","Adjusted P-value"]].to_string(index=False))
        except Exception as e:
            print("    下调富集失败:", e)

if __name__ == "__main__":
    main()
