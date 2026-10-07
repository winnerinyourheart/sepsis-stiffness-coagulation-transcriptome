#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_deg_intersection_enrichment.py — 差异基因 + 交集 + 富集

生信文章主体模块 1：
  1. 刚度基因集(537) ∩ 凝血臂(42) → 交集基因（桥接基因）
  2. 交集基因在 GSE65682 脓毒症中的表达方向（死亡 vs 存活）
  3. 共变驱动基因：与刚度分数、凝血分数都相关的基因（在交集基因里）
  4. 交集基因的 GO/KEGG 富集（用 gseapy 或手写超几何）

输出：
  results/intersection_genes.csv
  results/intersection_direction.csv
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
MR_DIR   = r"" + EXT_DIR + "/lactylation_coagulation_mr"
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

    # 基因集
    stiff_df = pd.read_csv(f"{DATA_DIR}/../geneset/matrix_stiffness_genes.tsv", sep="\t")
    stiffness_genes = stiff_df["gene"].dropna().astype(str).str.strip().unique().tolist()
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]

    # 交集（在矩阵中可测的）
    stiff_avail = set(stiffness_genes) & set(expr.index)
    coag_avail = set(coag_genes) & set(expr.index)
    inter = sorted(stiff_avail & coag_avail)
    print(f"刚度集可测 {len(stiff_avail)}, 凝血臂可测 {len(coag_avail)}")
    print(f"交集基因 ({len(inter)}): {inter}")

    # 交集基因的表达方向（死亡 vs 存活）
    dead = seps[seps["mortality_28d"]==1].index
    alive = seps[seps["mortality_28d"]==0].index
    rows = []
    for g in inter:
        d = expr.loc[g, dead].values; a = expr.loc[g, alive].values
        l2fc = np.log2(np.mean(2**d) / np.mean(2**a)) if d.size and a.size else np.nan
        U,p = stats.mannwhitneyu(d, a)
        # 与两个分数的相关
        r_st, p_st = stats.spearmanr(expr.loc[g, seps.index], seps["stiffness_537_z"])
        r_co, p_co = stats.spearmanr(expr.loc[g, seps.index], seps["coag_42_z"])
        rows.append({"gene": g, "mean_dead": d.mean(), "mean_alive": a.mean(),
                     "log2FC_dead_vs_alive": l2fc, "p_mw": p,
                     "r_stiff": r_st, "p_stiff": p_st, "r_coag": r_co, "p_coag": p_co})
    inter_df = pd.DataFrame(rows)
    inter_df.to_csv(f"{OUT_DIR}/intersection_genes.csv", index=False)
    print("\n=== 交集基因 死亡 vs 存活 方向 ===")
    print(inter_df[["gene","log2FC_dead_vs_alive","p_mw"]].round(4).to_string(index=False))

    # 保存交集基因列表供后续用
    with open(f"{OUT_DIR}/intersection_gene_list.txt", "w") as f:
        f.write("\n".join(inter))

if __name__ == "__main__":
    main()
