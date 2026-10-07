#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_validation.py — 敏感性 + 外部验证

A. Consensus13 敏感性（GSE65682）：用已验证的 13 基因 LASSO 系数签名打分，
   检查与 537 集 ssGSEA 分数方向是否一致，及与死亡的关系。
B. GSE95233 外部验证：脓毒症休克队列复现「刚度×凝血共变方向」。
   注意：GSE95233 是 51 患者×多时点(配对)，不做独立生存分析，
   只验证：(1) 共变方向 (2) Non-Survivor vs Survivor 的分数方向。
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

def load_gene_expr(dataset):
    """dataset: 'GSE65682' 或 'GSE95233'。返回基因 x 样本。"""
    expr = pd.read_csv(f"{DATA_DIR}/{dataset}_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/{dataset}_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    expr = expr.groupby(expr.index).mean()
    return expr

def ssgsea(expr, genes):
    avail = [g for g in genes if g in expr.index]
    genes_list = expr.index.tolist()
    n = len(genes_list)
    out = {}
    for s in expr.columns:
        vals = expr[s].values
        order = np.argsort(-vals)
        ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
        from scipy.stats import norm
        w = np.abs(norm.ppf(ranks/(n+1)))
        idx = [genes_list.index(g) for g in avail]
        in_set = np.zeros(n, bool); in_set[idx] = True
        out[s] = w[in_set].sum() / w.sum()
    return pd.Series(out), avail

def main():
    print("========== A. Consensus13 敏感性（GSE65682） ==========")
    coef = pd.read_csv(f"{DATA_DIR}/../results/Consensus13_coef.csv")
    genes13 = coef["gene"].tolist()
    coef_map = dict(zip(coef["gene"], coef["coef"]))

    expr = load_gene_expr("GSE65682")
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()

    # Consensus13 加权签名分数（原始尺度，非 z）
    avail13 = [g for g in genes13 if g in expr.index]
    print(f"Consensus13 基因在矩阵中可测: {len(avail13)}/13 = {avail13}")
    # 逐基因 z-score 后加权求和（与 ssGSEA 不同，这里是签名打分）
    sub = expr.loc[avail13, seps.index]
    sub_z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1), axis=0)
    cons13_score = pd.Series(0.0, index=seps.index)
    for g in avail13:
        cons13_score += coef_map[g] * sub_z.loc[g]
    seps["cons13"] = cons13_score

    # 与 537 分数一致性
    r13, p13 = stats.spearmanr(seps["cons13"], seps["stiffness_537_z"])
    print(f"\nConsensus13 vs 537集 ssGSEA: Spearman r={r13:.4f}, p={p13:.3e}")
    # Consensus13 vs 凝血
    r13c, p13c = stats.spearmanr(seps["cons13"], seps["coag_42_z"])
    print(f"Consensus13 vs 凝血分数: Spearman r={r13c:.4f}, p={p13c:.3e}")
    # Consensus13 与死亡
    dead = seps[seps["mortality_28d"]==1]["cons13"]; alive = seps[seps["mortality_28d"]==0]["cons13"]
    U, p = stats.mannwhitneyu(dead, alive)
    print(f"Consensus13 死亡 vs 存活: 死均值={dead.mean():.3f}, 活均值={alive.mean():.3f}, p={p:.4f}")
    print(f"  -> {'死亡样本 Consensus13 更低（与537集同向，反直觉方向稳定）' if dead.mean()<alive.mean() else '死亡样本更高（方向与537集相反！）'}")

    print("\n========== B. GSE95233 外部验证 ==========")
    expr2 = load_gene_expr("GSE95233")
    pheno2 = pd.read_csv(f"{DATA_DIR}/GSE95233_pheno.csv", low_memory=False)
    pheno2_map = pheno2.set_index("Sample_geo_accession")

    stiff_df = pd.read_csv(f"{DATA_DIR}/../geneset/matrix_stiffness_genes.tsv", sep="\t")
    stiffness_genes = stiff_df["gene"].dropna().astype(str).str.strip().unique().tolist()
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]

    st_score, st_avail = ssgsea(expr2, stiffness_genes)
    co_score, co_avail = ssgsea(expr2, coag_genes)
    print(f"GSE95233 刚度集可测 {len(st_avail)}/537, 凝血臂可测 {len(co_avail)}/42")

    # 合并
    val = pd.DataFrame({"stiff": st_score, "coag": co_score})
    val["survival"] = val.index.map(lambda s: pheno2_map.loc[s,"survival"] if s in pheno2_map.index else np.nan)

    # 共变方向
    r_v, p_v = stats.spearmanr(val["stiff"], val["coag"])
    print(f"\nGSE95233 全样本 刚度×凝血 Spearman r={r_v:.4f}, p={p_v:.3e}")

    # Non-Survivor vs Survivor 分数方向
    ns = val[val["survival"]=="Non Survivor"]; sv = val[val["survival"]=="Survivor"]
    print(f"\nNon-Survivor (n={len(ns)}) vs Survivor (n={len(sv)}):")
    for col in ["stiff","coag"]:
        U,p = stats.mannwhitneyu(ns[col], sv[col])
        print(f"  {col}: NS均值={ns[col].mean():.4f}, SV均值={sv[col].mean():.4f}, p={p:.4f}, "
              f"{'NS更低(与GSE65682同向)' if ns[col].mean()<sv[col].mean() else 'NS更高(与GSE65682反向!)'}")

    # 配对患者：只看 D01（首时点，避免重复）
    d01 = val[val.index.map(lambda s: pheno2_map.loc[s,'time_point'] if s in pheno2_map.index else None)=="D01"]
    print(f"\n仅 D01 首时点 (n={len(d01)}):")
    ns1 = d01[d01["survival"]=="Non Survivor"]; sv1 = d01[d01["survival"]=="Survivor"]
    if len(ns1)>0 and len(sv1)>0:
        r_v1, p_v1 = stats.spearmanr(d01["stiff"], d01["coag"])
        print(f"  刚度×凝血 r={r_v1:.4f}, p={p_v1:.3e}")
        for col in ["stiff","coag"]:
            U,p = stats.mannwhitneyu(ns1[col], sv1[col])
            print(f"  {col}: NS均值={ns1[col].mean():.4f}, SV均值={sv1[col].mean():.4f}, p={p:.4f}, "
                  f"{'NS更低(同向)' if ns1[col].mean()<sv1[col].mean() else 'NS更高(反向)'}")

    # 保存
    val.to_csv(f"{OUT_DIR}/validation_gse95233.csv")

if __name__ == "__main__":
    main()
