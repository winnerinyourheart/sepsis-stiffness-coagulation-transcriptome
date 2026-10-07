#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_gse185263_validate.py — GSE185263 (RNA-seq) 外部验证

RNA-seq 队列（Baghela 2022, 392 脓毒症全血），有 in-hospital mortality + SOFA。
这是与 GSE65682(微阵列) 完全独立的平台/队列，是最强的外部验证。

验证：
  1. 刚度 × 凝血 共变方向（Spearman）
  2. non-survivor vs survivor 的分数方向
  3. mHLA-DR 方向（免疫麻痹叙事）

方法：raw counts -> log2(CPM+1) -> 探针/基因映射(ensembl) -> ssGSEA 打分 -> 队列内 z-score
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


VAL_DIR = r"" + DATA_DIR + "/validation"
DATA_DIR = r"" + EXT_DIR + "/sepsis_bioinfo/data"
MR_DIR   = r"" + EXT_DIR + "/lactylation_coagulation_mr"
OUT_DIR  = r"" + RESULT_DIR + ""

def ensembl_to_symbol():
    """ensembl id -> gene symbol（用已有 GSE65682 注释里的 Gene Symbol + Ensembl 列）"""
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["Gene Symbol", "Ensembl"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol", "Ensembl"])
    m = {}
    for _, r in annot.iterrows():
        for e in str(r["Ensembl"]).split("///"):
            e = e.strip()
            if e.startswith("ENSG"):
                m[e] = r["Gene Symbol"]
    return m

def load_counts():
    counts = pd.read_csv(f"{VAL_DIR}/GSE185263_raw_counts.csv.gz", compression="gzip", index_col=0)
    # 转 CPM + log2
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    expr = np.log2(cpm + 1)
    # ensembl -> symbol
    sym = ensembl_to_symbol()
    expr = expr.rename(index=lambda x: sym.get(x, x))
    # 去重（多 ensembl 映射同一 symbol 取均值）
    expr = expr[~expr.index.str.startswith("ENSG")]
    expr = expr.groupby(expr.index).mean()
    return expr

def ssgsea(expr, genes):
    avail = [g for g in genes if g in expr.index]
    glist = expr.index.tolist(); n = len(glist)
    out = {}
    for s in expr.columns:
        vals = expr[s].values
        order = np.argsort(-vals)
        ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
        from scipy.stats import norm
        w = np.abs(norm.ppf(ranks/(n+1)))
        idx = [glist.index(g) for g in avail]
        in_set = np.zeros(n, bool); in_set[idx] = True
        out[s] = w[in_set].sum()/w.sum()
    return pd.Series(out), avail

def main():
    print("[1] 加载 raw counts -> log2(CPM+1) ...")
    expr = load_counts()
    print(f"    基因 x 样本 = {expr.shape}")

    print("[2] 加载 metadata ...")
    meta = pd.read_csv(f"{VAL_DIR}/gse185263_metadata.csv")
    meta = meta.set_index("title")  # title = sepcol 编号，与 counts 列名一致
    print(f"    metadata 样本数 = {len(meta)}")
    print("    mortality 分布:")
    print(meta["mortality"].value_counts(dropna=False))

    # 基因集
    stiff_df = pd.read_csv(f"{DATA_DIR}/../geneset/matrix_stiffness_genes.tsv", sep="\t")
    stiffness_genes = stiff_df["gene"].dropna().astype(str).str.strip().unique().tolist()
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]

    print("[3] ssGSEA 打分 ...")
    st, st_avail = ssgsea(expr, stiffness_genes)
    co, co_avail = ssgsea(expr, coag_genes)
    print(f"    刚度集可测 {len(st_avail)}/537, 凝血臂可测 {len(co_avail)}/42")

    # mHLA-DR
    mhla_genes = ["HLA-DRA","HLA-DRB1","HLA-DQA1"]
    mhla_avail = [g for g in mhla_genes if g in expr.index]
    mhla_sub = expr.loc[mhla_avail]
    mhla = mhla_sub.sub(mhla_sub.mean(axis=1),axis=0).div(mhla_sub.std(axis=1),axis=0).mean(axis=0)

    val = pd.DataFrame({"stiff": st, "coag": co, "mhla": mhla})
    # 对齐 metadata（counts 列名 = sepcol = title）
    val["mortality"] = val.index.map(lambda s: meta.loc[s,"mortality"] if s in meta.index else np.nan)
    for c in ["stiff","coag","mhla"]:
        val[f"{c}_z"] = (val[c]-val[c].mean())/val[c].std()

    print("\n========== GSE185263 验证结果 ==========")
    # 1. 共变
    r,p = stats.spearmanr(val["stiff"], val["coag"])
    print(f"[1] 刚度×凝血共变: r={r:+.3f}, p={p:.2e}  {'✅正相关' if r>0 else '⚠️负相关'}")

    # 2. 方向
    ns = val[val["mortality"].astype(str).str.lower().str.contains("died|non|dead", na=False)]
    sv = val[val["mortality"].astype(str).str.lower().str.contains("surviv", na=False)]
    print(f"\n[2] non-survivor(n={len(ns)}) vs survivor(n={len(sv)}):")
    for c in ["stiff_z","coag_z","mhla_z"]:
        if len(ns)>2 and len(sv)>2:
            U,p = stats.mannwhitneyu(ns[c], sv[c])
            d = "NS更低(同向✅)" if ns[c].mean()<sv[c].mean() else "NS更高(反向⚠️)"
            print(f"    {c:8s} NS={ns[c].mean():+.3f} vs SV={sv[c].mean():+.3f}, p={p:.4f}, {d}")

    # 3. mHLA-DR 关联
    print(f"\n[3] mHLA-DR 关联:")
    for c in ["stiff","coag"]:
        r,p = stats.spearmanr(val["mhla"], val[c])
        print(f"    mHLA-DR × {c}: r={r:+.3f}, p={p:.2e}")

    val.to_csv(f"{OUT_DIR}/validation_gse185263.csv")

if __name__ == "__main__":
    main()
