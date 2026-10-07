#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_emtab4451_validation.py — 第二独立外部验证（E-MTAB-4451, Davenport GAinS）

验证两个方向（跨平台，队列内标准化，沿用前项目口径 B/C 的合法做法）：
  1. 刚度 × 凝血 共变方向（Spearman，是否仍正相关）
  2. non-survivor vs survivor 的分数方向（是否重症/死亡更低 = 免疫麻痹叙事）

队列：英国 ICU 严重脓毒症（CAP），106 例白细胞，survivor 54 / non-survivor 52。
平台：Illumina HumanHT-12 V4（跨平台，与前两个 Affymetrix 队列不同）。
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

def load_expr():
    expr = pd.read_csv(f"{DATA_DIR}/Davenport_sepsis_Feb2016_normalised_106.txt",
                       sep="\t", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GPL10558_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Symbol"])
    annot["Symbol"] = annot["Symbol"].astype(str).str.strip()
    # 去掉 non-gene 符号
    annot = annot[~annot["Symbol"].str.contains("phage_lambda|thrB|low|genome|:|Biotin|Random")]
    id2sym = annot.set_index("ID")["Symbol"].to_dict()
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    expr = expr.groupby(expr.index).mean()
    return expr

def load_survival():
    sdrf = pd.read_csv(f"{DATA_DIR}/E-MTAB-4451.sdrf.txt", sep="\t", low_memory=False)
    surv = sdrf[["Source Name", "Characteristics[28 day survival]"]].copy()
    surv.columns = ["sample", "survival"]
    surv = surv.drop_duplicates(subset="sample").set_index("sample")
    return surv["survival"]

def ssgsea(expr, genes):
    avail = [g for g in genes if g in expr.index]
    glist = expr.index.tolist()
    n = len(glist)
    out = {}
    for s in expr.columns:
        vals = expr[s].values
        order = np.argsort(-vals)
        ranks = np.empty(n); ranks[order] = np.arange(1, n+1)
        from scipy.stats import norm
        w = np.abs(norm.ppf(ranks/(n+1)))
        idx = [glist.index(g) for g in avail]
        in_set = np.zeros(n, bool); in_set[idx] = True
        out[s] = w[in_set].sum() / w.sum()
    return pd.Series(out), avail

def main():
    expr = load_expr()
    survival = load_survival()
    print(f"表达矩阵: {expr.shape} (基因 x 样本)")

    # 基因集
    stiff_df = pd.read_csv(f"{DATA_DIR}/../geneset/matrix_stiffness_genes.tsv", sep="\t")
    stiffness_genes = stiff_df["gene"].dropna().astype(str).str.strip().unique().tolist()
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]

    st, st_avail = ssgsea(expr, stiffness_genes)
    co, co_avail = ssgsea(expr, coag_genes)
    print(f"刚度集可测 {len(st_avail)}/537, 凝血臂可测 {len(co_avail)}/42")

    # mHLA-DR 分数（队列内标准化）
    mhla_genes = ["HLA-DRA", "HLA-DRB1", "HLA-DQA1"]
    mhla_avail = [g for g in mhla_genes if g in expr.index]
    mhla_sub = expr.loc[mhla_avail]
    mhla = mhla_sub.sub(mhla_sub.mean(axis=1), axis=0).div(mhla_sub.std(axis=1), axis=0).mean(axis=0)
    print(f"mHLA-DR 可测 {mhla_avail}")

    # 合并
    val = pd.DataFrame({"stiff": st, "coag": co, "mhla": mhla})
    val["survival"] = val.index.map(lambda s: survival.get(s, np.nan))

    # 队列内 z-score（用于方向比较）
    for col in ["stiff", "coag", "mhla"]:
        val[f"{col}_z"] = (val[col] - val[col].mean()) / val[col].std()

    # ---------- 1. 共变 ----------
    r, p = stats.spearmanr(val["stiff"], val["coag"])
    print(f"\n========== E-MTAB-4451 结果 (n={len(val)}) ==========")
    print(f"[1] 刚度 × 凝血 共变: Spearman r = {r:+.3f}, p = {p:.3e}")
    print(f"    -> {'✅ 正相关（共变方向复现）' if r > 0 else '⚠️ 负相关（与发现队列相反！）'}")

    # ---------- 2. 方向 ----------
    ns = val[val["survival"]=="non survivor"]; sv = val[val["survival"]=="survivor"]
    print(f"\n[2] non-survivor (n={len(ns)}) vs survivor (n={len(sv)}):")
    for col in ["stiff_z", "coag_z", "mhla_z"]:
        U, p = stats.mannwhitneyu(ns[col], sv[col])
        direction = "NS更低(同向✅)" if ns[col].mean() < sv[col].mean() else "NS更高(反向⚠️)"
        print(f"    {col:8s} NS={ns[col].mean():+.3f} vs SV={sv[col].mean():+.3f}, p={p:.4f}, {direction}")

    # ---------- 3. mHLA-DR × 凝血/刚度 ----------
    print(f"\n[3] mHLA-DR 关联:")
    for col in ["stiff", "coag"]:
        r, p = stats.spearmanr(val["mhla"], val[col])
        print(f"    mHLA-DR × {col}: r={r:+.3f}, p={p:.2e}")

    # 保存
    val.to_csv(f"{OUT_DIR}/validation_emtab4451.csv")
    print(f"\n已保存 validation_emtab4451.csv")

if __name__ == "__main__":
    main()
