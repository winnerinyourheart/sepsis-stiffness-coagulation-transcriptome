#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_scores.py — 刚度分数 × 凝血分数 单样本打分（GSE65682 MARS 脓毒症队列）

科学问题（方案B）：脓毒症中基质刚度转录组信号是否与凝血通路信号共变、并共同决定结局。

数据源头（审计铁律：从原始探针矩阵起步，不用历史中间产物）：
  - GSE65682_expr_probes.csv.gz  原始探针级表达矩阵（RMA 归一化）
  - GSE65682_gpl_annot.csv.gz    探针 -> Gene Symbol 注释
  - GSE65682_pheno.csv           表型（pneumonia_diagnoses 区分脓毒症/对照；mortality/thrombocytopenia）

基因集：
  - 刚度：matrix_stiffness_genes.tsv（537 基因 / 11 类，来自 stiffness 项目）
  - 凝血：KEGG hsa04610 凝血臂 42 基因（来自 MR 项目 kegg_hsa04610_arms_authoritative.json）
  - 阴性对照：炎症/免疫基因（用于排除"脓毒症整体炎症上调"的假相关）

输出：
  - results/scores_gse65682.csv  每样本 stiffness_score / coag_score / inflammation_score

方法：ssGSEA（rank-based，手写实现），逐样本 z-score。
"""
import json
import sys
import numpy as np
import pandas as pd
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


# ---------- 路径（可参数化） ----------
DATA_DIR = r"" + EXT_DIR + "/sepsis_bioinfo/data"
MR_DIR   = r"" + EXT_DIR + "/lactylation_coagulation_mr"
OUT_DIR  = r"" + RESULT_DIR + ""

def load_probe_matrix():
    """探针级表达矩阵 -> 基因级（多探针取均值）。"""
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    annot = annot[annot["Gene Symbol"] != ""]
    # 探针 -> 基因 映射
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    # 只保留矩阵里有的探针
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    # 多探针取均值
    expr = expr.groupby(expr.index).mean()
    return expr  # genes x samples

def load_pheno():
    pheno = pd.read_csv(f"{DATA_DIR}/GSE65682_pheno.csv", low_memory=False)
    return pheno

def ssgsea_scores(expr, gene_sets):
    """
    手写 ssGSEA（rank-based）。gene_sets: dict[name] -> list[genes]。
    返回 DataFrame: index=样本, columns=各基因集分数。
    参照 Barbie et al. 2009 的 ssGSEA 思路（秩 + 正态分布权重累加）。
    """
    genes = expr.index.tolist()
    # 基因集过滤为表达矩阵中实际存在的基因
    filtered = {}
    for name, gs in gene_sets.items():
        avail = [g for g in gs if g in expr.index]
        filtered[name] = avail

    samples = expr.columns.tolist()
    scores = pd.DataFrame(index=samples, columns=list(filtered.keys()), dtype=float)
    n_genes = len(genes)

    # 秩变换（每样本内）
    for s in samples:
        vals = expr[s].values
        # 排序的秩（从大到小，rank 1 = 最高表达）
        order = np.argsort(-vals)
        ranks = np.empty(n_genes, dtype=float)
        ranks[order] = np.arange(1, n_genes + 1)
        # 正态分布权重（rank -> normal CDF 分位）
        # w_i = |phi^-1(rank/(N+1))|
        from scipy.stats import norm
        p = ranks / (n_genes + 1)
        w = np.abs(norm.ppf(p))
        for name, gs in filtered.items():
            if len(gs) == 0:
                scores.loc[s, name] = np.nan
                continue
            idx = [genes.index(g) for g in gs]
            in_set = np.zeros(n_genes, bool)
            in_set[idx] = True
            # 上游累计：命中基因的权重和 / 总权重和
            w_hit = w[in_set].sum()
            w_all = w.sum()
            scores.loc[s, name] = w_hit / w_all
    return scores, filtered

def main():
    print("[1/4] 加载探针矩阵 -> 基因矩阵 ...")
    expr = load_probe_matrix()
    print(f"      基因 x 样本 = {expr.shape}")

    print("[2/4] 加载基因集 ...")
    # 刚度 537 基因集
    stiff_df = pd.read_csv(f"{DATA_DIR}/../geneset/matrix_stiffness_genes.tsv", sep="\t")
    stiffness_genes = stiff_df["gene"].dropna().astype(str).str.strip().unique().tolist()
    # 凝血臂 42 基因
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]
    # 阴性对照：炎症/免疫基因（用 stiffness 项目 geneset/immune_markers.gmt 若有，否则内置最小炎症集）
    # 这里用内置的一小组经典炎症基因做阴性对照（保证可复现、可解释）
    inflam_genes = ["IL1B", "IL6", "TNF", "IL8", "CXCL8", "IL10", "IFNG", "STAT1",
                    "NFKB1", "RELA", "PTGS2", "S100A8", "S100A9", "S100A12", "CCL2",
                    "CCL3", "CCL4", "CXCL1", "CXCL2", "CXCL10"]

    gene_sets = {
        "stiffness_537": stiffness_genes,
        "coag_42": coag_genes,
        "inflammation_20": inflam_genes,
    }

    print("[3/4] ssGSEA 打分 ...")
    scores, filtered = ssgsea_scores(expr, gene_sets)
    for name, gs in filtered.items():
        print(f"      {name}: 基因集 {len(gene_sets[name])} 个, 矩阵中可测 {len(gs)} 个")

    print("[4/4] 合并表型 + 逐样本 z-score ...")
    pheno = load_pheno()
    # 样本 ID 对齐：矩阵列名是 GSM ID
    pheno_map = pheno.set_index("Sample_geo_accession")
    scores["pneumonia"] = scores.index.map(
        lambda s: pheno_map.loc[s, "pneumonia_diagnoses"] if s in pheno_map.index else np.nan)
    scores["endotype_class"] = scores.index.map(
        lambda s: pheno_map.loc[s, "endotype_class"] if s in pheno_map.index else np.nan)
    scores["mortality_28d"] = scores.index.map(
        lambda s: pd.to_numeric(pheno_map.loc[s, "mortality_event_28days"], errors="coerce")
        if s in pheno_map.index else np.nan)
    scores["time_to_event_28d"] = scores.index.map(
        lambda s: pd.to_numeric(pheno_map.loc[s, "time_to_event_28days"], errors="coerce")
        if s in pheno_map.index else np.nan)
    scores["thrombocytopenia"] = scores.index.map(
        lambda s: pheno_map.loc[s, "thrombocytopenia"] if s in pheno_map.index else np.nan)

    # 脓毒症判定：mortality_event_28days 非空（n=479），与 endotype_class 非空一致
    scores["is_sepsis"] = scores["mortality_28d"].notna()

    # 逐样本 z-score（只在脓毒症样本内做，避免健康对照污染）
    seps = scores[scores["is_sepsis"]].copy()
    for col in list(gene_sets.keys()):
        mu = seps[col].mean(); sd = seps[col].std()
        seps[f"{col}_z"] = (seps[col] - mu) / sd
        scores[f"{col}_z"] = np.nan
        scores.loc[seps.index, f"{col}_z"] = seps[f"{col}_z"]

    out = f"{OUT_DIR}/scores_gse65682.csv"
    scores.to_csv(out)
    print(f"\n完成。输出: {out}")
    print(f"脓毒症样本数(is_sepsis): {(scores['is_sepsis']).sum()}")
    print(f"  其中 28d 死亡: {(seps['mortality_28d']==1).sum() if 'mortality_28d' in seps else 'NA'}")
    print(f"  其中 thrombocytopenia 记录数: {seps['thrombocytopenia'].notna().sum()}")

if __name__ == "__main__":
    main()
