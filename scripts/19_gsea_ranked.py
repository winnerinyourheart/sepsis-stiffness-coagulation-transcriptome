#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_gsea_ranked.py — 排序式 GSEA（全基因，不设阈值）

【为什么重做】
  11_covariation_network_enrichment.py 用"逐基因相关显著性"筛出 9382/11762 个基因
  （占 80%）再做富集，属于阈值筛选 + 多重比较膨胀，富集到的"核糖体/剪接体/RNA 转运"
  是全血 RNA 质量与细胞组成的已知混杂信号，不是生物学结论。

【本脚本做法】
  1. 全基因按与刚度/凝血分数的 Spearman rho 排序（不设 p 值阈值）
  2. 用 gseapy.prerank（GSEA）对 GO-BP / Reactome / KEGG 富集 → 得 NES + FDR
  3. 关键：额外做"残差化"版本——先回归掉单核/T/B/中性粒/红细胞分数，再对残差排序，
     以排除细胞组成驱动的富集

输出：
  results/gsea_stiffness_raw.csv / gsea_coag_raw.csv
  results/gsea_stiffness_resid.csv / gsea_coag_resid.csv
  results/gsea_summary.md
"""
import numpy as np
import pandas as pd
from scipy import stats
from numpy.linalg import lstsq
import gseapy as gp
from pathlib import Path
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
ROOT = Path(r"" + RESULT_DIR + "/..")
OUT = ROOT / "results"

GENESETS = ["GO_Biological_Process_2023", "Reactome_2022", "KEGG_2021_Human"]
CTRL_CELLS = ["Monocytes", "Neutrophils", "T_CD8", "B_naive", "Erythrocyte",
              "Megakaryocyte_platelet"]


def load_gene_expr():
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    annot = annot[annot["Gene Symbol"] != ""]
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    probes = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes]
    expr.index = expr.index.map(id2sym)
    return expr.groupby(expr.index).mean()


def rank_genes(expr, score, control_df=None):
    """返回按 Spearman rho 降序的基因列表（用于 prerank）。"""
    common = expr.columns.intersection(score.index)
    E = expr[common]
    # 消除低表达/低变异基因（GSEA 常规过滤：保留在 >20% 样本中检出的基因）
    keep = (E > 0).sum(axis=1) >= 0.2 * E.shape[1]
    E = E.loc[keep]
    y = score.loc[common].values

    rho = []
    for g in E.index:
        x = E.loc[g].values
        if np.std(x) == 0:
            rho.append(0.0); continue
        r, _ = stats.spearmanr(x, y)
        rho.append(0.0 if np.isnan(r) else r)
    rho = pd.Series(rho, index=E.index, name="rho")

    rho_resid = None
    if control_df is not None:
        ctrl = [c for c in control_df.columns if c in control_df.index or True]
        Z = np.column_stack([np.ones(len(common))] +
                            [control_df.loc[common, c].values for c in CTRL_CELLS
                             if c in control_df.columns])
        # 回归掉细胞分数：对每个基因的 rho 序列无法直接做，改为对表达残差化后重算
        rr = []
        for g in E.index:
            x = E.loc[g].values
            b = lstsq(Z, x, rcond=None)[0]
            rx = x - Z @ b
            r, _ = stats.spearmanr(rx, y)
            rr.append(0.0 if np.isnan(r) else r)
        rho_resid = pd.Series(rr, index=E.index, name="rho_resid")

    return rho.sort_values(ascending=False), (rho_resid.sort_values(ascending=False)
                                              if rho_resid is not None else None)


def run_gsea(ranked, tag):
    """prerank GSEA，返回合并结果。"""
    res = gp.prerank(rnk=ranked, gene_sets=GENESETS, min_size=15, max_size=500,
                     permutation_num=1000, seed=42, threads=4, outdir=None,
                     verbose=False, no_plot=True)
    df = res.res2d.copy()
    df["dataset"] = tag
    return df


def main():
    print("[1/4] 载入表达与分数 ...")
    expr = load_gene_expr()
    scores = pd.read_csv(OUT / "scores_gse65682.csv", index_col=0)
    sc = scores[scores["is_sepsis"] == True]
    imm = pd.read_csv(OUT / "immune_infiltration_scores.csv", index_col=0)
    common = sc.index.intersection(imm.index)
    sc = sc.loc[common]; imm = imm.loc[common]
    print(f"      基因 {expr.shape[0]}；脓毒症样本 n={len(sc)}")

    all_res = []
    for name, col in [("stiffness", "stiffness_537_z"), ("coag", "coag_42_z")]:
        print(f"[2/4] 排序 + GSEA：{name} ...")
        ranked, ranked_r = rank_genes(expr, sc[col], imm)
        for suffix, rk in [("raw", ranked), ("resid", ranked_r)]:
            if rk is None:
                continue
            try:
                d = run_gsea(rk, f"{name}_{suffix}")
                d.to_csv(OUT / f"gsea_{name}_{suffix}.csv", index=False)
                all_res.append(d)
                sig = d[d["FDR q-val"].astype(float) < 0.25]
                print(f"      {name}_{suffix}: {len(d)} 通路，FDR<0.25 有 {len(sig)} 条")
            except Exception as e:
                print(f"      [FAIL] {name}_{suffix}: {e}")

    if not all_res:
        print("无结果"); return
    big = pd.concat(all_res, ignore_index=True)

    print("[3/4] 撰写汇总 ...")
    lines = ["# GSEA（排序式，全基因，无阈值）汇总", "",
             "> 脚本：`scripts/19_gsea_ranked.py`；原始结果：`gsea_{stiffness,coag}_{raw,resid}.csv`",
             "> raw = 直接用样本排序；resid = 回归掉单核/中性粒/T/B/红细胞/巨核后排序",
             ""]
    for ds in big["dataset"].unique():
        d = big[big["dataset"] == ds].copy()
        d["FDR"] = d["FDR q-val"].astype(float)
        d = d[d["FDR"] < 0.25].sort_values("NES", ascending=False)
        lines += [f"## {ds}（FDR<0.25，共 {len(d)} 条）", "",
                  "| 通路 | NES | FDR |", "|---|---|---|"]
        for _, r in pd.concat([d.head(8), d.tail(8)]).iterrows():
            lines.append(f"| {r['Term']} | {r['NES']:.2f} | {r['FDR']:.3f} |")
        lines.append("")
    (OUT / "gsea_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))
    print("\n[4/4] 完成 ->", OUT / "gsea_summary.md")


if __name__ == "__main__":
    main()
