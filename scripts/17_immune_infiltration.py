#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_immune_infiltration.py — 免疫浸润（28 细胞 LM22 风格 marker 集 ssGSEA）

目的：
  回应审稿人最致命的攻击——"全血通路分数只是白细胞组成的反映"。
  做法：把白细胞组成本身量化出来，检验 LL/HH 象限的差异是否由细胞丰度解释，
  以及控制细胞丰度后刚度-凝血共变是否仍显著（把攻击反向变为论据）。

方法：
  - marker 集：LM22（Newman et al. 2015, Nat Methods）22 种 + 6 种扩展免疫/基质细胞
    (neutrophil, MDSC-like, megakaryocyte, endothelial, fibroblast, erythrocyte) = 28 类
  - 打分：rank-based ssGSEA（与 01_scores.py 同一实现，保证可比）
  - 输出：每样本 28 细胞分数 → 与 stiffness/coag 相关 → LL/HH 差异 → 偏相关控制

输入：GSE65682 原始探针矩阵 + 注释；results/scores_gse65682.csv
输出：results/immune_infiltration_scores.csv
      results/immune_infiltration_corr.csv
      results/immune_infiltration_quadrant.csv
      results/immune_infiltration_summary.md
"""
import numpy as np
import pandas as pd
from scipy import stats
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
OUT = Path(r"" + RESULT_DIR + "")

# ---------------------------------------------------------------------------
# LM22 风格 28 细胞 marker 集
# 来源：LM22 (Newman 2015) 核心标记 + 扩展类（来源见 results/文献药物事实核查表.md）
# 每类保留 8-30 个高特异标记，避免过长稀释信号
# ---------------------------------------------------------------------------
CELL_MARKERS = {
    # --- LM22 原 22 类 ---
    "B_naive": ["MS4A1","CD19","CD79A","CD79B","TCL1A","IGHD","IGHM","CR2","FCER2","BANK1","BLK","PNOC"],
    "B_memory": ["MS4A1","CD27","CD79A","CD79B","AIM2","TNFRSF13B","IGHG1","IGHG2","IGHG3","CRIP1","SPIB"],
    "Plasma_cells": ["MZB1","JCHAIN","IGKC","IGLL5","DERL3","XBP1","SDC1","PRDX4","TNFRSF17"],
    "T_CD8": ["CD8A","CD8B","CD3D","CD3E","CD3G","GZMK","CCL5","IL2RB","LAG3","NKG7"],
    "T_CD4_naive": ["CD4","CD3D","CD3E","CCR7","SELL","LEF1","TCF7","IL7R","NOSIP","TSHZ2"],
    "T_CD4_memory_resting": ["CD4","CD3D","IL7R","S100A4","CD40LG","AQP3","CD69","LDHB"],
    "T_CD4_memory_activated": ["CD4","CD3D","IL7R","CD69","CD38","HLA-DRA","MIR155HG","PTPRC"],
    "T_follicular_helper": ["CD4","CXCR5","ICOS","BCL6","PDCD1","SH2D1A","IL21","MAF"],
    "T_regulatory": ["FOXP3","IL2RA","CTLA4","IKZF2","TNFRSF18","LRRC32","CD4","CD3D"],
    "T_gamma_delta": ["TRDC","TRGC1","TRGC2","CD3D","KLRD1","NKG7","GZMB","IL2RB"],
    "NK_resting": ["KLRD1","KLRF1","NCR3","XCL1","XCL2","GZMK","PRF1","NKG7","GNLY"],
    "NK_activated": ["KLRD1","KLRF1","PRF1","GZMB","GNLY","FGFBP2","SPON2","NKG7"],
    "Monocytes": ["CD14","LYZ","VCAN","FCN1","S100A8","S100A9","CST3","FCGR3A","CSF1R","MNDA","CD68"],
    "Macrophages_M0": ["CD68","CD163","CSF1R","MSR1","MARCO","MRC1","FUT4","ITGAM"],
    "Macrophages_M1": ["NOS2","IL1B","TNF","CXCL9","CXCL10","CXCL11","SOCS3","IDO1","CD80"],
    "Macrophages_M2": ["CD163","MRC1","MSR1","VSIG4","MS4A4A","CCL18","CD209","MAF"],
    "Dendritic_cells_resting": ["CD1C","CLEC10A","FCER1A","CD209","HLA-DPB1","HLA-DQA1","CD1E","CLEC9A"],
    "Dendritic_cells_activated": ["CD83","CD86","CCR7","LAMP3","HLA-DRA","CD40","CD1C","IL12B"],
    "Mast_cells_resting": ["TPSAB1","TPSB2","CPA3","MS4A2","HDC","KIT","CMA1","GATA2"],
    "Mast_cells_activated": ["TPSAB1","TPSB2","CPA3","MS4A2","KIT","CCL4","IL6","PTGS2"],
    "Eosinophils": ["PRG2","PRG3","CLC","EPX","RNASE3","RNASE2","IL5RA","CCR3","SIGLEC8"],
    "Neutrophils": ["FCGR3B","CSF3R","S100A12","CXCR2","FPR2","AQP9","MME","ELANE","MPO","DEFA3","DEFA4","CEACAM8"],
    # --- 6 类扩展（脓毒症相关）---
    "Neutrophil_immature_band": ["DEFA3","DEFA4","CEACAM8","BASP1","CD24","MMP8","OLFM4","RETN","TSPO"],
    "MDSC_like": ["S100A8","S100A9","ARG1","ITGAM","CD33","IL4R","CXCR4","VEGFA"],
    "Megakaryocyte_platelet": ["PF4","PPBP","ITGA2B","GP9","GP6","SELP","THBS1","TUBB1"],
    "Endothelial": ["PECAM1","VWF","CDH5","KDR","ENG","CD34","ESAM","RAMP2","TIE1"],
    "Fibroblast": ["COL1A1","COL1A2","COL3A1","DCN","LUM","FAP","THY1","PDGFRA"],
    "Erythrocyte": ["HBB","HBA1","HBA2","HBD","ALAS2","AHSP","SLC4A1","EPB42"],
}


def load_gene_level_matrix():
    """探针级 -> 基因级（与 01_scores.py 完全一致的映射口径）。"""
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    annot = annot[annot["Gene Symbol"] != ""]
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    expr = expr.groupby(expr.index).mean()
    return expr


def ssgsea_scores(expr, gene_set):
    """基因集平均 z-score 打分（逐样本 z-score 后取集合均值）。

    说明：原先的"简化 ssGSEA"实现有误（分子恒等导致所有样本得分相同），
    已替换为业界通用的 gene-set mean-z（GSVA 的 z-score 形式，Hänzelmann 2013），
    与主分析 01_scores.py 的口径一致，可直接比较。
    """
    genes = [g for g in gene_set if g in expr.index]
    if len(genes) < 3:
        return None, len(genes)
    sub = expr.loc[genes]
    # 逐基因在样本间 z-score（消平台/量纲差异）-> 集合内取均值
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0), len(genes)


def main():
    print("[1/5] 载入基因级表达矩阵 ...")
    expr = load_gene_level_matrix()
    print(f"      基因 x 样本 = {expr.shape}")

    # 只保留脓毒症样本（与主分析口径一致）
    scores = pd.read_csv(OUT / "scores_gse65682.csv", index_col=0)
    sep_samples = scores.index[scores["is_sepsis"] == True]
    expr_sep = expr[expr.columns.intersection(sep_samples)]
    print(f"      脓毒症样本 n = {expr_sep.shape[1]}")

    print("[2/5] 计算 28 类免疫/基质细胞分数 ...")
    imm = {}
    coverage = {}
    for cell, markers in CELL_MARKERS.items():
        s, n = ssgsea_scores(expr_sep, markers)
        coverage[cell] = f"{n}/{len(markers)}"
        if s is not None:
            imm[cell] = s
    imm_df = pd.DataFrame(imm)
    # 逐样本 z-score（与主分数一致）
    imm_z = (imm_df - imm_df.mean()) / imm_df.std()
    imm_z.to_csv(OUT / "immune_infiltration_scores.csv")
    print(f"      完成 {imm_df.shape[1]} 类；marker 覆盖：{coverage}")

    # ------------------------------------------------------------------
    print("[3/5] 细胞分数 vs 刚度/凝血分数 相关 ...")
    rows = []
    for cell in imm_z.columns:
        for score, col in [("stiffness", "stiffness_537_z"), ("coagulation", "coag_42_z"),
                           ("mHLA-DR", None)]:
            if col is None:
                continue
            r, p = stats.spearmanr(imm_df[cell].loc[scores.index.intersection(imm_df.index)],
                                   scores.loc[imm_df.index.intersection(scores.index), col])
            rows.append({"cell": cell, "score": score, "r": r, "p": p})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT / "immune_infiltration_corr.csv", index=False)

    # ------------------------------------------------------------------
    print("[4/5] LL/HH 象限细胞差异 ...")
    common = scores.index.intersection(imm_df.index)
    sc = scores.loc[common]
    sc = sc[sc["is_sepsis"] == True]
    st_med = sc["stiffness_537_z"].median()
    cg_med = sc["coag_42_z"].median()
    q = pd.Series(index=sc.index, dtype=object)
    for i in sc.index:
        hi_st = sc.loc[i, "stiffness_537_z"] >= st_med
        hi_cg = sc.loc[i, "coag_42_z"] >= cg_med
        q[i] = ("HH" if (hi_st and hi_cg) else "HL" if (hi_st and not hi_cg)
                else "LH" if (not hi_st and hi_cg) else "LL")
    q.name = "quadrant"
    sc = sc.join(q)

    qrows = []
    for cell in imm_z.columns:
        ll = imm_df.loc[sc.index[sc["quadrant"] == "LL"], cell]
        hh = imm_df.loc[sc.index[sc["quadrant"] == "HH"], cell]
        u, p = stats.mannwhitneyu(ll, hh, alternative="two-sided")
        qrows.append({"cell": cell, "LL_mean": ll.mean(), "HH_mean": hh.mean(),
                      "diff_LL_minus_HH": ll.mean() - hh.mean(), "p": p})
    qdf = pd.DataFrame(qrows).sort_values("p")
    qdf.to_csv(OUT / "immune_infiltration_quadrant.csv", index=False)

    # 关键：控制单核+中性粒丰度后，刚度-凝血共变是否仍在
    from numpy.linalg import lstsq
    df = sc.join(imm_df)
    res_rows = []
    for ctrl in [None, ["Monocytes"], ["Monocytes", "Neutrophils"],
                 ["Monocytes", "Neutrophils", "T_CD8", "B_naive"]]:
        y = df["stiffness_537_z"].values
        x = df["coag_42_z"].values
        if ctrl is None:
            r, p = stats.spearmanr(x, y)
        else:
            Z = np.column_stack([np.ones(len(df))] + [df[c].values for c in ctrl])
            # 残差化
            bx = lstsq(Z, x, rcond=None)[0]; rx = x - Z @ bx
            by = lstsq(Z, y, rcond=None)[0]; ry = y - Z @ by
            r, p = stats.pearsonr(rx, ry)
        label = "none" if ctrl is None else "+".join(ctrl)
        res_rows.append({"controlled_for": label, "n_controls": 0 if ctrl is None else len(ctrl),
                         "r": r, "p": p})
    pd.DataFrame(res_rows).to_csv(OUT / "immune_infiltration_partialcorr.csv", index=False)

    # ------------------------------------------------------------------
    print("[5/5] 撰写汇总 ...")
    ll_top = qdf.sort_values("diff_LL_minus_HH").head(8)
    lines = [
        "# 免疫浸润分析汇总（28 细胞 marker 集 ssGSEA）",
        "",
        "> 脚本：`scripts/17_immune_infiltration.py`；输出：`immune_infiltration_*.csv`",
        "",
        "## 一、核心目的",
        "量化白细胞/基质细胞组成本身，检验 LL/HH 象限差异是否只是细胞丰度差异。",
        "",
        "## 二、各类细胞与刚度/凝血分数相关性（|r| 前 10）",
        "",
        "| 细胞类 | vs 刚度 r | p | vs 凝血 r | p |",
        "|---|---|---|---|---|",
    ]
    piv = corr_df.pivot(index="cell", columns="score", values=["r", "p"])
    piv["absmax"] = piv[("r", "stiffness")].abs()
    for cell in piv.sort_values("absmax", ascending=False).head(10).index:
        lines.append(f"| {cell} | {piv.loc[cell,('r','stiffness')]:.3f} | "
                     f"{piv.loc[cell,('p','stiffness')]:.2e} | "
                     f"{piv.loc[cell,('r','coagulation')]:.3f} | "
                     f"{piv.loc[cell,('p','coagulation')]:.2e} |")

    lines += [
        "",
        "## 三、LL vs HH 象限细胞差异（按 p 排序前 10）",
        "",
        "| 细胞类 | LL 均值 | HH 均值 | LL−HH | p |",
        "|---|---|---|---|---|",
    ]
    for _, r in qdf.head(10).iterrows():
        lines.append(f"| {r['cell']} | {r['LL_mean']:.3f} | {r['HH_mean']:.3f} | "
                     f"{r['diff_LL_minus_HH']:+.3f} | {r['p']:.2e} |")

    lines += [
        "",
        "## 四、控制细胞丰度后，刚度-凝血共变是否仍显著",
        "",
        "| 控制变量 | r | p |",
        "|---|---|---|",
    ]
    pc = pd.read_csv(OUT / "immune_infiltration_partialcorr.csv")
    for _, r in pc.iterrows():
        lines.append(f"| {r['controlled_for']} | {r['r']:.3f} | {r['p']:.2e} |")

    lines += [
        "",
        "## 五、结论（自动生成，需人工复核）",
        "",
        f"- 控制单核细胞后共变 r = {pc.loc[pc['controlled_for']=='Monocytes','r'].values[0]:.3f}",
        f"- 控制单核+中性粒后共变 r = {pc.loc[pc['controlled_for']=='Monocytes+Neutrophils','r'].values[0]:.3f}",
        f"- 全控制后共变 r = {pc.loc[pc['controlled_for']=='Monocytes+Neutrophils+T_CD8+B_naive','r'].values[0]:.3f}",
        "",
        "→ 若控制后仍显著为正，说明共变不是单纯细胞组成假象。",
    ]
    (OUT / "immune_infiltration_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
