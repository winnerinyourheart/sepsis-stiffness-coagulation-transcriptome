#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
23_singlecell_localization.py — 单细胞定位（GSE167363）

科学目的：
  把"全血转录组关联"升级为"细胞水平定位"。验证假设：
    H1. 6 个交集基因（PLAT/PLAU/PLAUR/SERPINE1/THBD/VWF）主要表达在
        单核细胞 / 内皮（PBMC 中主要为单核），而非 T/B 细胞
    H2. mHLA-DR 三基因（HLA-DRA/DRB1/DQA1）在单核细胞中表达最高，
        且在 non-survivor 患者中低于 survivor（验证免疫麻痹的细胞来源）

数据：data/sc/GSM*_*_{barcodes,features,matrix}.mtx.gz（10x 格式）
分组：HC（健康对照）/ S（survivor）/ NS（non-survivor）
输出：results/singlecell_*.csv + singlecell_report.md
"""
import gzip
import warnings
from pathlib import Path

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


warnings.filterwarnings("ignore")

ROOT = Path(r"" + RESULT_DIR + "/..")
SC = ROOT / "data" / "sc"
OUT = ROOT / "results"

# 样本 -> 分组
GROUPS = {
    "HC1": "HC", "HC2": "HC",
    "P25_T0": "NS", "P25_T6": "NS", "NSES_T0": "NS", "NSES_T6": "NS",
    "P50_T0": "S", "P50_T6": "S", "S2_T0": "S", "S2_T6": "S",
    "S3_T0": "S", "S3_T6": "S",
}

INTERSECTION = ["PLAT", "PLAU", "PLAUR", "SERPINE1", "THBD", "VWF"]
MHLA = ["HLA-DRA", "HLA-DRB1", "HLA-DQA1"]
# PBMC 主要细胞类型 marker（用于注释）
CELL_MARKERS = {
    "Monocyte": ["CD14", "LYZ", "FCN1", "VCAN", "S100A8", "S100A9"],
    "T_cell": ["CD3D", "CD3E", "IL7R", "CD8A", "CD4"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "PRF1"],
    "B_cell": ["MS4A1", "CD79A", "CD79B", "IGHM"],
    "Platelet": ["PPBP", "PF4", "ITGA2B", "GP9"],
    "Erythroid": ["HBB", "HBA1", "HBA2", "AHSP", "ALAS2"],
    "Dendritic": ["CD1C", "FCER1A", "CLEC10A", "LILRA4"],
}


def read_10x(sample_name):
    """读取一个样本的 10x 三件套 -> (genes, cells, counts[dense])。"""
    pref = SC / sample_name
    bcf = next(SC.glob(f"{sample_name}_*barcodes.tsv.gz"), None)
    ftf = next(SC.glob(f"{sample_name}_*features.tsv.gz"), None)
    mtf = next(SC.glob(f"{sample_name}_*matrix.mtx.gz"), None)
    if not (bcf and ftf and mtf):
        return None
    try:
        with gzip.open(mtf, "rt") as f:
            header = f.readline()
            dims = f.readline().split()
            if len(dims) < 3:
                return None
            n_genes, n_cells, n_nnz = int(dims[0]), int(dims[1]), int(dims[2])
            rows, cols, vals = [], [], []
            for line in f:
                p = line.split()
                if len(p) < 3:
                    break
                rows.append(int(p[0])); cols.append(int(p[1])); vals.append(int(p[2]))
        genes = [l.split("\t")[1] if "\t" in l else l.strip()
                 for l in gzip.open(ftf, "rt").read().splitlines()]
        cells = [l.strip() for l in gzip.open(bcf, "rt").read().splitlines()]
        M = np.zeros((n_genes, n_cells), dtype=np.int32)
        M[np.array(rows) - 1, np.array(cols) - 1] = np.array(vals)
        return genes, cells, M
    except Exception as e:
        print(f"  [{sample_name}] 读取失败: {e}")
        return None


def main():
    avail = sorted({p.name.split("_")[1] + "_" + p.name.split("_")[2]
                    for p in SC.glob("*_matrix.mtx.gz")})
    print(f"[1/3] 可用样本 {len(avail)}: {avail}")
    if not avail:
        print("无数据，退出"); return

    per_cell = {}   # 样本 -> DataFrame(cell x gene) 靶基因
    per_group_cells = {}
    for s in avail:
        r = read_10x(s)
        if r is None:
            continue
        genes, cells, M = r
        gi = {g: i for i, g in enumerate(genes)}
        want = [g for g in INTERSECTION + MHLA if g in gi]
        if not want:
            continue
        sub = pd.DataFrame(M[[gi[g] for g in want]].T, columns=want, index=cells)
        # 细胞类型注释：marker 平均表达最高者
        type_score = {}
        for ct, mk in CELL_MARKERS.items():
            present = [m for m in mk if m in gi]
            if len(present) >= 2:
                type_score[ct] = M[[gi[m] for m in present]].mean(axis=0)
        if type_score:
            ts = pd.DataFrame(type_score, index=cells)
            sub["cell_type"] = ts.idxmax(axis=1)
        per_cell[s] = sub
        per_group_cells[s] = GROUPS.get(s, "?")
        print(f"  {s}: {M.shape[1]} cells, 靶基因 {len(want)}")

    if not per_cell:
        print("无可用样本"); return

    # ---- 汇总：交集基因在细胞类型的表达占比 ----
    print("[2/3] 细胞类型定位 ...")
    rows = []
    for s, df in per_cell.items():
        if "cell_type" not in df.columns:
            continue
        for ct in CELL_MARKERS:
            cells_ct = df[df["cell_type"] == ct]
            if len(cells_ct) < 10:
                continue
            for g in INTERSECTION:
                if g in df.columns:
                    rows.append({"sample": s, "group": GROUPS.get(s, "?"),
                                 "cell_type": ct, "gene": g,
                                 "mean_expr": cells_ct[g].mean(),
                                 "pct_expressing": (cells_ct[g] > 0).mean() * 100,
                                 "n_cells": len(cells_ct)})
    loc = pd.DataFrame(rows)
    if len(loc):
        loc.to_csv(OUT / "singlecell_gene_by_celltype.csv", index=False)

    # ---- mHLA-DR 按分组 ----
    print("[3/3] mHLA-DR 分组比较 ...")
    mrow = []
    for s, df in per_cell.items():
        present = [g for g in MHLA if g in df.columns]
        if not present:
            continue
        mrow.append({"sample": s, "group": GROUPS.get(s, "?"),
                     "mHLA_DR": df[present].mean(axis=1).mean(),
                     "n_cells": len(df)})
    mdf = pd.DataFrame(mrow)
    if len(mdf):
        mdf.to_csv(OUT / "singlecell_mhla_by_sample.csv", index=False)

    lines = ["# 单细胞定位（GSE167363 脓毒症 PBMC）", "",
             f"> 脚本：`scripts/23_singlecell_localization.py`；纳入样本 {len(per_cell)} 个", ""]
    if len(loc):
        piv = loc.groupby(["gene", "cell_type"])["pct_expressing"].mean().unstack()
        lines += ["## 一、交集基因表达细胞占比（%）", "",
                  "| 基因 | " + " | ".join(piv.columns) + " |",
                  "|---|" + "---|" * len(piv.columns)]
        for g in piv.index:
            lines.append(f"| {g} | " + " | ".join(
                f"{piv.loc[g, c]:.1f}" if pd.notna(piv.loc[g, c]) else "-" for c in piv.columns) + " |")
    if len(mdf):
        lines += ["", "## 二、mHLA-DR 各样本", "",
                  "| 样本 | 分组 | 细胞数 | mHLA-DR |", "|---|---|---|---|"]
        for _, r in mdf.iterrows():
            lines.append(f"| {r['sample']} | {r['group']} | {r['n_cells']} | {r['mHLA_DR']:.3f} |")
        grp = mdf.groupby("group")["mHLA_DR"].mean()
        lines += ["", "**分组均值**：" + "；".join(f"{k}={v:.3f}" for k, v in grp.items())]

    (OUT / "singlecell_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
