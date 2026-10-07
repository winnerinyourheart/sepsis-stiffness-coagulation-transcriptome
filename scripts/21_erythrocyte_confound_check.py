#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
21_erythrocyte_confound_check.py — 核查红细胞信号是否为技术假象

【为什么做】
  17_immune_infiltration.py 发现 Erythrocyte 分数与刚度分数 r = -0.708（p=4e-74），
  高得可疑。全血转录组中红系基因（HBB/HBA）主要反映：
    (a) 样本溶血 / 红细胞污染（技术假象）
    (b) 应激性红系生成（erythroid precursors，GSE167363 单细胞已报道 — 生物学真实）
  必须区分，否则整个"刚度-凝血共变"可能被归因为溶血。

【检验设计】
  1. 管家基因对照：若为 RNA 质量假象，管家基因也应随刚度分数同向变化
  2. 血红蛋白 vs 其他红系基因分离：HBB/HBA（成熟红细胞）vs AHSP/ALAS2/SLC4A1（红系前体）
     -> 若仅 HBB/HBA 相关，提示溶血污染；若 AHSP/ALAS2 也相关，提示真红系生成
  3. 排除溶血样本后共变是否仍存在
  4. 与 GSE65682 的已知质量控制列（如有）交叉

输出：results/erythrocyte_confound_check.md + erythrocyte_confound.csv
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
ROOT = Path(r"" + RESULT_DIR + "/..")
OUT = ROOT / "results"

HK = ["GAPDH", "ACTB", "B2M", "RPLP0", "GUSB", "HPRT1", "PGK1", "TBP", "PPIA", "RPL13A"]
MATURE_RBC = ["HBB", "HBA1", "HBA2", "HBD"]
ERYTHROID_PRECURSOR = ["AHSP", "ALAS2", "SLC4A1", "EPB42", "GYPA", "KLF1", "GATA1", "TFRC", "CA1"]


def load_expr():
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


def main():
    expr = load_expr()
    scores = pd.read_csv(OUT / "scores_gse65682.csv", index_col=0)
    sc = scores[scores["is_sepsis"] == True]
    common = sc.index.intersection(expr.columns)
    sc = sc.loc[common]

    rows = []
    def corr_group(label, genes):
        present = [g for g in genes if g in expr.index]
        if not present:
            return
        sub = expr.loc[present, common]
        # 组内平均（先逐基因 z-score 消量纲）
        z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, np.nan), axis=0)
        grp = z.mean(axis=0)
        for name, col in [("stiffness", "stiffness_537_z"), ("coagulation", "coag_42_z")]:
            r, p = stats.spearmanr(grp, sc[col])
            rows.append({"group": label, "n_genes": len(present), "score": name, "r": r, "p": p})
        # 输出组分基因明细
        for g in present:
            r1, p1 = stats.spearmanr(expr.loc[g, common], sc["stiffness_537_z"])
            rows.append({"group": f"single:{g}", "n_genes": 1, "score": "stiffness",
                         "r": r1, "p": p1})

    corr_group("housekeeping", HK)
    corr_group("mature_RBC", MATURE_RBC)
    corr_group("erythroid_precursor", ERYTHROID_PRECURSOR)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "erythrocyte_confound.csv", index=False)

    # --- 溶血样本剔除敏感性分析 ---
    # 以 mature_RBC 组分数上 5% 为"疑似溶血"阈值
    mrbc = expr.loc[[g for g in MATURE_RBC if g in expr.index], common]
    z = mrbc.sub(mrbc.mean(axis=1), axis=0).div(mrbc.std(axis=1).replace(0, np.nan), axis=0)
    mrbc_score = z.mean(axis=0)
    thr = np.percentile(mrbc_score, 95)
    keep = mrbc_score < thr
    r_all, p_all = stats.spearmanr(sc["stiffness_537_z"], sc["coag_42_z"])
    r_f, p_f = stats.spearmanr(sc.loc[keep, "stiffness_537_z"], sc.loc[keep, "coag_42_z"])

    lines = [
        "# 红细胞信号混杂核查",
        "",
        "> 脚本：`scripts/21_erythrocyte_confound_check.py`；明细：`erythrocyte_confound.csv`",
        "",
        "## 一、分组相关性（vs 刚度分数）",
        "",
        "| 基因组 | 基因数 | vs 刚度 r | p |",
        "|---|---|---|---|",
    ]
    for g in ["housekeeping", "mature_RBC", "erythroid_precursor"]:
        sub = df[(df["group"] == g) & (df["score"] == "stiffness")]
        if len(sub):
            lines.append(f"| {g} | {sub.iloc[0]['n_genes']} | {sub.iloc[0]['r']:.3f} | "
                         f"{sub.iloc[0]['p']:.2e} |")

    lines += ["", "## 二、单个红系基因明细（vs 刚度）", "",
              "| 基因 | r | p |", "|---|---|---|"]
    for _, r in df[df["group"].str.startswith("single:")].iterrows():
        lines.append(f"| {r['group'].split(':')[1]} | {r['r']:.3f} | {r['p']:.2e} |")

    lines += [
        "", "## 三、剔除疑似溶血样本（成熟 RBC 分数上 5%）后的共变", "",
        f"- 全部样本（n={len(sc)}）：r = {r_all:.3f}, p = {p_all:.2e}",
        f"- 剔除后（n={int(keep.sum())}）：r = {r_f:.3f}, p = {p_f:.2e}",
        "",
        "## 四、判读",
        "",
        "- 若管家基因组同样与刚度强相关 → 提示**技术性 RNA 质量/溶血混杂**，需谨慎。",
        "- 若仅 HBB/HBA（成熟红细胞）强相关、而 AHSP/ALAS2（红系前体）弱 → 提示溶血污染。",
        "- 若红系前体基因同样显著 → 支持**应激性红系生成**这一生物学解释（与 GSE167363 单细胞一致）。",
        "- 剔除疑似溶血样本后共变若稳定 → 共变结论不是溶血假象。",
    ]
    (OUT / "erythrocyte_confound_check.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
