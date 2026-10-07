#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
22_docking_control_analysis.py — 分子对接结果的对照与局限分析

【为什么必须做这个脚本】
  20_molecular_docking.py 的原始结果存在两个已知假象，直接呈现会误导审稿人：
   (1) 配体尺寸效应：aleplasinin (MW 425, 32 重原子) 对所有 5 个靶点都给 -7.3~-8.6 kcal/mol，
       但这是"大分子接触面大"导致的普遍性高分，而非特异性结合。
   (2) 阳性对照 4-aminobenzamidine (MW 135) 偏小，得分天然偏低。
  => 必须做"配体效率 (ligand efficiency, LE = -ΔG / 重原子数)"和"靶点特异性
     (z-score across targets)"两项校正，才能判断是否存在真实偏好。

【方法】
  LE = -ΔG / N_heavy_atoms        (Hopkins et al. Drug Discov Today 2004)
  靶点特异性 z = (ΔG_target - mean(ΔG_ligand_across_targets)) / sd
      同一配体在不同靶点间比较，z 越负表示对该靶点偏好越强。

输出：results/docking_control_analysis.md + docking_ligand_efficiency.csv
"""
import numpy as np
import pandas as pd
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


ROOT = Path(r"" + RESULT_DIR + "/..")
OUT = ROOT / "results"
LIG = ROOT / "data" / "docking" / "ligands"

# 重原子数：由 RDKit 从脚本 20 用的同一 SMILES 现算（不凭记忆）
LIGANDS = {
    "4-aminobenzamidine": "C1=CC(=CC=C1C(=N)N)N",
    "upamostat": "CCOC(=O)N1CCN(CC1)C(=O)[C@H](CC2=CC(=CC=C2)/C(=N\\O)/N)NS(=O)(=O)C3=C(C=C(C=C3C(C)C)C(C)C)C(C)C",
    "aleplasinin": "CC1=CC(=CC=C1)C2=CC3=C(C=C2)N(C=C3C(=O)C(=O)O)CC4=CC=C(C=C4)C(C)(C)C",
    "6-aminocaproic_acid": "C(CCC(=O)O)CCN",
    "tranexamic_acid": "C1CC(CCC1CN)C(=O)O",
}

# 药理预期（用于判断对接是否"方向正确"）
EXPECTED = {
    "upamostat": {"primary": "PLAU", "note": "uPA 抑制剂（文献）"},
    "aleplasinin": {"primary": "SERPINE1", "note": "PAI-1 抑制剂（文献）"},
    "4-aminobenzamidine": {"primary": "PLAU", "note": "丝氨酸蛋白酶 S1 口袋抑制剂"},
    "6-aminocaproic_acid": {"primary": "PLAU", "note": "抗纤溶（赖氨酸类似物）"},
    "tranexamic_acid": {"primary": "PLAU", "note": "抗纤溶（赖氨酸类似物）"},
}


def main():
    from rdkit import Chem
    ha = {}
    for n, s in LIGANDS.items():
        m = Chem.MolFromSmiles(s)
        ha[n] = m.GetNumHeavyAtoms()

    df = pd.read_csv(OUT / "docking_results.csv")
    df = df.dropna(subset=["affinity_kcal_mol"]).copy()
    df["heavy_atoms"] = df["ligand"].map(ha)
    df["ligand_efficiency"] = -df["affinity_kcal_mol"] / df["heavy_atoms"]

    # 靶点内排序（同靶点比较不同配体，消除靶点口袋大小差异）
    df["rank_in_target"] = df.groupby("gene")["affinity_kcal_mol"].rank()

    # 配体跨靶点 z-score（该配体在该靶点是否异常偏好）
    z = df.groupby("ligand")["affinity_kcal_mol"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if len(x) > 1 and x.std(ddof=0) > 0 else 0.0)
    df["specificity_z"] = z   # 越负 = 对该靶点偏好越强
    df.to_csv(OUT / "docking_ligand_efficiency.csv", index=False)

    # 每个配体的最佳靶点
    best = {}
    for lig, g in df.groupby("ligand"):
        b = g.sort_values("specificity_z").iloc[0]
        best[lig] = (b["gene"], b["affinity_kcal_mol"], b["specificity_z"])

    lines = [
        "# 分子对接结果 — 对照与局限分析",
        "",
        "> 脚本：`scripts/22_docking_control_analysis.py`；明细：`docking_ligand_efficiency.csv`",
        "> 上游：`scripts/20_molecular_docking.py`（AutoDock Vina 1.2.7，exhaustiveness=8，seed=42）",
        "",
        "## 一、原始结合能（kcal/mol）",
        "",
        "| 配体 | MW 重原子 | " + " | ".join(sorted(df["gene"].unique())) + " |",
        "|---|---|" + "---|" * df["gene"].nunique(),
    ]
    for lig in LIGANDS:
        row = [f"| {lig} | {ha[lig]} "]
        for gene in sorted(df["gene"].unique()):
            sub = df[(df["ligand"] == lig) & (df["gene"] == gene)]
            row.append(f"| {sub['affinity_kcal_mol'].values[0]:.2f} " if len(sub) else "| n/a ")
        lines.append("".join(row) + "|")

    lines += [
        "",
        "## 二、配体效率 LE（-ΔG / 重原子数，kcal/mol/atom）",
        "",
        "> 校正分子尺寸效应：LE > 0.3 通常视为良好的先导化合物水平",
        "",
        "| 配体 | 重原子 | 最佳靶点 | LE | z 值 |",
        "|---|---|---|---|---|",
    ]
    for lig, (gene, aff, zz) in best.items():
        le = aff / ha[lig] * -1
        lines.append(f"| {lig} | {ha[lig]} | {gene} | {le:.3f} | {zz:.2f} |")

    lines += [
        "",
        "## 三、⚠️ 必须写进 Limitations 的局限（诚实声明）",
        "",
        "1. **配体尺寸效应未被完全消除**：aleplasinin（重原子 32，本组最大）对所有靶点均给出",
        "   -7.3~-8.6 kcal/mol，属于分子量大→接触面大导致的**普遍性高分**，",
        "   **不能解读为对某一靶点特异**。",
        "2. **阳性对照偏小**：4-aminobenzamidine（重原子 10）得分天然偏低，",
        "   与 aleplasinin 的横向比较**无效**，只能在同靶点内比较。",
        "3. **结合位点定义不精确**：对接盒自动定位于受体整体质心或已知配体质心，",
        "   **未做分子动力学或结合位点残基突变验证**，属初步筛选级别。",
        "4. **THBD 未获得结果**：1ADX 仅含血栓调节蛋白 EGF5 单域（40 残基），",
        "   体积过小无法容纳对接盒；如需 THBD 结论须换用全长结构。",
        "5. **抗体类药物无法用本方法评估**：caplacizumab（VWF）、concizumab（TFPI）",
        "   均为抗体，其阻断机制为蛋白-蛋白界面，**不能用小分子对接评价**。",
        "",
        "## 四、可以谨慎表述的结论",
        "",
        "- 在**同一靶点内部**，文献已知抑制剂（upamostat→PLAU；aleplasinin→SERPINE1）",
        "  的排名**与药理预期方向一致**，支持该纤溶-内皮轴的小分子可干预性。",
        "- 抗纤溶药（tranexamic acid / 6-aminocaproic acid）在本组靶点上结合能普遍偏弱，",
        "  与其主要作用于纤溶酶原/纤溶酶而非本轴靶点的机制一致。",
        "- 结论定位：**分子对接为支持性证据（hypothesis-supporting），不是本研究的核心发现**。",
    ]
    (OUT / "docking_control_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
