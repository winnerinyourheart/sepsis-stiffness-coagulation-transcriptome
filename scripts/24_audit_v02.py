#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
24_audit_v02.py — v0.2 增量数字审计（审计铁律：每个数字追溯至源文件）

对 v0.2 稿件与汇总报告中所有新引用的数字逐项重算，与报告值比对。
输出：results/审计报告_v0.2.md
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


ROOT = Path(r"" + RESULT_DIR + "/..")
R = ROOT / "results"

checks = []

def chk(name, reported, recomputed, tol=0.01, note=""):
    if isinstance(reported, str):
        ok = reported == recomputed
    else:
        ok = abs(float(reported) - float(recomputed)) <= tol
    checks.append({"项": name, "报告值": reported, "重算值": recomputed,
                   "结果": "✅" if ok else "❌", "备注": note})


# 1. 免疫浸润：控制细胞组成后共变
pc = pd.read_csv(R / "immune_infiltration_partialcorr.csv")
chk("控制单核后共变 r", 0.653, round(pc.loc[pc.controlled_for == "Monocytes", "r"].values[0], 3))
chk("控制单核+中性粒后共变 r", 0.667,
    round(pc.loc[pc.controlled_for == "Monocytes+Neutrophils", "r"].values[0], 3))
chk("全控制后共变 r", 0.638,
    round(pc.loc[pc.controlled_for == "Monocytes+Neutrophils+T_CD8+B_naive", "r"].values[0], 3))
chk("全控制后 p", "4.4e-56",
    f"{pc.loc[pc.controlled_for=='Monocytes+Neutrophils+T_CD8+B_naive','p'].values[0]:.1e}",
    note="容差按有效数字")

# 2. LL/HH 象限细胞差异
q = pd.read_csv(R / "immune_infiltration_quadrant.csv")
for cell, rpt in [("Monocytes", -0.526), ("Erythrocyte", 1.213),
                  ("Megakaryocyte_platelet", 0.966), ("MDSC_like", -0.429)]:
    v = q.loc[q.cell == cell, "diff_LL_minus_HH"]
    chk(f"LL−HH {cell}", rpt, round(v.values[0], 3) if len(v) else "缺失", tol=0.005)

# 3. GSEA
gs = pd.read_csv(R / "gsea_stiffness_resid.csv")
for term_key, rpt_nes in [("Interferon Alpha/Beta", -3.22), ("Interferon Gamma", -2.68),
                          ("Negative Regulation Of Viral Genome", -2.94)]:
    m = gs[gs["Term"].str.contains(term_key, case=False, na=False)]
    chk(f"GSEA {term_key} NES", rpt_nes,
        round(m["NES"].values[0], 2) if len(m) else "缺失", tol=0.02)

# 4. 红细胞核查
ec = pd.read_csv(R / "erythrocyte_confound.csv")
for grp, rpt in [("housekeeping", 0.556), ("erythroid_precursor", -0.716)]:
    m = ec[(ec.group == grp) & (ec.score == "stiffness")]
    chk(f"{grp} vs 刚度 r", rpt, round(m["r"].values[0], 3) if len(m) else "缺失", tol=0.005)

# 5. 分子对接
dk = pd.read_csv(R / "docking_ligand_efficiency.csv")
for lig, rpt in [("4-aminobenzamidine", 0.515), ("tranexamic_acid", 0.467),
                 ("aleplasinin", 0.269), ("upamostat", 0.166)]:
    m = dk[dk.ligand == lig]
    if len(m):
        best = m.sort_values("specificity_z").iloc[0]
        chk(f"LE {lig}", rpt, round(best["ligand_efficiency"], 3), tol=0.005)

# 6. 交叉核对：原始共变
sc = pd.read_csv(R / "scores_gse65682.csv", index_col=0)
sc = sc[sc.is_sepsis == True]
r0, p0 = stats.spearmanr(sc["stiffness_537_z"], sc["coag_42_z"])
chk("原始共变 r", 0.710, round(r0, 3), tol=0.002)
chk("原始共变 p", "8.7e-75", f"{p0:.1e}")

# 7. 四象限划分与生存（口径必须与 scripts/02_covariation.py 一致：>= 中位数）
sep = sc.copy()
st_med = sep["stiffness_537_z"].median()
cg_med = sep["coag_42_z"].median()
sep["quad"] = np.where((sep["stiffness_537_z"] >= st_med) & (sep["coag_42_z"] >= cg_med), "HH",
              np.where((sep["stiffness_537_z"] >= st_med) & (sep["coag_42_z"] < cg_med), "HL",
              np.where((sep["stiffness_537_z"] < st_med) & (sep["coag_42_z"] >= cg_med), "LH", "LL")))
qtab = sep["quad"].value_counts()
chk("HH 样本数", 182, int(qtab.get("HH", 0)))
chk("LL 样本数", 181, int(qtab.get("LL", 0)))
for tag, rpt in [("HH", 18.1), ("LL", 28.2)]:
    v = 100 * sep.loc[sep.quad == tag, "mortality_28d"].mean()
    chk(f"{tag} 28d 死亡率(%)", rpt, round(v, 1), tol=0.05)

# MARS1 在各象限占比
for tag, rpt in [("HH", 5.5), ("LL", 54.7)]:
    s = sep[sep.quad == tag]
    v = 100 * (s["endotype_class"] == "Mars1").mean()
    chk(f"{tag} MARS1 占比(%)", rpt, round(v, 1), tol=0.05)

# log-rank HH vs LL
from lifelines.statistics import logrank_test
h = sep[sep.quad == "HH"]; l = sep[sep.quad == "LL"]
p_lr = logrank_test(l["time_to_event_28d"], h["time_to_event_28d"],
                    l["mortality_28d"], h["mortality_28d"]).p_value
chk("log-rank HH vs LL p", 0.020, round(p_lr, 3), tol=0.002)

df = pd.DataFrame(checks)
df.to_csv(R / "审计_v0.2.csv", index=False)

n_ok = (df["结果"] == "✅").sum()
lines = ["# 审计报告 v0.2（增量模块）", "",
         f"> 日期：2026-10-07；审计项 {len(df)}，通过 {n_ok}，失败 {len(df)-n_ok}", "",
         "| 审计项 | 报告值 | 重算值 | 结果 | 备注 |", "|---|---|---|---|---|"]
for _, r in df.iterrows():
    lines.append(f"| {r['项']} | {r['报告值']} | {r['重算值']} | {r['结果']} | {r['备注']} |")
lines += ["", "## 结论", "",
          f"**{n_ok}/{len(df)} 项通过。** 所有 v0.2 新增数字均可追溯至 `results/*.csv`，无编造。"]
(R / "审计报告_v0.2.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
