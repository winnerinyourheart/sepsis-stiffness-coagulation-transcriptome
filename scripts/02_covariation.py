#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_covariation.py — Layer 1 核心分析：刚度 × 凝血 共变 + 四象限生存分层

在 GSE65682 脓毒症 n=479 内：
  1. stiffness_537_z 与 coag_42_z 的 Spearman 相关（主假设检验）
  2. 阴性对照：inflammation_20_z 与两分数相关（排除炎症背景噪声）
  3. 四象限分组（刚度高/低 × 凝血高/低，中位数切分）→ 28d 死亡 + log-rank
  4. 单变量 Cox（各分数连续 → 死亡），校正 age+sex 可选

输出：
  results/covariation_corr.csv      相关性矩阵
  results/quadrant_survival.csv     四象限死亡统计
  results/cox_univariate.csv        单变量 Cox
"""
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
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


OUT_DIR = r"" + RESULT_DIR + ""

def main():
    df = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = df[df["is_sepsis"]].copy()
    print(f"脓毒症样本 n = {len(seps)}, 死亡 = {(seps['mortality_28d']==1).sum()}")

    # ---------- 1. 相关性 ----------
    zcols = ["stiffness_537_z", "coag_42_z", "inflammation_20_z"]
    rows = []
    for a in zcols:
        for b in zcols:
            r, p = stats.spearmanr(seps[a], seps[b])
            rows.append({"x": a, "y": b, "spearman_r": r, "p": p})
    corr = pd.DataFrame(rows)
    corr.to_csv(f"{OUT_DIR}/covariation_corr.csv", index=False)
    print("\n=== 相关性（Spearman, 脓毒症 n=479） ===")
    print(corr[corr["x"] < corr["y"]].round(4).to_string(index=False))

    # 核心：刚度 vs 凝血
    r, p = stats.spearmanr(seps["stiffness_537_z"], seps["coag_42_z"])
    print(f"\n★★ 核心假设：stiffness vs coag  Spearman r={r:.4f}, p={p:.3e}")

    # ---------- 2. 四象限 ----------
    st_med = seps["stiffness_537_z"].median()
    co_med = seps["coag_42_z"].median()
    seps["quad"] = np.where(
        (seps["stiffness_537_z"] >= st_med) & (seps["coag_42_z"] >= co_med), "HH",
        np.where((seps["stiffness_537_z"] >= st_med) & (seps["coag_42_z"] < co_med), "HL",
        np.where((seps["stiffness_537_z"] < st_med) & (seps["coag_42_z"] >= co_med), "LH", "LL")))

    quad_rows = []
    for q in ["HH", "HL", "LH", "LL"]:
        sub = seps[seps["quad"] == q]
        mort = (sub["mortality_28d"] == 1).mean()
        quad_rows.append({"quadrant": q, "n": len(sub), "n_death": int((sub['mortality_28d']==1).sum()),
                          "mort_rate": mort})
    quad = pd.DataFrame(quad_rows)
    quad.to_csv(f"{OUT_DIR}/quadrant_survival.csv", index=False)
    print("\n=== 四象限 28d 死亡率 ===")
    print(quad.round(4).to_string(index=False))

    # log-rank：HH vs LL（最极端对比）
    t_hh = seps[seps["quad"]=="HH"]["time_to_event_28d"]
    e_hh = seps[seps["quad"]=="HH"]["mortality_28d"]
    t_ll = seps[seps["quad"]=="LL"]["time_to_event_28d"]
    e_ll = seps[seps["quad"]=="LL"]["mortality_28d"]
    res = logrank_test(t_hh, t_ll, e_hh, e_ll)
    print(f"\nlog-rank HH vs LL: p={res.p_value:.4f}")

    # 高刚度 vs 低刚度（单看刚度）
    hi = seps[seps["stiffness_537_z"]>=st_med]; lo = seps[seps["stiffness_537_z"]<st_med]
    r2 = logrank_test(hi["time_to_event_28d"], lo["time_to_event_28d"],
                      hi["mortality_28d"], lo["mortality_28d"])
    print(f"log-rank 高刚度 vs 低刚度: p={r2.p_value:.4f}")

    # 高凝血 vs 低凝血
    hc = seps[seps["coag_42_z"]>=co_med]; lc = seps[seps["coag_42_z"]<co_med]
    r3 = logrank_test(hc["time_to_event_28d"], lc["time_to_event_28d"],
                      hc["mortality_28d"], lc["mortality_28d"])
    print(f"log-rank 高凝血 vs 低凝血: p={r3.p_value:.4f}")

    # ---------- 3. 单变量 Cox ----------
    from lifelines import CoxPHFitter
    cdf = seps[["time_to_event_28d", "mortality_28d", "stiffness_537_z", "coag_42_z",
                "inflammation_20_z"]].copy()
    cdf.columns = ["duration", "event", "stiffness_z", "coag_z", "inflam_z"]
    cph = CoxPHFitter()
    cox_rows = []
    for var in ["stiffness_z", "coag_z", "inflam_z"]:
        cph.fit(cdf[["duration", "event", var]], duration_col="duration", event_col="event")
        cox_rows.append({"var": var, "HR_per_SD": np.exp(cph.params_[var]),
                         "p": cph.summary.loc[var, "p"]})
    cox = pd.DataFrame(cox_rows)
    cox.to_csv(f"{OUT_DIR}/cox_univariate.csv", index=False)
    print("\n=== 单变量 Cox（HR per SD） ===")
    print(cox.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
