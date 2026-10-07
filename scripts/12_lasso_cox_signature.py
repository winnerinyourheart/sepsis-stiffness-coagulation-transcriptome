#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_lasso_cox_signature.py — LASSO-Cox 刚度-凝血联合预后签名

生信文章主体模块 3：
  用「刚度∩凝血交集基因 + 刚度/凝血通路基因」构建候选特征池，
  LASSO-Cox（sksurv CoxnetSurvivalAnalysis）在 GSE65682 训练，
  输出联合风险签名，评估 C-index 与 KM 分层。

输出：
  results/lasso_cox_signature_coef.csv
  results/lasso_cox_risk.csv
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

def load_gene_expr():
    expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz", compression="gzip", index_col=0)
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["ID", "Gene Symbol"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol"])
    annot["Gene Symbol"] = annot["Gene Symbol"].astype(str).str.strip()
    id2sym = annot.set_index("ID")["Gene Symbol"].to_dict()
    expr = expr.loc[expr.index.intersection(annot["ID"])]
    expr.index = expr.index.map(id2sym)
    return expr.groupby(expr.index).mean()

def main():
    expr = load_gene_expr()
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()

    # 候选特征池 = 刚度∩凝血 6 基因 + 凝血臂可测 19 + 刚度核心（Consensus13）
    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]
    inter = ["PLAT","PLAU","PLAUR","SERPINE1","THBD","VWF"]
    cons13 = ["ADAMTS13","EGR1","ELANE","EVL","KCNK4","KLF9","MAP4K2","PARD6A","PLOD3","PRRX2","TGFBI","TGIF1","ZEB1"]
    pool = sorted(set(inter + coag_genes + cons13) & set(expr.index))
    print(f"候选特征池: {len(pool)} 个基因")
    print(f"  = 交集6 + 凝血臂可测 + Consensus13")

    # 特征矩阵（z-score 标准化）
    X = expr.loc[pool, seps.index].T
    X = (X - X.mean()) / X.std()
    y = seps[["mortality_28d", "time_to_event_28d"]].copy()
    y.columns = ["event", "duration"]
    y["event"] = y["event"].astype(bool)
    y["duration"] = y["duration"].astype(float)

    # 结构化数组给 sksurv
    import sksurv
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    y_surv = Surv.from_dataframe("event", "duration", y)

    print("\n[1] LASSO-Cox 训练（5折CV选 alpha）...")
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import StratifiedKFold
    # 用宽 alpha 网格 + 交叉验证选最优
    alphas = np.logspace(-3, 0.5, 50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mean_c = []
    for a in alphas:
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[a])
        cs = []
        for tr, te in cv.split(X.values, y["event"]):
            model.fit(X.values[tr], y_surv[tr])
            cs.append(concordance_index_censored(
                y_surv["event"][te], y_surv["duration"][te], model.predict(X.values[te]))[0])
        mean_c.append(np.mean(cs))
    best = alphas[np.argmax(mean_c)]
    print(f"    最优 alpha = {best:.4f} (CV C-index = {max(mean_c):.4f})")

    model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[best])
    model.fit(X.values, y_surv)
    coef = model.coef_

    # 非零系数基因
    nonzero = [(pool[i], coef[i]) for i in range(len(pool)) if abs(coef[i]) > 1e-6]
    print(f"    入选基因数: {len(nonzero)}")
    coef_df = pd.DataFrame(nonzero, columns=["gene","coef"]).sort_values("coef", key=abs, ascending=False)
    coef_df.to_csv(f"{OUT_DIR}/lasso_cox_signature_coef.csv", index=False)
    print(coef_df.to_string(index=False))

    # 风险分数
    risk = np.ravel(X.values @ coef)
    seps["risk"] = risk

    # C-index
    c = concordance_index_censored(y_surv["event"], y_surv["duration"], risk)[0]
    print(f"\n[2] 样本内 C-index = {c:.4f}")

    # KM 分层（中位数）
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    med = np.median(risk)
    hi = seps[seps["risk"]>=med]; lo = seps[seps["risk"]<med]
    r = logrank_test(hi["time_to_event_28d"], lo["time_to_event_28d"],
                     hi["mortality_28d"], lo["mortality_28d"])
    print(f"    高危(n={len(hi)}) 死亡 {(hi['mortality_28d']==1).mean():.1%}, "
          f"低危(n={len(lo)}) 死亡 {(lo['mortality_28d']==1).mean():.1%}, log-rank p={r.p_value:.4f}")

    # 风险分数与象限/内型关系
    print(f"\n[3] 风险分数 vs 象限（高低 vs 低低）:")
    st_med = seps["stiffness_537_z"].median(); co_med = seps["coag_42_z"].median()
    ll = seps[(seps["stiffness_537_z"]<st_med)&(seps["coag_42_z"]<co_med)]
    hh = seps[(seps["stiffness_537_z"]>=st_med)&(seps["coag_42_z"]>=co_med)]
    print(f"    LL象限 risk均值={ll['risk'].mean():+.3f} vs HH象限={hh['risk'].mean():+.3f}")

    seps["risk"].to_csv(f"{OUT_DIR}/lasso_cox_risk.csv")
    print("\n完成。")

if __name__ == "__main__":
    main()
