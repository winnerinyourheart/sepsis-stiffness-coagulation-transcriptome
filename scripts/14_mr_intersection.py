#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_mr_intersection.py — 6 交集基因的 pQTL MR → 脓毒症结局

复用 lactylation_coagulation_mr 项目的 pQTL（deCODE）+ 结局（OpenGWAS/FinnGen）数据。
对刚度∩凝血交集基因（PLAT/PLAU/PLAUR/SERPINE1/VWF，THBD无pQTL跳过）做 MR，
方法：IVW + Egger + Weighted Median，F>10 过滤，palindromic EAF 规则。

输出：
  results/mr_intersection_results.csv
"""
import json, os, math, csv
import numpy as np
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


CLEAN = r"" + EXT_DIR + "/pqtl_decode"
OUT = r"" + RESULT_DIR + ""

GENES = ["PLAT", "PLAU", "PLAUR", "SERPINE1", "VWF"]  # 6交集里5个有pQTL
OUTCOMES = [
    ("ieu-b-4980", "Sepsis occurrence (UKB)"),
    ("ieu-b-5086", "28-day mortality (UKB)"),
    ("finn-b-AB1_OTHER_SEPSIS", "Sepsis (FinnGen)"),
]

def load_pqtl(gene):
    path = os.path.join(CLEAN, f"{gene}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows

def load_outcome(gw):
    path = os.path.join(CLEAN, f"_outcome_intersection_{gw}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))

def f_stat(bx, bx_se, eaf, n):
    if eaf is None or eaf <= 0 or eaf >= 1:
        return bx**2 / bx_se**2
    r2 = 2*bx**2*eaf*(1-eaf) / (2*bx**2*eaf*(1-eaf) + 2*bx_se**2*n*eaf*(1-eaf))
    return r2*(n-2)/(1-r2)

def harmonise(exp_rows, outc):
    """返回 harmonised [(bx, bx_se, by, by_se, snp)]"""
    h = []
    for r in exp_rows:
        o = outc.get(r["SNP"])
        if not o:
            continue
        # 处理 list 型（FinnGen）或 dict 型（OpenGWAS）
        if isinstance(o, list):
            matched = None
            for cand in o:
                res = harmonise_one(r, cand)
                if res: matched = res; break
            if matched: h.append(matched)
        else:
            res = harmonise_one(r, o)
            if res: h.append(res)
    # F>10 过滤
    h = [x for x in h if f_stat(x["bx"], x["bx_se"], x.get("eaf"), x.get("n")) > 10]
    return h

def harmonise_one(r, o):
    if "alt" in o:  # FinnGen
        ea_o, nea_o, b_o, se_o = o["alt"], o["ref"], o["beta"], o["se"]
    else:
        ea_o, nea_o, b_o, se_o = o["ea"], o["nea"], o["beta"], o["se"]
    if se_o is None or float(se_o) <= 0:
        return None
    a1, a2 = r["A1"].upper(), r["A2"].upper()
    ea_o, nea_o = ea_o.upper(), nea_o.upper()
    if a1 == ea_o and a2 == nea_o:
        b_e, b_o2 = float(r["BETA"]), float(b_o)
    elif a1 == nea_o and a2 == ea_o:
        b_e, b_o2 = float(r["BETA"]), -float(b_o)
    else:
        return None
    if {a1, a2} in ({"A","T"},{"C","G"}):
        eaf = r.get("EAF")
        if eaf not in (None,"") and (float(eaf) > 0.42 or float(eaf) < 0.08):
            return None
    eaf = float(r["EAF"]) if r.get("EAF") not in (None,"") else None
    n = float(r["N"]) if r.get("N") not in (None,"") else None
    return {"snp": r["SNP"], "bx": b_e, "bx_se": float(r["SE"]), "by": b_o2,
            "by_se": float(se_o), "eaf": eaf, "n": n}

def ivw(h):
    w = 1/np.array([x["by_se"] for x in h])**2
    bx = np.array([x["bx"] for x in h]); by = np.array([x["by"] for x in h])
    num = np.sum(w*bx*by); den = np.sum(w*bx**2)
    b = num/den; se = math.sqrt(1/den)
    return b, se

def egger(h):
    w = 1/np.array([x["by_se"] for x in h])**2
    bx = np.array([x["bx"] for x in h]); by = np.array([x["by"] for x in h])
    W = np.diag(w); X = np.vstack([np.ones(len(h)), bx]).T
    beta = np.linalg.inv(X.T@W@X) @ (X.T@W@by)
    return beta[1], math.sqrt(np.linalg.inv(X.T@W@X)[1,1]), beta[0]  # b, se, intercept

def weighted_median(h):
    bx = np.array([x["bx"] for x in h]); by = np.array([x["by"] for x in h])
    se = np.array([x["by_se"] for x in h])
    w = 1/se**2
    ratios = by/bx
    order = np.argsort(ratios)
    ratios = ratios[order]; w = w[order]
    cw = np.cumsum(w); total = cw[-1]
    idx = np.searchsorted(cw, total/2)
    return ratios[idx]

def z2p(z):
    return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))

def main():
    results = []
    for gw, lab in OUTCOMES:
        outc = load_outcome(gw)
        if not outc:
            print(f"结局 {gw} 数据缺失，跳过")
            continue
        print(f"\n{'='*70}\n结局: {lab} ({gw})")
        for g in GENES:
            exp = load_pqtl(g)
            if not exp:
                print(f"  {g}: pQTL 缺失")
                continue
            h = harmonise(exp, outc)
            if len(h) < 2:
                print(f"  {g}: harmonised IV < 2 (n={len(h)})")
                results.append({"outcome": gw, "gene": g, "nsnp": len(h), "method": "IVW",
                                "b": None, "se": None, "OR": None, "lo": None, "hi": None, "p": None})
                continue
            b_ivw, se_ivw = ivw(h)
            orr = math.exp(b_ivw); lo = math.exp(b_ivw-1.96*se_ivw); hi = math.exp(b_ivw+1.96*se_ivw)
            p = z2p(b_ivw/se_ivw)
            # Egger
            b_e, se_e, intercept = egger(h)
            p_egger = z2p(b_e/se_e)
            print(f"  {g:10s} IVW OR={orr:.2f} [{lo:.2f}-{hi:.2f}] p={p:.3g} | Egger p={p_egger:.3g} | n_IV={len(h)}")
            results.append({"outcome": gw, "gene": g, "nsnp": len(h), "method": "IVW",
                            "b": b_ivw, "se": se_ivw, "OR": orr, "lo": lo, "hi": hi, "p": p,
                            "egger_p": p_egger, "egger_intercept": intercept})

    df = pd.DataFrame(results) if False else None
    import pandas as pd
    pd.DataFrame(results).to_csv(f"{OUT}/mr_intersection_results.csv", index=False)
    print(f"\n保存 -> {OUT}/mr_intersection_results.csv")

if __name__ == "__main__":
    main()
