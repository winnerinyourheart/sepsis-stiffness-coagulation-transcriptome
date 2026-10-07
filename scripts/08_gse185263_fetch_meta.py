#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_gse185263_fetch_meta.py — 批量拉取 GSE185263 样本结局注释

从 GEO 拉取每个 GSM 样本的 "in hospital mortality" 和 "sofa" 字段，
生成 样本 -> outcome 映射表。
"""
import subprocess
import pandas as pd
import time
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


VAL_DIR = r"" + DATA_DIR + "/validation"

def fetch_one(gsm):
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}&targ=self&form=text&view=brief"
    try:
        out = subprocess.run(["curl", "-sL", "--max-time", "30", url],
                             capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return None
    meta = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("!Sample_title"):
            meta["title"] = line.split("=",1)[1].strip()
        elif "in hospital mortality" in line:
            meta["mortality"] = line.split(":",1)[1].strip()
        elif line.startswith("!Sample_characteristics") and "sofa" in line.lower():
            meta["sofa"] = line.split(":",1)[1].strip()
    return meta

def main():
    samples = [l.strip() for l in open(f"{VAL_DIR}/gse185263_samples.txt") if l.strip()]
    rows = []
    for i, gsm in enumerate(samples):
        meta = fetch_one(gsm)
        if meta is None:
            continue
        rows.append({"gsm": gsm, "title": meta.get("title",""),
                     "mortality": meta.get("mortality",""), "sofa": meta.get("sofa","")})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(samples)} 完成")
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df.to_csv(f"{VAL_DIR}/gse185263_metadata.csv", index=False)
    print(f"\n完成。共 {len(df)} 个样本注释。")
    print(df["mortality"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
