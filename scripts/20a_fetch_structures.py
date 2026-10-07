#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20a_fetch_structures.py — 获取分子对接所需蛋白结构

靶点选择依据（来自 16_dgidb_intersection.py + 文献核实）：
  1. VWF  – caplacizumab（已获批 aTTP，VWF A1 结构域）
  2. THBD – 重组血栓调节蛋白（日本获批 DIC）
  3. PLAU – upamostat / uPA 抑制剂
  4. PLAUR– uPAR（MR 正向关联 28 天死亡，本研究的核心因果靶点）
  5. TFPI – concizumab（已获批血友病）
  6. SERPINE1 – PAI-1 抑制剂（aleplasinin）

结构来源优先级：
  a) RCSB PDB 实验结构（若有、且覆盖配体结合位点）
  b) AlphaFold DB 预测结构（UniProt 号）

输出：data/docking/<GENE>_<pdb|af>_<id>.pdb + structures_manifest.csv
"""
import json
import time
import urllib.request
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


OUT = Path(r"" + DATA_DIR + "/docking")
OUT.mkdir(parents=True, exist_ok=True)

# UniProt 号（人工核实，来源 UniProt）
TARGETS = {
    "VWF":      {"uniprot": "P04275", "pdb": None,  "note": "caplacizumab 靶点（A1 域）"},
    "THBD":     {"uniprot": "P07204", "pdb": "1ADX","note": "rhsTM（日本 DIC 获批）"},
    "PLAU":     {"uniprot": "P00749", "pdb": "1C5W","note": "uPA 催化域 + upamostat"},
    "PLAUR":    {"uniprot": "Q03405", "pdb": "3BT1","note": "uPAR（MR 因果靶点）"},
    "TFPI":     {"uniprot": "P10646", "pdb": "1TFX","note": "concizumab 靶点（K2 域）"},
    "SERPINE1": {"uniprot": "P05121", "pdb": "1B3K","note": "PAI-1，aleplasinin"},
}

def dl(url, out, retries=5):
    if out.exists() and out.stat().st_size > 2000:
        print(f"  [skip] {out.name}"); return True
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
                f.write(r.read())
            print(f"  [ok] {out.name} ({out.stat().st_size/1024:.0f} KB)"); return True
        except Exception as e:
            print(f"  [retry {i+1}] {out.name}: {e}"); time.sleep(4)
    return False

def main():
    manifest = []
    for gene, info in TARGETS.items():
        print(f"=== {gene} ({info['note']}) ===")
        got = None
        # 1) 尝试 RCSB 实验结构
        if info["pdb"]:
            fid = info["pdb"].lower()
            url = f"https://files.rcsb.org/download/{fid}.pdb"
            out = OUT / f"{gene}_pdb_{fid}.pdb"
            if dl(url, out):
                got = out.name
        # 2) 兜底 AlphaFold
        if got is None:
            up = info["uniprot"]
            url = f"https://alphafold.ebi.ac.uk/files/AF-{up}-F1-model_v4.pdb"
            out = OUT / f"{gene}_af_{up}.pdb"
            if dl(url, out):
                got = out.name
        manifest.append({"gene": gene, "uniprot": info["uniprot"],
                         "pdb_id": info["pdb"], "file": got, "note": info["note"]})

    import pandas as pd
    pd.DataFrame(manifest).to_csv(OUT / "structures_manifest.csv", index=False)
    print("\n=== manifest ===")
    for m in manifest:
        print(f"  {m['gene']:9s} {m['file']}")

if __name__ == "__main__":
    main()
