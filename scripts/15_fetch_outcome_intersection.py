#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_fetch_outcome_intersection.py — 为 6 交集基因提取结局 SNP 数据

复用 OpenGWAS API + 现有 JWT token，为 PLAT/PLAU/PLAUR/SERPINE1/VWF 的 pQTL SNP
提取三个结局（ieu-b-4980/5086, finngen）的关联数据，存为 _outcome_intersection_*.json
"""
import sys, os, json, ssl, urllib.request, csv, time
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

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLEAN = r"" + EXT_DIR + "/pqtl_decode"
WS = r"" + EXT_DIR + "/lactylation_coagulation_mr"

GENES = ["PLAT", "PLAU", "PLAUR", "SERPINE1", "VWF"]
OUTCOMES = ["ieu-b-4980", "ieu-b-5086", "finn-b-AB1_OTHER_SEPSIS"]

def main():
    tok = open(os.path.join(WS, "_opengwas_jwt.txt"), encoding="utf-8").read().strip()
    ctx = ssl._create_unverified_context()
    URL = "https://api.opengwas.io/api/associations"

    # 收集所有 6 交集基因的 pQTL SNP
    snps = []
    for g in GENES:
        rows = list(csv.DictReader(open(os.path.join(CLEAN, f"{g}.csv"), encoding="utf-8")))
        for r in rows:
            snps.append(r["SNP"])
    snps = list(dict.fromkeys(snps))  # 去重保序
    print(f"总 IV SNP 数: {len(snps)}")

    for gw in OUTCOMES:
        print(f"\n=== 结局 {gw} ===")
        outres = {}
        for i in range(0, len(snps), 50):
            batch = snps[i:i+50]
            q = "&variant=".join(batch)
            req = urllib.request.Request(f"{URL}?id={gw}&variant={q}", data=b"", method="POST",
                                         headers={"Authorization": f"Bearer {tok}"})
            items = []
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=180, context=ctx) as r:
                        items = json.loads(r.read().decode())
                    break
                except Exception as e:
                    print(f"  batch err {i} {type(e).__name__} {str(e)[:60]} retry", flush=True)
                    time.sleep(5)
            if isinstance(items, dict):
                items = items.get("result", []) if isinstance(items.get("result"), list) else []
            for it in items:
                if isinstance(it, dict) and it.get("rsid"):
                    outres[it["rsid"]] = it
            print(f"  {min(i+50,len(snps))}/{len(snps)} 完成, hits {len(outres)}", flush=True)
        out_path = os.path.join(CLEAN, f"_outcome_intersection_{gw}.json")
        json.dump(outres, open(out_path, "w"), indent=1)
        print(f"  保存 {out_path}, 总 hits {len(outres)}")

if __name__ == "__main__":
    main()
