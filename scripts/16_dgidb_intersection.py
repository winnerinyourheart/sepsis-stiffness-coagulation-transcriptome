#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_dgidb_intersection.py — 6 交集基因 + LASSO 签名基因的药物靶点预测

DGIdb GraphQL API（无需 token）。查询刚度∩凝血 6 交集基因
（PLAT/PLAU/PLAUR/SERPINE1/THBD/VWF）+ LASSO-Cox 签名里的关键凝血/刚度基因，
识别可药物靶向的基因及其抑制剂/激活剂。

输出：
  results/dgidb_intersection.csv
"""
import json, urllib.request
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


OUT = r"" + RESULT_DIR + ""

# 6 交集基因 + LASSO 签名里的凝血相关基因
GENES = ["PLAT", "PLAU", "PLAUR", "SERPINE1", "THBD", "VWF",
         "F5", "F13A1", "TFPI", "SERPINE2", "SERPINF2", "PLAT", "F12", "SERPINA1",
         "ADAMTS13", "EGR1", "ELANE", "KLF9"]

def query_dgidb(genes):
    QL = """{ genes(names: %s) { nodes { name interactions { drug { name } interactionTypes { type } } } } }"""
    qy = QL % json.dumps(genes)
    req = urllib.request.Request("https://dgidb.org/api/graphql",
                                 data=json.dumps({"query": qy}).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "replace"))
    return res

def main():
    res = query_dgidb(GENES)
    rec = []
    for node in res["data"]["genes"]["nodes"]:
        if not node.get("interactions"):
            continue
        for it in node["interactions"]:
            types = ";".join(sorted({t["type"] for t in it["interactionTypes"]})) or "NA"
            rec.append(dict(gene=node["name"], drug=it["drug"]["name"], type=types))

    df = pd.DataFrame(rec).drop_duplicates() if rec else pd.DataFrame(columns=["gene","drug","type"])
    df.to_csv(f"{OUT}/dgidb_intersection.csv", index=False)

    # 标记方向
    INH = {"inhibitor", "antagonist", "blocker", "negative modulator", "antibody", "neutralizer",
           "inhibitory allosteric modulator", "suppressor"}
    ACT = {"activator", "agonist", "positive modulator", "stimulator"}
    df["dir"] = df["type"].str.lower().apply(
        lambda t: "inhibitor" if any(k in t for k in INH)
        else ("activator" if any(k in t for k in ACT) else "other"))

    print(f"=== DGIdb 结果：{len(df)} 个药物-基因对 ===")
    print(f"有药物靶点的基因: {df['gene'].nunique()}")
    print()
    for g in GENES:
        sub = df[df["gene"]==g]
        if len(sub):
            inh = sub[sub["dir"]=="inhibitor"]["drug"].tolist()
            act = sub[sub["dir"]=="activator"]["drug"].tolist()
            print(f"  {g:10s} (总{len(sub)}):")
            if inh:
                print(f"      抑制剂: {', '.join(inh[:8])}")
            if act:
                print(f"      激活剂: {', '.join(act[:5])}")

if __name__ == "__main__":
    main()
