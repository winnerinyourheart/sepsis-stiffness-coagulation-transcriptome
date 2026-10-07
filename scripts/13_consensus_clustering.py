#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_consensus_clustering.py — 共识聚类（刚度+凝血联合基因）

生信文章主体模块 4：
  用「刚度∩凝血 6 交集 + 凝血臂可测 19」联合基因做 consensus clustering，
  识别脓毒症亚型，验证"低刚度-低凝血"亚型与死亡/Mars1 的关系。

方法：手写简单 consensus（K-means 多次 bootstrap + 共识矩阵），
     因样本量 479，用 sklearn KMeans + 重复采样。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def consensus_kmeans(X, k, n_rep=100, seed=42):
    """简单共识聚类：bootstrap 重采样 + KMeans，构建共识矩阵。"""
    n = X.shape[0]
    co_occ = np.zeros((n, n))
    rng = np.random.RandomState(seed)
    for _ in range(n_rep):
        idx = rng.choice(n, int(n*0.8), replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X[idx])
        labels = km.labels_
        for i in range(len(idx)):
            for j in range(len(idx)):
                if labels[i] == labels[j]:
                    co_occ[idx[i], idx[j]] += 1
    # 归一化
    diag = np.diag(co_occ).copy()
    co_occ = co_occ / (diag[:, None] + 1e-9)
    np.fill_diagonal(co_occ, 0)
    # 最终聚类（用共识矩阵做谱聚类或层次聚类）
    from sklearn.cluster import AgglomerativeClustering
    dist = 1 - co_occ
    cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit(dist)
    return cl.labels_, co_occ

def main():
    expr = load_gene_expr()
    scores = pd.read_csv(f"{OUT_DIR}/scores_gse65682.csv", index_col=0)
    seps = scores[scores["is_sepsis"]].copy()

    kegg = json.load(open(f"{MR_DIR}/kegg_hsa04610_arms_authoritative.json"))
    coag_genes = kegg["coagulation_arm"]
    inter = ["PLAT","PLAU","PLAUR","SERPINE1","THBD","VWF"]
    genes = sorted(set(inter + coag_genes) & set(expr.index))
    print(f"聚类基因: {len(genes)} 个（交集6 + 凝血臂可测）")

    X = expr.loc[genes, seps.index].T
    X = (X - X.mean()) / X.std()

    print("\n=== 共识聚类 K=2~4 的 silhouette ===")
    best_k = 2; best_sil = -1
    for k in [2, 3, 4]:
        labels, co = consensus_kmeans(X.values, k, n_rep=50)
        sil = silhouette_score(X.values, labels)
        print(f"  K={k}: silhouette = {sil:.4f}")
        if sil > best_sil:
            best_sil = sil; best_k = k

    print(f"\n最优 K = {best_k}")
    labels, co = consensus_kmeans(X.values, best_k, n_rep=100)
    seps["cluster"] = labels

    # 各簇的特征
    print(f"\n=== 各簇特征 ===")
    for c in sorted(seps["cluster"].unique()):
        sub = seps[seps["cluster"]==c]
        print(f"  簇{c}: n={len(sub)}, 死亡={(sub['mortality_28d']==1).mean():.1%}, "
              f"刚度z={sub['stiffness_537_z'].mean():+.3f}, 凝血z={sub['coag_42_z'].mean():+.3f}, "
              f"Mars1占={(sub['endotype_class']=='Mars1').mean():.1%}")

    # 簇间生存差异
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    groups = [seps[seps["cluster"]==c] for c in sorted(seps["cluster"].unique())]
    if len(groups) == 2:
        r = logrank_test(groups[0]["time_to_event_28d"], groups[1]["time_to_event_28d"],
                         groups[0]["mortality_28d"], groups[1]["mortality_28d"])
        print(f"\n两簇 log-rank p = {r.p_value:.4f}")

    seps[["cluster"]].to_csv(f"{OUT_DIR}/consensus_clusters.csv")
    print("\n完成。")

if __name__ == "__main__":
    main()
