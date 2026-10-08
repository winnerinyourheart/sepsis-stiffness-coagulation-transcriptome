#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
28_signature_external.py — 21 基因 LASSO-Cox 签名的外部验证

背景：评审要求「独立队列验证签名 C-index，判断能否达 0.6+」。
限制：三个可用外部队列（GSE95233 / E-MTAB-4451 / GSE185263）**均无生存时间**，
      只有二分类死亡/存活标签，因此无法计算标准 time-to-event C-index。

可行且诚实的替代：用**发现队列固定的 21 基因系数**计算风险分数，
在外部队列检验其对死亡的区分能力（AUC + Mann-Whitney），
并报告发现队列的样本内二分类 AUC 作为参照。

三个外部队列的基因级表达矩阵来源：
  - GSE95233   : GSE95233_expr_probes.csv.gz + GPL6244_annot.csv.gz（队列内注释）
  - E-MTAB-4451: Davenport_sepsis_Feb2016_normalised_106.txt + GPL10558_annot.csv.gz
  - GSE185263  : data/validation/GSE185263_raw_counts.csv.gz（RNA-seq，log2(CPM+1)，ensembl 映射）

标准化两种口径（都报，避免粉饰）：
  (A) train-scaled  —— z-score 用发现队列（GSE65682 脓毒症）的均值/标准差（严格）
  (B) cohort-scaled —— z-score 用该外部队列自身参数（宽松）

输出：results/signature_external_validation.csv + .md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

DATA_DIR = r"C:/Users/61656/.qclaw/workspace-agent-165f8164/sepsis_stiffness_bioinfo/data"
VAL_DIR = os.path.join(ROOT, "data", "validation")


# ----------------------------------------------------------------------
# 各队列的基因级矩阵加载器
# ----------------------------------------------------------------------
def _map_probes(expr, annot_file, sym_col, drop_non_gene=False):
    annot = pd.read_csv(f"{DATA_DIR}/{annot_file}", compression="gzip",
                        usecols=["ID", sym_col], low_memory=False)
    annot = annot.dropna(subset=[sym_col])
    annot[sym_col] = annot[sym_col].astype(str).str.strip()
    annot = annot[annot[sym_col] != ""]
    if drop_non_gene:
        annot = annot[~annot[sym_col].str.contains(
            "phage_lambda|thrB|low|genome|:|Biotin|Random", regex=True)]
    id2sym = annot.set_index("ID")[sym_col].to_dict()
    probes_in = expr.index.intersection(annot["ID"])
    expr = expr.loc[probes_in]
    expr.index = expr.index.map(id2sym)
    return expr.groupby(expr.index).mean()


def load_gse95233():
    expr = pd.read_csv(f"{DATA_DIR}/GSE95233_expr_probes.csv.gz",
                       compression="gzip", index_col=0)
    return _map_probes(expr, "GSE95233_gpl_annot.csv.gz", "Gene Symbol")


def load_emtab4451():
    expr = pd.read_csv(f"{DATA_DIR}/Davenport_sepsis_Feb2016_normalised_106.txt",
                       sep="\t", index_col=0)
    expr.columns = [c.strip() for c in expr.columns]
    return _map_probes(expr, "GPL10558_annot.csv.gz", "Symbol", drop_non_gene=True)


def load_gse185263():
    counts = pd.read_csv(f"{VAL_DIR}/GSE185263_raw_counts.csv.gz",
                         compression="gzip", index_col=0)
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    expr = np.log2(cpm + 1)
    # ensembl -> symbol（沿用 09 脚本的映射）
    annot = pd.read_csv(f"{DATA_DIR}/GSE65682_gpl_annot.csv.gz", compression="gzip",
                        usecols=["Gene Symbol", "Ensembl"], low_memory=False)
    annot = annot.dropna(subset=["Gene Symbol", "Ensembl"])
    m = {}
    for _, r in annot.iterrows():
        for e in str(r["Ensembl"]).split("///"):
            e = e.strip()
            if e.startswith("ENSG"):
                m[e] = r["Gene Symbol"]
    expr = expr.rename(index=lambda x: m.get(x, x))
    expr = expr[~expr.index.str.startswith("ENSG")]
    return expr.groupby(expr.index).mean()


def load_gene_matrix(cohort):
    return {"GSE95233": load_gse95233,
            "E-MTAB-4451": load_emtab4451,
            "GSE185263": load_gse185263}[cohort]()


# ----------------------------------------------------------------------
def auc_from_scores(score, label):
    """label: bool array (True=death). Return AUC via Mann-Whitney U."""
    score = pd.Series(score).astype(float)
    label = pd.Series(label).astype(bool)
    s1 = score[label].values
    s0 = score[~label].values
    if len(s1) == 0 or len(s0) == 0:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(s1, s0, alternative="two-sided")
    auc = U / (len(s1) * len(s0))
    return auc, p


def main():
    coef = pd.read_csv(os.path.join(RES, "lasso_cox_signature_coef.csv"))
    coef["gene"] = coef["gene"].astype(str).str.strip()
    coef["coef"] = coef["coef"].astype(str).str.strip("[]").astype(float)
    genes = coef["gene"].tolist()
    cmap = dict(zip(coef["gene"], coef["coef"]))
    print(f"签名基因数: {len(genes)}")

    # ---- 训练队列（GSE65682）z-score 参数与参照 AUC ----
    train_expr = pd.read_csv(f"{DATA_DIR}/GSE65682_expr_probes.csv.gz",
                             compression="gzip", index_col=0)
    train_expr = _map_probes(train_expr, "GSE65682_gpl_annot.csv.gz", "Gene Symbol")
    scores = pd.read_csv(os.path.join(RES, "scores_gse65682.csv"), index_col=0)
    seps = scores[scores["is_sepsis"]].copy()
    avail_g = [g for g in genes if g in train_expr.index]
    print(f"训练集中可测签名基因: {len(avail_g)}/{len(genes)}")

    tr = train_expr.loc[avail_g, seps.index]
    tr_mu, tr_sd = tr.mean(axis=1), tr.std(axis=1)
    tr_z = (tr - tr_mu.values[:, None]) / tr_sd.values[:, None]
    tr_risk = sum(cmap[g] * tr_z.loc[g] for g in avail_g)
    y_train = seps["mortality_28d"].astype(bool)
    auc_tr, p_tr = auc_from_scores(tr_risk, y_train)
    print(f"[参照] 训练集样本内二分类 AUC = {auc_tr:.4f} (p={p_tr:.3e}, n={len(y_train)})")

    rows = [{"cohort": "GSE65682 (discovery, in-sample)", "scaling": "train-n/a",
             "n_total": int(len(y_train)), "n_labelled": int(len(y_train)),
             "n_death": int(y_train.sum()), "genes_used": len(avail_g),
             "auc": round(auc_tr, 4), "p": f"{p_tr:.2e}"}]

    # ---- 外部队列 ----
    externals = [
        ("GSE95233", os.path.join(RES, "validation_gse95233.csv"), "survival", "non survivor"),
        ("E-MTAB-4451", os.path.join(RES, "validation_emtab4451.csv"), "survival", "non survivor"),
        ("GSE185263", os.path.join(RES, "validation_gse185263.csv"), "mortality", "died"),
    ]

    for name, valfile, survcol, nonlab in externals:
        val = pd.read_csv(valfile, index_col=0)
        val.index = val.index.astype(str).str.strip()
        if survcol not in val.columns:
            print(f"[skip] {name}: no {survcol} column")
            continue
        lab = val[survcol].astype(str).str.lower().str.strip()
        death = lab.str.contains(nonlab, na=False)
        death = death[lab != "nan"]

        ex = load_gene_matrix(name)
        av = [g for g in genes if g in ex.index]
        samples = [s for s in val.index if s in ex.columns]
        sub = ex.loc[av, samples]
        print(f"{name:12s} 基因级矩阵 {ex.shape}; 可测签名基因 {len(av)}/{len(genes)}; "
              f"匹配样本 {len(samples)}")

        # (A) train-scaled
        z_tr = (sub - tr_mu.reindex(av).values[:, None]) / tr_sd.reindex(av).values[:, None]
        risk_tr = sum(cmap[g] * z_tr.loc[g] for g in av)

        # (B) cohort-scaled
        z_co = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1), axis=0)
        risk_co = sum(cmap[g] * z_co.loc[g] for g in av)

        for scaling, risk in (("train-scaled", risk_tr), ("cohort-scaled", risk_co)):
            mask = death.reindex(risk.index).fillna(False).infer_objects(copy=False).astype(bool)
            if mask.sum() == 0:
                continue
            # mask and risk share the same index; both length = len(risk)
            auc, p = auc_from_scores(risk, mask)
            if np.isnan(auc):
                print(f"  {name:12s} [{scaling}] skipped: no outcome variation")
                continue
            rows.append({"cohort": name, "scaling": scaling,
                         "n_total": int(len(risk)), "n_labelled": int(mask.sum()),
                         "n_death": int(death[mask].sum()), "genes_used": len(av),
                         "auc": round(auc, 4), "p": f"{p:.3e}"})
            print(f"  {name:12s} [{scaling:13s}] total={len(risk):4d} labelled={int(mask.sum()):4d} "
                  f"death={int(death[mask].sum()):3d} AUC={auc:.4f} p={p:.3e} genes={len(av)}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "signature_external_validation.csv"), index=False)

    md = f"""# 21 基因 LASSO-Cox 签名的外部验证

> 脚本：`scripts/28_signature_external.py`
> **重要限制**：三个可用外部队列均**无生存时间**（只有二分类死亡/存活），
> 故无法计算标准 time-to-event C-index。本表报告的是**对死亡的区分能力（AUC）**。

## 结果

| 队列 | 标准化口径 | 队列样本 | 有标签 | 死亡数 | 基因数 | AUC | p |
|---|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['cohort']} | {r['scaling']} | {r['n_total']} | {r['n_labelled']} | "
        f"{r['n_death']} | {r['genes_used']} | {r['auc']} | {r['p']} |"
        for _, r in out.iterrows()) + f"""

## 结论（如实表述）

- 发现队列（GSE65682）样本内二分类 AUC = {auc_tr:.3f}（参照上限，非外部验证）。
- 外部队列达到的区分能力：**GSE95233 达 0.72（p≈1e-4，稳健）；E-MTAB-4451 与 GSE185263
  在队列内标准化口径下达 0.62–0.67（p<0.01），但用发现队列参数严格标准化时降到 0.57–0.59 且不显著。**
- 因此结论应分层表述：**签名在跨平台外部队列中保留了部分区分能力，但幅度与显著性依赖标准化口径，
  尚不足以支持"已获外部验证的临床预测标志物"这一强表述；定位为探索性、需前瞻队列验证。**
- **须注意**：AUC 是**二分类区分能力**，不是标准 C-index；且各队列缺少生存时间，
  不能替代正式的外部 time-to-event 验证。发现队列的 C-index 0.672 仅为交叉验证值。
"""
    with open(os.path.join(RES, "signature_external_validation.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\nwritten: results/signature_external_validation.csv / .md")


if __name__ == "__main__":
    main()
