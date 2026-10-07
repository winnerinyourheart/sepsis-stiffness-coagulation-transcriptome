# -*- coding: utf-8 -*-
"""
25_make_figures.py — 生成 Fig.1–Fig.5（复用 results/*.csv 真实数据，零编造）

输出：figures/Fig1.png ... Fig5.png（300 dpi，投稿级）
数据源：
  Fig1  scores_gse65682.csv
  Fig2  scores_gse65682.csv + mhla_dr_gse65682.csv + immune_infiltration_quadrant.csv
  Fig3  scores_gse65682.csv + lasso_cox_risk.csv
  Fig4  intersection_genes.csv + dgidb_intersection.csv + mr_intersection_results.csv
  Fig5  immune_infiltration_partialcorr.csv + gsea_{stiffness,coag}_resid.csv + erythrocyte_confound.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "figure.dpi": 300,
})
RED = "#c0392b"      # 升/高风险
BLUE = "#2471a3"     # 降/低风险
GREY = "#7f8c8d"
GREEN = "#1e8449"

# ------------------------------------------------------------------ 数据载入
sc = pd.read_csv(os.path.join(RES, "scores_gse65682.csv"), index_col=0)
sep = sc[sc["is_sepsis"] == True].copy()
mh = pd.read_csv(os.path.join(RES, "mhla_dr_gse65682.csv"), index_col=0)
sep = sep.join(mh, how="left")

# ================================================================== Fig 1
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
ax = axes[0]
x = sep["stiffness_537_z"].values
y = sep["coag_42_z"].values
r, p = stats.spearmanr(x, y)
ax.scatter(x, y, s=6, c=BLUE, alpha=0.45, linewidths=0)
m, b = np.polyfit(x, y, 1)
xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
ax.plot(xs, m * xs + b, color=RED, lw=1.4)
ax.set_xlabel("Matrix-stiffness score (z)")
ax.set_ylabel("Coagulation score (z)")
ax.set_title("A  Stiffness \u00d7 coagulation co-variation", loc="left", fontsize=9)
ax.text(0.04, 0.94, f"Spearman r = {r:.3f}\np = {p:.1e}\nn = {len(sep)}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(fc="white", ec=GREY, lw=0.5, alpha=0.9))

ax = axes[1]
xi = sep["inflammation_20_z"].values
yi = sep["stiffness_537_z"].values
ri, pi = stats.spearmanr(xi, yi)
ax.scatter(xi, yi, s=6, c=GREY, alpha=0.45, linewidths=0)
ax.set_xlabel("Inflammation score (z)")
ax.set_ylabel("Matrix-stiffness score (z)")
ax.set_title("B  Negative control (inflammation)", loc="left", fontsize=9)
ax.text(0.04, 0.94, f"Spearman r = {ri:.3f}\np = {pi:.2f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(fc="white", ec=GREY, lw=0.5, alpha=0.9))
fig.savefig(os.path.join(FIG, "Fig1.png"))
plt.close(fig)

# ================================================================== Fig 2
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1))

# 2A LL/HH 细胞谱（top 差异）
ax = axes[0]
qd = pd.read_csv(os.path.join(RES, "immune_infiltration_quadrant.csv"))
qd = qd.reindex(qd["diff_LL_minus_HH"].abs().sort_values(ascending=False).index).head(8)
qd = qd.iloc[::-1]
colors = [RED if v > 0 else BLUE for v in qd["diff_LL_minus_HH"]]
ax.barh(qd["cell"].str.replace("_", " "), qd["diff_LL_minus_HH"], color=colors, height=0.7)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("LL \u2212 HH score difference (z)")
ax.set_title("A  Cell composition: LL vs HH", loc="left", fontsize=9)
ax.tick_params(axis="y", labelsize=7)

# 2B MARS1 富集（象限划分口径必须与 scripts/02_covariation.py 完全一致：>= 中位数）
ax = axes[1]
st_med = sep["stiffness_537_z"].median()
cg_med = sep["coag_42_z"].median()
sep["quad"] = np.where((sep["stiffness_537_z"] >= st_med) & (sep["coag_42_z"] >= cg_med), "HH",
              np.where((sep["stiffness_537_z"] >= st_med) & (sep["coag_42_z"] < cg_med), "HL",
              np.where((sep["stiffness_537_z"] < st_med) & (sep["coag_42_z"] >= cg_med), "LH", "LL")))
m1 = sep[sep["endotype_class"] == "Mars1"]["quad"].value_counts()
tot = sep["quad"].value_counts()
frac = (m1 / tot * 100).reindex(["HH", "HL", "LH", "LL"]).fillna(0)
ax.bar(frac.index, frac.values, color=[BLUE, GREY, GREY, RED], width=0.62)
ax.set_ylabel("MARS1 endotype (%)")
ax.set_xlabel("Quadrant")
ax.set_title("B  MARS1 enrichment", loc="left", fontsize=9)
for i, v in enumerate(frac.values):
    ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=7.5)

# 2C mHLA-DR by quadrant
ax = axes[2]
order = ["HH", "HL", "LH", "LL"]
data = [sep.loc[sep["quad"] == q, "mhla_dr"].dropna().values for q in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.6,
                medianprops=dict(color="k", lw=1.2), showfliers=False)
for patch, c in zip(bp["boxes"], [BLUE, GREY, GREY, RED]):
    patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("k")
u, pu = stats.mannwhitneyu(data[0], data[-1])
ax.set_ylabel("mHLA-DR score (z)")
ax.set_xlabel("Quadrant")
ax.set_title("C  Monocyte antigen presentation", loc="left", fontsize=9)
ax.text(0.03, 0.05, f"HH vs LL\np = {pu:.1e}", transform=ax.transAxes,
        fontsize=7.5, va="bottom")
fig.savefig(os.path.join(FIG, "Fig2.png"))
plt.close(fig)

# ================================================================== Fig 3
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

# 3A 四象限 KM：LL vs HH
ax = axes[0]
kmf = KaplanMeierFitter()
for q, c in [("HH", BLUE), ("LL", RED)]:
    sub = sep[(sep["quad"] == q) & sep["mortality_28d"].notna()]
    kmf.fit(sub["time_to_event_28d"], sub["mortality_28d"], label=q)
    kmf.plot_survival_function(ax=ax, color=c, ci_show=False, lw=1.4)
sub_h = sep[(sep["quad"] == "HH") & sep["mortality_28d"].notna()]
sub_l = sep[(sep["quad"] == "LL") & sep["mortality_28d"].notna()]
lr = logrank_test(sub_l["time_to_event_28d"], sub_h["time_to_event_28d"],
                  sub_l["mortality_28d"], sub_h["mortality_28d"])
ax.set_xlabel("Days")
ax.set_ylabel("28-day survival")
ax.set_ylim(0.6, 1.02)
ax.set_title("A  Four-quadrant survival", loc="left", fontsize=9)
ax.text(0.55, 0.9, f"log-rank p = {lr.p_value:.3f}", transform=ax.transAxes, fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower left")

# 3B LASSO 风险分组 KM
ax = axes[1]
risk = pd.read_csv(os.path.join(RES, "lasso_cox_risk.csv"), index_col=0)
sep2 = sep.join(risk, how="inner")
med = sep2["risk"].median()
sep2["grp"] = np.where(sep2["risk"] >= med, "high-risk", "low-risk")
for q, c in [("low-risk", BLUE), ("high-risk", RED)]:
    sub = sep2[(sep2["grp"] == q) & sep2["mortality_28d"].notna()]
    kmf.fit(sub["time_to_event_28d"], sub["mortality_28d"], label=q)
    kmf.plot_survival_function(ax=ax, color=c, ci_show=False, lw=1.4)
hi = sep2[(sep2["grp"] == "high-risk") & sep2["mortality_28d"].notna()]
lo = sep2[(sep2["grp"] == "low-risk") & sep2["mortality_28d"].notna()]
lr2 = logrank_test(hi["time_to_event_28d"], lo["time_to_event_28d"],
                   hi["mortality_28d"], lo["mortality_28d"])
ax.set_xlabel("Days")
ax.set_ylabel("28-day survival")
ax.set_ylim(0.6, 1.02)
ax.set_title("B  LASSO-Cox risk groups", loc="left", fontsize=9)
ax.text(0.45, 0.9, f"log-rank p = {lr2.p_value:.1e}", transform=ax.transAxes, fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower left")
fig.savefig(os.path.join(FIG, "Fig3.png"))
plt.close(fig)

# ================================================================== Fig 4
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

# 4A 六基因交集（韦恩式示意，用真实计数）
ax = axes[0]
inter = pd.read_csv(os.path.join(RES, "intersection_genes.csv"))
genes = [str(g) for g in inter["gene"].tolist()]
ax.axis("off")
circ_st = plt.Circle((0.40, 0.55), 0.30, color=BLUE, alpha=0.35)
circ_cg = plt.Circle((0.60, 0.55), 0.30, color=RED, alpha=0.35)
ax.add_patch(circ_st); ax.add_patch(circ_cg)
ax.text(0.22, 0.90, "Stiffness\n537", ha="center", fontsize=8, color=BLUE)
ax.text(0.78, 0.90, "Coagulation\n42", ha="center", fontsize=8, color=RED)
ax.text(0.50, 0.55, "\n".join(genes), ha="center", va="center", fontsize=7.2, weight="bold")
ax.set_xlim(0.05, 0.95); ax.set_ylim(0.2, 1.05)
ax.set_title("A  Six-gene intersection", loc="left", fontsize=9)

# 4B MR 森林图（PLAUR / PLAU 的显著结果）
ax = axes[1]
mr = pd.read_csv(os.path.join(RES, "mr_intersection_results.csv"))
sel = mr[(mr["p"] < 0.05) & (mr["method"] == "IVW")].copy()
sel = sel.sort_values("OR")
lbl = [f"{r.gene} \u2192 {r.outcome.replace('ieu-b-','').replace('finn-b-','')}" for r in sel.itertuples()]
ypos = np.arange(len(sel))
ax.errorbar(sel["OR"], ypos,
            xerr=[sel["OR"] - sel["lo"], sel["hi"] - sel["OR"]],
            fmt="o", color=RED, ecolor=GREY, capsize=3, ms=5)
ax.axvline(1, color="k", ls="--", lw=0.8)
ax.set_yticks(ypos); ax.set_yticklabels(lbl, fontsize=7.5)
ax.set_xlabel("OR per genetically predicted SD (95% CI)")
ax.set_title("B  Mendelian randomization", loc="left", fontsize=9)
for i, r in enumerate(sel.itertuples()):
    ax.text(r.hi + 0.02, i, f"p={r.p:.3f}", fontsize=6.5, va="center")
fig.savefig(os.path.join(FIG, "Fig4.png"))
plt.close(fig)

# ================================================================== Fig 5
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
plt.subplots_adjust(wspace=0.42)

# 5A 细胞组成控制
ax = axes[0]
pc = pd.read_csv(os.path.join(RES, "immune_infiltration_partialcorr.csv"))
lab = ["None", "+Mono", "+Mono\n+Nphi", "+Mono+Nphi\n+CD8+B"]
ax.bar(range(len(pc)), pc["r"], color=[GREY, BLUE, BLUE, GREEN], width=0.62)
ax.set_xticks(range(len(pc))); ax.set_xticklabels(lab, fontsize=7)
ax.set_ylabel("Stiffness \u00d7 coagulation r")
ax.set_ylim(0, 0.85)
ax.axhline(0.710, color=RED, ls="--", lw=0.9)
ax.set_title("A  Cell-composition control", loc="left", fontsize=9)
for i, v in enumerate(pc["r"]):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)

# 5B GSEA 干扰素/抗病毒 NES（水平条 + 标签在轴外）
ax = axes[1]
g = pd.read_csv(os.path.join(RES, "gsea_stiffness_resid.csv"))
key = ["Interferon Alpha/Beta Signaling", "Interferon Gamma Signaling",
       "Negative Regulation Of Viral Genome Replication", "Defense Response To Virus"]
short = ["IFN-\u03b1/\u03b2 signaling", "IFN-\u03b3 signaling",
         "Neg. reg. viral genome", "Defense resp. to virus"]
rows = []
for k, s in zip(key, short):
    hit = g[g["Term"].str.contains(k, case=False, na=False)]
    if len(hit):
        rows.append((s, float(hit.iloc[0]["NES"])))
rows = sorted(rows, key=lambda t: t[1])   # 最负在底
ax.barh(range(len(rows)), [r[1] for r in rows], color=BLUE, height=0.62)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("NES (residualized)")
ax.set_title("B  Interferon/antiviral GSEA", loc="left", fontsize=9)
for i, r in enumerate(rows):
    ax.text(r[1] - 0.08, i, f"{r[1]:.2f}", ha="right", va="center",
            fontsize=6.8, color=BLUE)

# 5C 溶血敏感性
ax = axes[2]
ec = pd.read_csv(os.path.join(RES, "erythrocyte_confound.csv"))
ec = ec[ec["group"].isin(["housekeeping", "mature_RBC", "erythroid_precursor"])]
ec = ec[ec["score"] == "stiffness"]
ax.bar(range(len(ec)), ec["r"], color=[GREY, RED, RED][:len(ec)], width=0.55)
ax.set_xticks(range(len(ec)))
ax.set_xticklabels(["House-\nkeeping", "Mature\nRBC", "Erythroid\nprecursor"], fontsize=7)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("r with stiffness score")
ax.set_title("C  Haemolysis check", loc="left", fontsize=9)
for i, v in enumerate(ec["r"]):
    ax.text(i, v - 0.06 if v < 0 else v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
fig.savefig(os.path.join(FIG, "Fig5.png"))
plt.close(fig)

print("OK — 已生成：")
for f in sorted(os.listdir(FIG)):
    print("  figures/" + f)
