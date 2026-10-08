# Coordinated downregulation of matrix-stiffness-associated and coagulation transcriptional programmes marks monocyte immunoparalysis and a high-mortality endotype in sepsis

Analysis code accompanying the manuscript submitted to *Journal of Translational Medicine*.

Xueshu Yu¹, Yincai Ye², Jingye Pan¹\* — ¹Department of Intensive Care Medicine, ²Department of Blood Transfusion, The First Affiliated Hospital of Wenzhou Medical University, Wenzhou, China.

---

## Overview

The pipeline tests whether a matrix-stiffness-associated transcriptional programme co-varies with the coagulation programme in septic whole blood, what clinical state that co-variation marks, and whether the intersecting genes are druggable. Primary cohort is the MARS sepsis cohort (GSE65682); directional replication uses GSE95233, E-MTAB-4451 and GSE185263.

Every statistic reported in the manuscript is traceable to a table in the accompanying `results/` directory, and every table is produced by a script in `scripts/`. The analysis deliberately starts from **raw probe-level matrices**, not from intermediate products, so that the whole chain can be re-derived.

## Pipeline

| Stage | Script | Purpose | Key output |
|---|---|---|---|
| 1. Scoring | `01_scores.py` | ssGSEA stiffness / coagulation / inflammation scores in GSE65682 | `scores_gse65682.csv` |
| 2. Co-variation | `02_covariation.py` | Spearman co-variation, endotype/quadrant stratification | `covariation_corr.csv`, `quadrant_survival.csv` |
| 3. Technical QC | `03_technical_check.py` | Housekeeping-gene / detection-rate artefacts | — |
| 4. Validation | `04_validation.py`, `07_emtab4451_validation.py`, `09_gse185263_validate.py` | Directional replication in three external cohorts | `validation_*.csv` |
| 5. Immunoparalysis | `05_immunoparalysis.py`, `06_mhla_score.py` | mHLA-DR transcriptional proxy, T-cell marker direction | `immunoparalysis_corr.csv`, `mhla_dr_gse65682.csv` |
| 6. Differential intersection | `10_deg_intersection_enrichment.py` | Stiffness ∩ coagulation gene intersection + enrichment | `intersection_genes.csv`, `enrichment_*.csv` |
| 7. Co-variation network | `11_covariation_network_enrichment.py` | Co-downregulated network + Reactome enrichment | `covariation_driver_genes.csv` |
| 8. Prognostic signature | `12_lasso_cox_signature.py` | LASSO-Cox over the 32-gene candidate pool | `lasso_cox_signature_coef.csv`, `lasso_cox_risk.csv` |
| 9. Consensus clustering | `13_consensus_clustering.py` | Unsupervised structure check (reported as weak) | `consensus_clusters.csv` |
| 10. Mendelian randomization | `14_mr_intersection.py`, `15_fetch_outcome_intersection.py` | cis-pQTL MR for the six intersecting genes | `mr_intersection_results.csv` |
| 11. Drug–gene mapping | `16_dgidb_intersection.py` | DGIdb GraphQL drug–gene interactions | `dgidb_intersection.csv` |
| 12. Immune deconvolution | `17_immune_infiltration.py` | 26-population marker scoring, partial-correlation control | `immune_infiltration_*.csv` |
| 13. Ranked GSEA | `19_gsea_ranked.py` | Permutation GSEA on the Spearman-ranked gene list | `gsea_*_raw.csv`, `gsea_*_resid.csv` |
| 14. Molecular docking | `20_molecular_docking.py`, `20a_fetch_structures.py`, `22_docking_control_analysis.py` | AutoDock-Vina docking + ligand-efficiency control | `docking_results.csv`, `docking_ligand_efficiency.csv` |
| 15. Erythrocyte confound | `21_erythrocyte_confound_check.py` | Haemolysis sensitivity analysis | `erythrocyte_confound.csv` |
| 16. Figures | `25_make_figures.py` | Fig. 1–5 (300 dpi PNG) | `figures/Fig1-5.png` |
| 17. Audit | `24_audit_v02.py` | 26-item recomputation audit of all reported numbers | `审计_v0.2.csv` |
| 18. CI of external r | `26_validation_ci.py` | Percentile-bootstrap 95% CI for the cross-cohort co-variation (10,000 resamples, seed 42) | `validation_coverage_ci.csv` |
| 19. Orthogonal ssGSEA | `27_ssgsea_orthogonal.py` | Independent gseapy re-implementation of the primary scores | `ssgsea_orthogonal_check.csv`, `ssgsea_orthogonal_summary.md` |
| 20. Signature transport | `28_signature_external.py` | External discrimination (AUC) of the 21-gene signature under two scaling schemes | `signature_external_validation.csv`, `.md` |
| — | `08_gse185263_fetch_meta.py`, `18a_fetch_gse167363.py`, `23_singlecell_localization.py` | Data fetchers / single-cell utility (not used for a reported result; retained for completeness) | — |

### Note on the single-cell scripts

`18a_fetch_gse167363.py` and `23_singlecell_localization.py` are retained for transparency but **no single-cell result is reported in the manuscript**. The manuscript contains no cell-level claim; all conclusions are whole-blood–level inferences.

## File index (original working name → released name)

Scripts are released under their **original working filenames** so that the traceability chain from manuscript numbers to code is unbroken. No renaming was performed.

## Data sources and accessions

| Dataset | Source | Accession | Role |
|---|---|---|---|
| MARS sepsis cohort | GEO | GSE65682 | Discovery (n = 479 sepsis) |
| Septic shock cohort | GEO | GSE95233 | Replication (n = 102) |
| GAinS severe sepsis | ArrayExpress | E-MTAB-4451 | Replication (n = 106) |
| Sepsis RNA-seq cohort | GEO | GSE185263 | Replication (n = 392) |
| Sepsis susceptibility | IEU OpenGWAS | ieu-b-4980 | MR outcome |
| 28-day sepsis mortality | IEU OpenGWAS | ieu-b-5086 | MR outcome |
| Other septicaemia | FinnGen | finn-b-AB1_OTHER_SEPSIS | MR outcome |
| deCODE cis-pQTL | deCODE (Ferkingstad 2021) | — | MR exposure |
| DGIdb | dgidb.org GraphQL API | — | Drug–gene mapping |

Datasets are not redistributed here; download them from the accessions above.

## Running the code

```bash
pip install -r requirements.txt

# Point the scripts at your local copies of the inputs
export STS_DATA_DIR="/path/to/expression/inputs"      # raw probe matrices, phenotype tables
export STS_RESULT_DIR="/path/to/output/results"       # result tables are written here
export STS_EXT_DIR="/path/to/shared/resources"        # deCODE pQTL tables, shared gene sets

python scripts/01_scores.py
python scripts/02_covariation.py
# ... run stages in numeric order
python scripts/25_make_figures.py
```

`STS_EXT_DIR` is only required by the scripts that consume shared resources (the deCODE pQTL clean tables for the MR stage and the curated stiffness gene set). See the header of each script for the exact filenames it expects.

## Environment

Tested on Python 3.12 with the versions pinned in `requirements.txt`. Docking stages additionally require AutoDock Vina 1.2.7 on `PATH`.

## Citation

If you use this code, please cite the manuscript (under review). 

## License

MIT — see `LICENSE`.
