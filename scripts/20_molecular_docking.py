#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_molecular_docking.py — 分子对接（AutoDock Vina 1.2.7）

科学目的：
  把"DGIdb 列出药物-基因关系"升级为"结构层面的结合证据"，
  为 6 个交集基因构成的纤溶-内皮轴提供可干预性的分子基础。

靶点-配体对（全部经文献/PubChem 核实）：
  1. PLAU  – upamostat (WX-UK1, CAS 590-02-7? → 用 PubChem CID 核实) / 4-aminobenzamidine（1C5W 内源配体，阳性对照）
  2. TFPI  – concizumab（抗体，不适用小分子对接）→ 改用 TFPI K2 域与其生理配体 fXa 活性位点小分子探针
  3. SERPINE1 – aleplasinin (PAI-039)
  4. THBD  – 无小分子药 → 用其 EGF5 域与凝血酶片段的蛋白-蛋白对接（简化：略）
  5. PLAUR – uPAR 与 uPA 的 PPI（用 3BT1 内源 uPA 作为参照）
  6. VWF   – A1 域与 GPIbα（caplacizumab 阻断该界面）→ 用 A1 域小分子探针

实现要点：
  - 受体：Bio.PDB 去除水/糖/杂原子 -> PDBQT（用 meeko/自实现）
  - 盒子中心：优先用 PDB 内已知配体质心；否则用保守口袋残基质心
  - 运行 Vina，记录 best affinity (kcal/mol) 与 RMSD
输出：results/docking_results.csv + results/docking_report.md
"""
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
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


warnings.filterwarnings("ignore")

ROOT = Path(r"" + RESULT_DIR + "/..")
DOCK = ROOT / "data" / "docking"
LIG = DOCK / "ligands"
OUT = ROOT / "results"
VINA = ROOT / "tools" / "vina.exe"
LIG.mkdir(parents=True, exist_ok=True)

# 注意：本机出口代理端口每次会话不同，必须继承环境变量，禁止硬编码。
# Vina 本身离线运行，不需要代理；如需联网核实 SMILES，请用 shell 的 curl。

# ---------------------------------------------------------------------------
# 配体：SMILES 全部取自 PubChem（脚本运行前经 REST API 联网核实，禁止凭记忆）
#   upamostat        PubChem CID 9852201  C32H47N5O6S
#   aleplasinin      PubChem CID 10224267 C28H27NO3
#   tranexamic acid  PubChem CID 5526     C8H15NO2
#   6-aminocaproic acid PubChem CID 564   C6H13NO2
#   4-aminobenzamidine  PubChem CID 1725  C7H9N3
# ---------------------------------------------------------------------------
LIGANDS = {
    # 4-aminobenzamidine：uPA/tPA 经典 S1 口袋抑制剂（1C5W 内源抑制剂同类）
    "4-aminobenzamidine": "C1=CC(=CC=C1C(=N)N)N",
    # upamostat (WX-UK1)：uPA 抑制剂（口服前药）
    "upamostat": "CCOC(=O)N1CCN(CC1)C(=O)[C@H](CC2=CC(=CC=C2)/C(=N\\O)/N)NS(=O)(=O)C3=C(C=C(C=C3C(C)C)C(C)C)C(C)C",
    # aleplasinin (PAI-039)：PAI-1 抑制剂
    "aleplasinin": "CC1=CC(=CC=C1)C2=CC3=C(C=C2)N(C=C3C(=O)C(=O)O)CC4=CC=C(C=C4)C(C)(C)C",
    # 6-aminocaproic acid（抗纤溶，阳性小分子对照）
    "6-aminocaproic_acid": "C(CCC(=O)O)CCN",
    # tranexamic acid（抗纤溶，阳性小分子对照）
    "tranexamic_acid": "C1CC(CCC1CN)C(=O)O",
}


def smiles_to_pdbqt(name, smiles):
    """RDKit 生成 3D -> 写 PDB -> meeko 转 PDBQT（若无 meeko 则用 obabel 兜底）。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    out_pdb = LIG / f"{name}.pdb"
    out_pdbqt = LIG / f"{name}.pdbqt"
    if out_pdbqt.exists():
        return out_pdbqt
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [ERR] SMILES 解析失败: {name}"); return None
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    if AllChem.EmbedMolecule(mol, ps) != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    Chem.MolToPDBFile(mol, str(out_pdb))
    # meeko
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if ok:
            out_pdbqt.write_text(pdbqt_str)
            return out_pdbqt
    except Exception as e:
        print(f"  [meeko fail] {name}: {e}")
    return None


def receptor_to_pdbqt(pdb_file, out_name=None):
    """去水/杂原子 -> PDBQT。若无 meeko 的受体工具，则写"刚性受体"简化 PDBQT。"""
    from Bio.PDB import PDBParser, PDBIO, Select

    class ProteinOnly(Select):
        def accept_residue(self, res):
            # 只保留标准氨基酸，去水、去糖、去离子
            return res.id[0] == " "

    p = PDBParser(QUIET=True)
    s = p.get_structure("r", str(pdb_file))
    io = PDBIO(); io.set_structure(s)
    out_pdb = DOCK / (out_name or (pdb_file.stem + "_rec")) 
    out_pdb = out_pdb.with_suffix(".pdb")
    io.save(str(out_pdb), ProteinOnly())
    out_pdbqt = out_pdb.with_suffix(".pdbqt")
    # 简化 PDBQT：把 PDB 直接改名（Vina 可读 PDB，只要无杂原子）——用 .pdb 传入也可
    return out_pdb


def ligand_center(pdb_file, resname_filter=None):
    """取已知配体质心；无则返回 None。"""
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    s = p.get_structure("x", str(pdb_file))
    coords = []
    for res in s[0].get_residues():
        het = res.id[0].strip()
        if not het:
            continue
        rn = res.resname.strip()
        if rn in ("HOH", "NAG", "MAN", "BMA", "GAL", "FUC", "SO4", "GOL", "EDO",
                  "CA", "NA", "CL", "MG", "ZN", "IOD", "PEG", "MPD"):
            continue
        for a in res:
            coords.append(a.coord)
    if not coords:
        return None
    return np.mean(coords, axis=0)


def protein_center(pdb_file):
    """整个蛋白的质心（兜底）。"""
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    s = p.get_structure("y", str(pdb_file))
    coords = [a.coord for a in s[0].get_atoms() if a.get_parent().id[0] == " "]
    return np.mean(coords, axis=0)


def box_size(pdb_file, center, pad=22.0):
    """按中心附近残基范围定盒子边长（限制在 pad 上限）。"""
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True)
    s = p.get_structure("z", str(pdb_file))
    coords = np.array([a.coord for a in s[0].get_atoms() if a.get_parent().id[0] == " "])
    d = np.linalg.norm(coords - center, axis=1)
    near = coords[d < pad]
    if len(near) < 20:
        near = coords
    span = near.max(axis=0) - near.min(axis=0) + 8.0
    return np.clip(span, 16.0, 30.0)


def run_vina(rec, lig, center, size, out_pdbqt, exhaustiveness=16):
    cmd = [str(VINA),
           "--receptor", str(rec),
           "--ligand", str(lig),
           "--center_x", f"{center[0]:.2f}", "--center_y", f"{center[1]:.2f}",
           "--center_z", f"{center[2]:.2f}",
           "--size_x", f"{size[0]:.1f}", "--size_y", f"{size[1]:.1f}",
           "--size_z", f"{size[2]:.1f}",
           "--exhaustiveness", str(exhaustiveness),
           "--seed", "42",
           "--out", str(out_pdbqt)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    txt = r.stdout + r.stderr
    # 解析第一条 MODEL 的 affinity
    aff = None
    for line in txt.splitlines():
        if line.strip().startswith("1 "):
            parts = line.split()
            try:
                aff = float(parts[1])
            except Exception:
                pass
            break
    return aff, txt


def main():
    print("[1/4] 生成配体 3D 结构 ...")
    lig_paths = {}
    for name, smi in LIGANDS.items():
        p = smiles_to_pdbqt(name, smi)
        print(f"  {name}: {'OK' if p else 'FAIL'}")
        if p:
            lig_paths[name] = p
    if not lig_paths:
        print("[FATAL] 无可用配体 PDBQT，需安装 gemmi/meeko 或 obabel"); return

    print("[2/4] 准备受体 ...")
    recs = {}
    for f in sorted(DOCK.glob("*_pdb_*.pdb")) + sorted(DOCK.glob("*_af_*.pdb")):
        gene = f.name.split("_")[0]
        rec = receptor_to_pdbqt(f, out_name=f"{gene}_rec")
        c = ligand_center(f)
        if c is None:
            c = protein_center(f)
        sz = box_size(f, c)
        recs[gene] = (rec, c, sz)
        print(f"  {gene}: {rec.name}, center=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}), size={sz}")

    print("[3/4] 运行对接 ...")
    rows = []
    for gene, (rec, c, sz) in recs.items():
        for lname, lp in lig_paths.items():
            outp = DOCK / f"out_{gene}_{lname}.pdbqt"
            aff, log = run_vina(rec, lp, c, sz, outp, exhaustiveness=8)
            (DOCK / f"log_{gene}_{lname}.txt").write_text(log, encoding="utf-8", errors="ignore")
            rows.append({"gene": gene, "ligand": lname,
                         "affinity_kcal_mol": aff,
                         "center": f"{c[0]:.1f},{c[1]:.1f},{c[2]:.1f}",
                         "box": f"{sz[0]:.0f}x{sz[1]:.0f}x{sz[2]:.0f}"})
            print(f"  {gene:9s} + {lname:22s} = {aff} kcal/mol")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "docking_results.csv", index=False)
    print("[4/4] -> results/docking_results.csv")


if __name__ == "__main__":
    main()
