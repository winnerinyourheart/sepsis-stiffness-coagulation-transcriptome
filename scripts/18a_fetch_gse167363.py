#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18a_fetch_gse167363.py — 下载 GSE167363 全部样本 10x 矩阵

【关键经验（本机环境，2026-10-07 实测）】
  直接 curl -o 写入项目目录会间歇性得到 990 字节的 404 页面
  （代理缓存 + 目标目录写入竞态）。先下载到系统临时目录再 copy 到
  项目目录可稳定成功。脚本据此实现。

代理：继承 shell 已导出的 http(s)_proxy，不硬编码端口。
"""
import os
import shutil
import subprocess
import tempfile
import time
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


DEST = Path(r"" + DATA_DIR + "/sc")
DEST.mkdir(parents=True, exist_ok=True)
TMP = Path(tempfile.gettempdir()) / "sc_dl"
TMP.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("GSM5102900", "HC1"), ("GSM5102901", "HC2"),
    ("GSM5102902", "P25_T0"), ("GSM5102903", "P25_T6"),
    ("GSM5102904", "P50_T0"), ("GSM5102905", "P50_T6"),
    ("GSM5511351", "NSES_T0"), ("GSM5511352", "NSES_T6"),
    ("GSM5511353", "S2_T0"), ("GSM5511354", "S2_T6"),
    ("GSM5511355", "S3_T0"), ("GSM5511356", "S3_T6"),
]


def is_valid_gz(p):
    """确认是真实 gzip 且能读出内容（排除 404 HTML 页）。"""
    try:
        import gzip
        with gzip.open(p, "rb") as f:
            return len(f.read(500)) > 0
    except Exception:
        return False


def geo_suppl_dir(gsm):
    """按 NCBI GEO 规范构造样本 suppl 目录名。

    规则：取 GSM 后**第一个数字起的前 6 位**，再补 'nnn'。
      GSM5102900 -> GSM5102nnn   （不是 GSM510nnn！）
      GSM5511351 -> GSM5511nnn   （不是 GSM551nnn！）

    历史 bug：早期脚本用 gsm[:6] ('GSM510') + 'nnn' = 'GSM510nnn'，
    导致全部请求 404（先返回 200 代理隧道、再 404，被误判为网络问题）。
    """
    digits = gsm[3:]                    # 去掉 'GSM'
    if not digits.isdigit():
        raise ValueError(f"非法 GSM 号: {gsm}")
    return "GSM" + digits[:6] + "nnn"


def fetch(url, out, retries=5):
    """先下到临时目录，校验后 copy 到项目目录。"""
    if out.exists() and out.stat().st_size > 5000 and is_valid_gz(out):
        print(f"   [skip] {out.name}", flush=True)
        return True, out.stat().st_size

    tmp = TMP / out.name
    for i in range(retries):
        tmp.unlink(missing_ok=True)
        r = subprocess.run(["curl", "-sS", "--max-time", "900", "-o", str(tmp), url],
                           capture_output=True, text=True)
        if tmp.exists() and tmp.stat().st_size > 5000 and is_valid_gz(tmp):
            shutil.copy(str(tmp), str(out))
            return True, out.stat().st_size
        print(f"   [retry {i+1}] {out.name} (size="
              f"{tmp.stat().st_size if tmp.exists() else 0})", flush=True)
        time.sleep(4)
    return False, 0


def main():
    ok = 0
    total_n = 0
    for gsm, name in SAMPLES:
        sub = f"{geo_suppl_dir(gsm)}/{gsm}/suppl"
        for suf in ["barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz"]:
            fn = f"{gsm}_{name}_{suf}"
            total_n += 1
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sub}/{fn}"
            print(f"-> {fn}", flush=True)
            good, sz = fetch(url, DEST / fn)
            print(f"   {'OK' if good else 'FAIL'} {sz/1e6:.2f} MB", flush=True)
            if good:
                ok += 1
    print(f"\n完成 {ok}/{total_n}", flush=True)


if __name__ == "__main__":
    main()
