# -*- coding: utf-8 -*-
"""批量 OCR：对本目录（含 答案/ 子目录）下所有 PDF 逐页识别，输出 UTF-8 文本。

用法:
    python ocr_all.py                 # 增量运行，已完成的页面自动跳过
    python ocr_all.py --workers 6
    python ocr_all.py --force         # 忽略缓存全部重做
"""
import argparse
import os
import sys
import time

import numpy as np
import pymupdf
from multiprocessing import Pool

import mdrender

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = ("题目", "解析")          # 每个目录下再分 PDF/（原件）与 TXT/（OCR 文本）
CACHE = os.path.join(ROOT, "校验", ".pages")
DPI = 300            # 渲染分辨率
LSL = 1400           # 检测模型长边/短边限制，越大越能识别小字
INTRA = 3            # 每个进程 onnxruntime 线程数
BATCH = 8            # 识别批大小
USE_CLS = False      # 关闭0/180度分类器（本批文档均为正向，开启会误翻转代码行为乱码）

_ocr = None


def _init():
    global _ocr
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR(det_limit_side_len=LSL, intra_op_num_threads=INTRA,
                    rec_batch_num=BATCH, text_score=0.5, use_cls=USE_CLS)


def list_pdfs():
    """返回 [(pdf绝对路径, rel名称)]，rel名称不含扩展名、保留子目录。"""
    items = []
    for sub in DIRS:
        d = os.path.join(ROOT, sub, "PDF")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".pdf"):
                items.append((os.path.join(d, f), os.path.join(sub, f[:-4])))
    return items


def cache_path(rel, pno):
    safe = rel.replace("\\", "__").replace("/", "__")
    return os.path.join(CACHE, safe, "p%04d.txt" % (pno + 1))


def render(path, pno):
    doc = pymupdf.open(path)
    page = doc[pno]
    pm = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    return np.ascontiguousarray(img[:, :, :3][:, :, ::-1]), len(doc)


def work(task):
    """task = (path, pno, rel, force) -> (rel, pno, npages, status, nlines)"""
    path, pno, rel, force = task
    cp = cache_path(rel, pno)
    if not force and os.path.exists(cp) and os.path.getsize(cp) > 0:
        return (rel, pno, None, "skip", 0)
    try:
        img, npages = render(path, pno)
        res, _ = _ocr(img)
        lines = [r[1].strip() for r in (res or []) if r[1] and r[1].strip()]
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return (rel, pno, npages, "ok", len(lines))
    except Exception as exc:                       # noqa: BLE001
        return (rel, pno, None, "ERR:%s" % exc, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pdfs = list_pdfs()
    os.makedirs(CACHE, exist_ok=True)

    # 统计页数
    plan, nskip = [], 0
    npages_of = {}
    for path, rel in pdfs:
        with pymupdf.open(path) as d:
            n = len(d)
        npages_of[rel] = n
        for p in range(n):
            cp = cache_path(rel, p)
            if not args.force and os.path.exists(cp) and os.path.getsize(cp) > 0:
                nskip += 1
            else:
                plan.append((path, p, rel, args.force))

    print("PDF 文件 %d 个，总页数 %d，待识别 %d 页（跳过缓存 %d 页）"
          % (len(pdfs), sum(npages_of.values()), len(plan), nskip), flush=True)

    t0 = time.time()
    done = 0
    errors = []
    todo = len(plan)
    if todo:
        with Pool(args.workers, initializer=_init) as pool:
            for rel, pno, npages, status, nlines in pool.imap_unordered(work, plan, chunksize=1):
                done += 1
                if status.startswith("ERR"):
                    errors.append((rel, pno + 1, status))
                if done % 5 == 0 or done == todo:
                    el = time.time() - t0
                    eta = el / done * (todo - done)
                    print("[%d/%d] %.1fs  已用%.1fmin  预计剩余%.1fmin  %s p%d"
                          % (done, todo, el, el / 60, eta / 60, rel.split(os.sep)[-1][:24], pno + 1),
                          flush=True)

    # 组装每个 PDF 的最终 Markdown
    print("组装 Markdown ...", flush=True)
    for path, rel in pdfs:
        pages, missing = {}, []
        for p in range(npages_of[rel]):
            cp = cache_path(rel, p)
            if os.path.exists(cp):
                with open(cp, encoding="utf-8") as fh:
                    pages[p + 1] = [l for l in fh.read().split("
") if l.strip()]
            else:
                missing.append(p + 1)
                pages[p + 1] = []
        grp, name = rel.split(os.sep)
        title = "%s年408真题%s" % (name[:4], "解析" if grp == "解析" else "")
        note = ("由 PDF 经 OCR 生成（RapidOCR / PP-OCRv4，%d DPI），共 %d 页，"
                "仅含文字层；图、公式、表格请对照 PDF。" % (DPI, npages_of[rel]))
        out_file = os.path.join(ROOT, grp, "MD", name + ".md")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as fh:
            fh.write(mdrender.render_md(pages, title, note,
                                        kind="answer" if grp == "解析" else "exam"))
        flag = "" if not missing else "  !! 缺页 %s" % missing
        print("  %-60s %d 页%s" % (name[:58], npages_of[rel], flag), flush=True)

    if errors:
        print("\n识别报错页面：")
        for e in errors:
            print("  ", e)
    print("全部完成，用时 %.1f 分钟。输出：题目/MD/、解析/MD/ 下与 PDF 同名的 .md" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
