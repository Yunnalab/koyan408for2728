# -*- coding: utf-8 -*-
"""用已有页面缓存（校验/.pages）生成 Markdown 文本，无需重新 OCR。

    python build_md.py --dry    # 只统计
    python build_md.py          # 写出 题目/MD/*.md 与 解析/MD/*.md
"""
import os
import sys

import pymupdf

import mdrender

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "校验", ".pages")
DIRS = ("题目", "解析")


def cache_dir(rel):
    return os.path.join(CACHE, rel.replace(os.sep, "__"))


def title_of(grp, name):
    year = name[:4]
    return "%s年408真题%s" % (year, "解析" if grp == "解析" else "")


def main():
    dry = "--dry" in sys.argv
    n_ok = 0
    for grp in DIRS:
        for f in sorted(os.listdir(os.path.join(ROOT, grp, "PDF"))):
            if not f.endswith(".pdf"):
                continue
            name = f[:-4]
            rel = os.path.join(grp, name)
            cd = cache_dir(rel)
            with pymupdf.open(os.path.join(ROOT, grp, "PDF", f)) as doc:
                npages = len(doc)
            pages, missing = {}, []
            for p in range(1, npages + 1):
                cf = os.path.join(cd, "p%04d.txt" % p)
                if os.path.exists(cf):
                    with open(cf, encoding="utf-8") as fh:
                        pages[p] = [l for l in fh.read().split("\n") if l.strip()]
                else:
                    missing.append(p)
                    pages[p] = []
            note = ("由 PDF 经 OCR 生成（RapidOCR / PP-OCRv4，300 DPI），共 %d 页，"
                    "仅含文字层；图、公式、表格请对照 PDF。" % npages)
            md = mdrender.render_md(pages, title_of(grp, name), note,
                                    kind="answer" if grp == "解析" else "exam")
            out = os.path.join(ROOT, grp, "MD", name + ".md")
            if dry:
                print("%-42s %d 页 %d 字%s" % (rel.replace(os.sep, "/"), npages, len(md),
                                              "  缺页:%s" % missing if missing else ""))
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(md)
            n_ok += 1
    print("%s%d 个 Markdown 文件" % ("[dry] " if dry else "已写出 ", n_ok))


if __name__ == "__main__":
    main()
