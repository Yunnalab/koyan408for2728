# -*- coding: utf-8 -*-
"""把 题目/ 与 解析/ 各自再拆成 PDF/ 与 TXT/ 两个子目录：

    题目/PDF/*.pdf   题目/TXT/*.txt
    解析/PDF/*.pdf   解析/TXT/*.txt

用法:  python split_txt_pdf.py --dry | python split_txt_pdf.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    dry = "--dry" in sys.argv
    pairs = []
    for grp in ("题目", "解析"):
        base = os.path.join(ROOT, grp)
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            src = os.path.join(base, f)
            if not os.path.isfile(src):
                continue
            sub = "PDF" if f.lower().endswith(".pdf") else ("TXT" if f.lower().endswith(".txt") else None)
            if sub:
                pairs.append((src, os.path.join(base, sub, f)))
    print("移动 %d 个文件：" % len(pairs))
    for a, b in pairs[:4] + pairs[-4:]:
        print("  %s -> %s" % (os.path.relpath(a, ROOT), os.path.relpath(b, ROOT)))
    if dry:
        return
    for grp in ("题目", "解析"):
        for sub in ("PDF", "TXT"):
            os.makedirs(os.path.join(ROOT, grp, sub), exist_ok=True)
    for a, b in pairs:
        if os.path.exists(b):
            print("跳过（已存在）:", os.path.relpath(b, ROOT))
            continue
        shutil.move(a, b)
    print("完成")


if __name__ == "__main__":
    main()
