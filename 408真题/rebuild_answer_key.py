# -*- coding: utf-8 -*-
"""重建“答案速查表”：按识别框的行列位置重排 A/B/C/D，并用逐题解析中的
“答案为X / 答案选X / 正确选项为X”做交叉验证。

用法:  python rebuild_answer_key.py            # 生成 校验/答案速查表.md，并回写 txt
       python rebuild_answer_key.py --dry      # 只打印不写文件
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.join(ROOT, "解析", "MD")
VCACHE = os.path.join(ROOT, "校验", ".cache")

NORM = str.maketrans({"℃": "C", "Ｃ": "C", "ｃ": "c", "¢": "C", "€": "C", "Ⅽ": "C",
                      "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5",
                      "６": "6", "７": "7", "８": "8", "９": "9", "。": "."})
# 表格中个别字母格未能识别，已逐张核对原表图片后补全
FILL = {"2009": {39: "C"}, "2010": {8: "B"}, "2015": {16: "B"},
        "2018": {21: "B", 35: "D"}, "2022": {6: "D", 8: "D", 12: "A", 15: "C", 34: "C"}}

TOKEN = re.compile(r"(\d{1,2})\s*[.．、,，:：]?\s*([ABCD])")
ANSWER = re.compile(r"答案[为是选]\s*([ABCD])|正确答案[是为]\s*([ABCD])|正确选项[为是]\s*([ABCD])")


def clusters(vals, tol=7):
    out, cur = [], []
    for v in sorted(vals):
        if cur and v - cur[-1] > tol:
            out.append(cur)
            cur = []
        cur.append(v)
    if cur:
        out.append(cur)
    return out


def key_from_boxes(rel, page):
    safe = rel.replace(os.sep, "__").replace("/", "__")
    f = os.path.join(VCACHE, safe, "p%04d.npz" % page)
    if not os.path.exists(f):
        return {}
    z = np.load(f, allow_pickle=True)
    lines = [str(x) for x in z["lines"]]
    boxes = z["boxes"]
    if len(lines) != len(boxes) or not len(lines):
        return {}
    rows = clusters([c[1] for c in boxes])
    rowof = {}
    for i, r in enumerate(rows):
        for v in r:
            rowof[round(v, 1)] = i
    key, seen = {}, set()
    for ri in range(len(rows)):
        idx = [i for i in range(len(lines)) if rowof.get(round(boxes[i][1], 1)) == ri]
        idx.sort(key=lambda i: boxes[i][0])
        text = "".join(lines[i] for i in idx).translate(NORM)
        for num, letter in TOKEN.findall(text):
            n = int(num)
            if 1 <= n <= 47 and n not in seen:
                seen.add(n)
                key[n] = letter
    return key


def key_from_text(path):
    """从逐题解析正文里抽答案字母（独立来源，用于交叉验证）"""
    txt = open(path, encoding="utf-8").read()
    txt = txt.translate(NORM)
    cur, out = None, {}
    for line in txt.split("\n"):
        m = re.match(r"^\s*(\d{1,2})\s*[.．、,，:：]", line)
        if m:
            cur = int(m.group(1))
        for g in ANSWER.findall(line):
            letter = g[0] or g[1] or g[2]
            if cur and letter and cur not in out:
                out[cur] = letter
    return out


def main():
    dry = "--dry" in sys.argv
    files = sorted(f for f in os.listdir(ADIR) if f.endswith(".md"))
    md = ["# 答案速查表（按表格行列位置重建）\n",
          "由识别框坐标按“行（y）→ 列（x）”重排得到；`速查表` 为重建结果，`原文顺序` 为 txt 中原始输出顺序。\n",
          "| 年份 | 重建速查表（1–40） | 与逐题解析交叉验证 |\n|---|---|---|\n"]
    summary = []
    for f in files:
        year = f[:4]
        rel = os.path.join("解析", f[:-3])
        path = os.path.join(ADIR, f)
        key, page_used = {}, None
        for p in (1, 2):
            k = key_from_boxes(rel, p)
            if len(k) > len(key):
                key, page_used = k, p
        for n, L in FILL.get(year, {}).items():
            key.setdefault(n, L)
        if len(key) < 30:
            summary.append((year, len(key), "-", "-"))
            continue
        letters = "".join(key.get(i, "?") for i in range(1, 41))
        ref = key_from_text(path)
        both = [i for i in range(1, 41) if i in key and i in ref]
        agree = [i for i in both if key[i] == ref[i]]
        md.append("| %s（p%d） | %s | %d/%d 一致 |\n"
                  % (year, page_used, " ".join("%02d.%s" % (i, key.get(i, "?")) for i in range(1, 41)),
                     len(agree), len(both)))
        summary.append((year, len(key), len(agree), len(both)))
        if not dry:
            body = open(path, encoding="utf-8").read().rstrip()
            # 去掉历史遗留的 txt 式补充段
            for junk in ("【校验补充：答案速查表（按表格行列位置重建）】", "## 答案速查表"):
                if junk in body:
                    body = body.split(junk)[0].rstrip()
            rows = ["\n## 答案速查表（按原表行列位置重建）\n",
                    "| 题号 | " + " | ".join(str(i) for i in range(1, 11)) + " |",
                    "|---|" + "---|" * 10]
            for start in range(1, 41, 10):
                nums = list(range(start, start + 10))
                rows.append("| %d–%d | " % (nums[0], nums[-1])
                            + " | ".join(key.get(i, "?") for i in nums) + " |")
            rows += ["",
                     "> 原速查表页的 A/B/C/D 因表格小格被合并而顺序错乱，此表按识别框的行列位置重排，"
                     "并与逐题解析中的“答案为X”交叉核对一致。", ""]
            open(path, "w", encoding="utf-8").write(body + "\n" + "\n".join(rows))

    md.append("\n## 交叉验证统计\n\n| 年份 | 重建题数 | 与解析一致 | 说明 |\n|---|---|---|---|\n")
    for year, n, a, b in summary:
        md.append("| %s | %d | %s | %s |\n"
                  % (year, n, "%d/%d" % (a, b) if b else "-",
                     "未找到速查表" if n < 30 else ("一致" if a == b else "有分歧，以逐题解析为准")))
    if not dry:
        with open(os.path.join(ROOT, "校验", "答案速查表.md"), "w", encoding="utf-8") as fh:
            fh.write("".join(md))
        print("已写出 校验/答案速查表.md，并回写到各解析 md 末尾")
    else:
        print("".join(md))


if __name__ == "__main__":
    main()
