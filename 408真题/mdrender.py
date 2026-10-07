# -*- coding: utf-8 -*-
"""把 OCR 文本行渲染成结构化 Markdown（排版版）。

渲染规则
--------
- `# 标题`               文件标题
- `## 第 N 页`            页分隔
- `### NN．题干…`         题号行 -> 小标题（答案表里的 `01.D` 纯条目保持普通行）
- 解析行 `01.D。【解析】…` -> `### 1.D` + 正文
- `- A．…`               选项行 -> 列表（一行多选项自动拆开）
- ```` ```c / ```asm ````  连续代码行 -> 代码块（C 伪代码 / 汇编）
- `*题33～41图*`         图、表题注 -> 斜体
- 其余按段落回流：被折行截断的句子合并成一段（表格短行不合并）
"""
import re

PAGE_RE = re.compile(r"^##\s*第\s*(\d+)\s*页\s*$")
LEAD = r"[\s·•.．,，;；:：'\"“”‘’\-_—)*]*"
# 题号后可紧跟数字（如 `6．5个字符…`），IP 之类由“题号不能超出期望值+3”拦掉
QNUM = re.compile(r"^" + LEAD + r"(\d{1,2})\s*([．.。、:：])\s*(\S.*)?$")
KEYLINE = re.compile(r"^([ABCD])[.．。、]?\s*(.*)$")
OPT_SPLIT = re.compile(r"(?<![A-Za-z0-9])([ABCD])\s*([．.、])\s*")
CAPTION = re.compile(r"^(题|图|表)\s*[0-9０-９a-zA-Z]{0,3}\s*[～~\-—]\s*[0-9０-９]{1,2}\s*图$"
                     r"|^(图|表)\s*[0-9０-９IVXivx]{1,3}\s*[图表示]?$")
CJK = re.compile(r"[\u4e00-\u9fff]")

FUNCALL = re.compile(r"^\s*[A-Za-z_]\w*\s*\([^)]*\)\s*;?\s*$")
ASM = re.compile(r"\b(MOV|ADD|SUB|MUL|IMUL|DIV|IDIV|INC|DEC|CMP|JMP|JE|JNE|JG|JL|JGE|JLE|"
                 r"CALL|RET|PUSH|POP|XOR|AND|OR|NOT|NEG|TEST|SHL|SHR|SAR|LOAD|STORE|LDA|STA|"
                 r"NOP|HLT|LEA|XCHG)\b")
REG = re.compile(r"\b(EAX|EBX|ECX|EDX|ESP|EBP|ESI|EDI|RAX|AX|BX|CX|DX|AL|AH|BL|BH|"
                 r"PC|IR|MAR|MDR|ACC|PSW|SP|R[0-7])\b")
CFUNC = re.compile(r"\b(int|char|void|bool|float|double|long|short|unsigned|struct|typedef|"
                   r"static|const|return|sizeof|printf|scanf|malloc|free|new|delete|"
                   r"for|while|if|else|switch|case|break|continue)\b")
CSTRONG = re.compile(r"^\s*(//|/\*|\*|#include|#define)")
ASSIGN = re.compile(r"(\+\+|--|\+=|-=|\*=|/=|->|<-|=)")
ASCII_MIX = re.compile(r"[A-Za-z]")
CJK_PARA_END = re.compile(r"[。！？；：”』）】….;:!?]$")


def is_code(line):
    """返回 (是否代码, 语言 or None)；强特征才算代码起点"""
    asm = bool(ASM.search(line)) and not CJK.search(line)
    reg = bool(REG.search(line)) and len(CJK.findall(line)) <= 2
    strong = bool(CSTRONG.match(line)) or (line.rstrip().endswith(";")
                                           and bool(ASSIGN.search(line) or "(" in line))
    if line.strip() in "{}":
        strong = True
    func = bool(CFUNC.search(line)) and (bool(ASSIGN.search(line)) or "(" in line or ";" in line) \
        and len(CJK.findall(line)) <= 2
    weak = bool(ASSIGN.search(line)) and len(CJK.findall(line)) == 0 and len(line) <= 60
    if asm or reg:
        return True, "asm"
    if strong or func or FUNCALL.match(line):
        return True, "c"
    if weak:
        return True, "text"
    # 中英混排的代码行（如 `swap(key,lock); //交换…`）；含中文标点的一律视为正文
    if (ASCII_MIX.search(line) and re.search(r"[;,(){}]", line)
            and len(CJK.findall(line)) < 10
            and not re.search(r"[，。；：、！？‘’“”《》]", line)):
        return True, "c"
    return False, None


def escape(line):
    """转义行首 Markdown 标记字符"""
    return ("\\" + line) if line and line[0] in "#>+-*" else line


SECTION = re.compile(r"单项选择|一、单项选择题|单项选择题：")


def render_page(lines, state):
    """把一页的文本行渲染成 Markdown 行列表；state 记录题号期望值与文件类型"""
    kind = state.get("kind", "exam")
    if kind == "answer":
        state["started"] = True          # 解析文件从第 1 题解析开始就是题号
    out = []
    code_buf, code_lang = [], None

    def flush_code():
        nonlocal code_buf, code_lang
        if code_buf:
            out.append("```" + (code_lang or "text"))
            out.extend(code_buf)
            out.append("```")
            out.append("")
            code_buf, code_lang = [], None

    prev_prose = None           # 上一段正文（用于判断是否需要合并）

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if kind == "exam" and not state.get("started") and SECTION.search(line):
            state["started"] = True          # 试卷封面/注意事项之后才开始数题号

        # ---- 题号行 ----
        m = QNUM.match(line)
        heading = None
        if m:
            n = int(m.group(1))
            exp = state.get("expect", 1)
            used = state.setdefault("used", set())
            if (kind == "answer" and n == 1 and exp > 30
                    and ("解析" in line or "解答" in line)):     # 答案速查表（1–40）-> 逐题解析：重新计数
                used.clear()
                exp = 1
            if not state.get("started"):
                pass                                            # 试卷封面/注意事项里的编号不算题号
            elif n not in used and n <= exp + 3:
                # 未出现过的题号都可作为标题（容忍缺号、乱序），重复编号（如选项 I.）不算
                state["started"] = True
                used.add(n)
                state["expect"] = max(exp, n + 1)
                punct, rest = m.group(2), (m.group(3) or "")
                km = KEYLINE.match(rest) if rest else None
                if km:
                    letter, tail = km.group(1), km.group(2).strip()
                    heading = "### %d%s%s" % (n, punct, letter) if tail else None
                    if heading is None:
                        line = escape("%d%s%s" % (n, punct, letter))
                    else:
                        line = tail
                elif rest:
                    heading = "### %d%s%s" % (n, punct, rest)
                    line = None
                elif kind == "exam":                            # 题号独占一行（题干在下一行）
                    heading = "### %d" % n
                    line = None
                else:                                           # 解析里的答案表条目，保持普通行
                    heading = None
                    line = escape("%d%s" % (n, punct))

        # ---- 选项行 ----
        marks = list(OPT_SPLIT.finditer(line)) if line else []
        if marks and marks[0].start() <= 1 and not heading:
            flush_code()
            for i, mk in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(line)
                out.append("- %s%s%s" % (mk.group(1), mk.group(2), line[mk.end():end].strip()))
            prev_prose = None
            continue

        if heading:
            flush_code()
            out.append(heading)
            prev_prose = None
            if line:
                prev_prose = len(out)
                out.append(escape(line))
            continue

        # ---- 代码 ----
        ics, lang = is_code(line)
        if not ics and code_buf:
            # 代码块内部的零碎行（函数名、变量、标点）继续留在块里
            short = len(line) <= 8
            if short or (not CJK_PARA_END.search(line)
                         and (len(CJK.findall(line)) <= 4 or len(line) <= 12)):
                ics, lang = True, None
        if ics:
            if not code_buf:
                out.append("")            # 代码块前后留空行
            code_buf.append(line)
            code_lang = code_lang or (lang if lang != "text" else None)
            prev_prose = None
            continue
        flush_code()

        # ---- 图/表题注 ----
        if CAPTION.match(line):
            out.append("*%s*" % line)
            prev_prose = None
            continue

        # ---- 正文：段落回流 ----
        line = escape(line)
        joined = False
        if prev_prose is not None and state.get("merge", True):
            prev = out[prev_prose]
            if (len(prev) >= 20 and not CJK_PARA_END.search(prev)
                    and not re.match(r"^[-*#|>]|^\(\d+\)|^（\d+）|^\d+[.．、]|^[一二三四五六七八九十]+、", line)):
                out[prev_prose] = prev + line
                joined = True
        if not joined:
            prev_prose = len(out)
            out.append(line)
    flush_code()
    return out


def render_md(pages, title, note, kind="exam"):
    """pages: {pageno: [lines]} -> 完整 Markdown 文本；kind=exam|answer"""
    state = {"kind": kind}
    body = []
    for pno in sorted(pages):
        body.append("## 第 %d 页\n" % pno)
        body.extend(render_page(pages[pno], state))
        body.append("")
    text = "\n".join(body)
    text = re.sub(r"\n{3,}", "\n\n", text)          # 压缩多余空行
    return "# %s\n\n> %s\n\n%s\n" % (title, note, text.rstrip() + "\n")
