# -*- coding: utf-8 -*-
"""生成论文一 SR 投稿版（8 图）的图集 figures_sr39。

输入是**手绘 SVG 源**（`experiments/results/figures_svg_src/Fig1..9.svg`，
即 v3.9.37 基线那一批设计稿），输出 600 dpi PNG 与 `manifest.json`。

本脚本做的两件事：

1. **Fig.7 拆成 a+b 两面板**。原 Fig.7 是四面板合图，其中 c/d 是检出率与误报率的
   重复呈现，SI 补充表 3 已有更全的十行版本；正文只承担「决策开销与路径数开销」，
   故按 y 坐标切掉下半部分（CUT 以下），画布同步改高，尺寸从 725 收到 340。
2. **Fig.1、Fig.2、Fig.6 按 SVG 源重渲染**，其余四张直接复用 `figures_sr35` 的 PNG。

图内文字**只在 SVG 源里改**，脚本不做运行时替换；否则源与脚本两份文本会互相打架
（上一轮就把替换写在这里，改源之后锚点必然失效）。

依赖 `_tools/render_svg.py`（Chrome headless）。Chrome 不在默认路径时设 `CHROME_EXE`。
"""
import io
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_svg import render, png_size          # noqa: E402

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PAPER, "experiments", "results", "figures_svg_src")
OLD = os.path.join(PAPER, "experiments/results/figures_sr35/en")
DST = os.path.join(PAPER, "experiments/results/figures_sr39")
W_PX = 4016          # 170 mm @ 600 dpi，与 sr35 批一致

os.makedirs(os.path.join(DST, "en"), exist_ok=True)


def read_svg(n):
    return io.open(os.path.join(SRC, "Fig%d.svg" % n), encoding="utf-8").read()


# ---------------------------------------------------------------- Fig1 / Fig2 / Fig6
def fig1():
    """治理参照与准入路径。图内底注已在 SVG 源改为正面陈述。"""
    return read_svg(1), 1200, 500


def fig2():
    """证据构造与最终处置。面板分隔注已在 SVG 源改为 Two reference populations。"""
    return read_svg(2), 1200, 860


def fig6():
    """数据使用与评估路径。五处 separate 已在 SVG 源改写。"""
    return read_svg(6), 1200, 760


# ---------------------------------------------------------------- Fig7 -> a+b
CUT = 340.0          # 上半（a、b 面板）到此为止；c/d 从 382 起


def fig7ab():
    t = read_svg(7)
    dropped = {"text": 0, "rect": 0, "polyline": 0, "circle": 0}

    def y_of(tag, attrs):
        if tag == "rect":
            m = re.search(r'\by="([-\d.]+)"', attrs)
            return float(m.group(1)) if m else 0.0
        if tag == "circle":
            return float(re.search(r'\bcy="([-\d.]+)"', attrs).group(1))
        if tag == "polyline":
            pts = re.search(r'points="([^"]*)"', attrs).group(1)
            ys = [float(p.split(",")[1]) for p in pts.split() if "," in p]
            return min(ys)
        if tag == "text":
            m = re.search(r'\by="([-\d.]+)"', attrs)
            return float(m.group(1)) if m else 0.0
        return 0.0

    # 1) 成对元素 <text>...</text>、<polyline .../> 等
    def repl_pair(m):
        tag, attrs = m.group(1), m.group(2)
        if tag == "text" and y_of("text", attrs) >= CUT:
            dropped["text"] += 1
            return ""
        return m.group(0)

    t = re.sub(r'<(text)\b([^>]*)>.*?</\1>', repl_pair, t, flags=re.S)

    def repl_self(m):
        tag, attrs = m.group(1), m.group(2)
        if tag in ("polyline", "circle", "rect") and y_of(tag, attrs) >= CUT:
            dropped[tag] += 1
            return ""
        return m.group(0)

    t = re.sub(r'<(polyline|circle|rect)\b([^>]*)/>', repl_self, t)

    # 2) 白底 rect 高度与画布
    assert t.count('<rect width="740" height="725" fill="white"/>') == 1
    t = t.replace('<rect width="740" height="725" fill="white"/>',
                  '<rect width="740" height="%d" fill="white"/>' % int(CUT))
    m = re.search(r'<svg\b[^>]*>', t)
    head = m.group(0)
    new_head = head.replace('height="166.554mm"', 'height="%.3fmm"' % (166.554 * CUT / 725))
    new_head = re.sub(r'viewBox="0 0 740 725"', 'viewBox="0 0 740 %d"' % int(CUT), new_head)
    t = t.replace(head, new_head)
    return t, 740, int(CUT), dropped


# ---------------------------------------------------------------- 主流程
def emit(name, svg_text, vb_w, vb_h, note):
    png = os.path.join(DST, "en", name)
    h_px = int(round(W_PX * vb_h / vb_w))
    render(svg_text, W_PX, h_px, png)
    w, h = png_size(png)
    assert (w, h) == (W_PX, h_px), (w, h, W_PX, h_px)
    print("  %-26s %4dx%-4d  %6.1f KB  %.2f x %.2f in  %s"
          % (name, w, h, os.path.getsize(png) / 1024,
             6.69, 6.69 * h / w, note))
    return w, h


MAN = []


def add(num, rel, w, h, cap):
    MAN.append({
        "num": num, "slug": "sr39_fig%d" % num, "lang": "en",
        "png": "results/figures_sr39/en/" + rel,
        "pdf": "", "caption_cn": "", "caption_en": cap,
        "width_in": 6.69, "height_in": round(6.69 * h / w, 4),
        "width_px": int(round(6.69 * 96)), "height_px": int(round(6.69 * h / w * 96)),
    })


COPY = [(4, "fig04_sr35_fig4.png"),
        (5, "fig05_sr35_fig5.png"), (8, "fig08_sr35_fig8.png"),
        (9, "fig09_sr35_fig9.png")]

print("== 重渲染（图形有改动）==")
t1, a, b = fig1()
w, h = emit("fig01_sr39_fig1.png", t1, a, b, "底注改正面陈述")
add(1, "fig01_sr39_fig1.png", w, h, "治理参照、本地准入与异步审计")

t2, a, b = fig2()
w, h = emit("fig02_sr39_fig2.png", t2, a, b, "面板注改写")
add(2, "fig02_sr39_fig2.png", w, h, "证据构造与最终处置")

t6, a, b = fig6()
w, h = emit("fig05_sr39_fig6.png", t6, a, b, "五处 separate 改写")
add(5, "fig05_sr39_fig6.png", w, h, "数据使用与评估路径")

t7, a, b, dropped = fig7ab()
w, h = emit("fig06_sr39_fig7ab.png", t7, a, b, "拆出 a+b；删 %s" % dropped)
add(6, "fig06_sr39_fig7ab.png", w, h, "决策开销与路径数开销")

print("== 复用原图（未改图形）==")
NUM = {2: 2, 4: 3, 5: 4, 8: 7, 9: 8}
for old_n, fn in COPY:
    src = os.path.join(OLD, fn)
    new_n = NUM[old_n]
    rel = "fig%02d_sr39_fig%d.png" % (new_n, new_n)
    shutil.copy2(src, os.path.join(DST, "en", rel))
    w, h = png_size(os.path.join(DST, "en", rel))
    print("  %-26s %4dx%-4d  %6.1f KB  (原 Fig%d)"
          % (rel, w, h, os.path.getsize(src) / 1024, old_n))
    add(new_n, rel, w, h, "")

MAN.sort(key=lambda x: x["num"])
assert [m["num"] for m in MAN] == list(range(1, 9)), [m["num"] for m in MAN]
json.dump(MAN, io.open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("manifest: 8 条 ->", os.path.join(DST, "manifest.json"))
for m in MAN:
    print("  %d  %-26s %.2f x %.2f in" % (m["num"], m["png"].split("/")[-1],
                                          m["width_in"], m["height_in"]))
