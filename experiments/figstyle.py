# -*- coding: utf-8 -*-
"""图表统一样式、双语文案与绘图小工具。

设计约束（都是踩过坑之后定下的）：

1. **图内文字随语言切换**。英文版与中文版由同一段绘图代码生成，脚本里用
   ``tr(中文, English)`` 就地取词，不存在两套绘图逻辑，也就不会出现
   「改了中文忘了英文」的漂移。整个流程只是换个 ``LANG`` 重跑一次。
2. **输出两种形态**：PNG（600 dpi，供 Word 插图）与 PDF（矢量，供投稿）。
   ``pdf.fonttype=42`` 保证字体以 TrueType 内嵌，避免期刊排版不接受的 Type 3。
3. **不用 Unicode 上下标**：SimHei 没有上标数字这类字形，会渲染成方框
   （实测 ``Glyph 8310 missing``）。上下标一律走 mathtext，希腊字母同理。
4. **字号按版面标定**：图在版面上占 5.9 英寸宽，正文不小于 6.2 pt、
   轴标签 8 pt，缩放后仍可读。
5. 配色为低饱和色盲友好方案；白底深字（图进 Word 文档，不能跟着 IDE 主题走）。
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Annotation, Text

# 中文用微软雅黑：SimHei 没有粗体，加粗会静默退化成常规体
# （实测 `findfont: Failed to find font weight bold for SimHei`），
# 而 msyhbd.ttc 是真粗体，注册后 fontweight="bold" 才起作用。
CN_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
]

# ---------------------------------------------------------------- 路径

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG_ROOT = os.path.join(RES, "figures")

# ---------------------------------------------------------------- 配色

C = {
    "blue": "#2b6cb0",
    "red": "#c0392b",
    "green": "#2f855a",
    "amber": "#b7791f",
    "purple": "#6b46c1",
    "teal": "#2c7a7b",
    "gray": "#4a5568",
    "lightgray": "#a0aec0",
}
SERIES = [C["blue"], C["red"], C["green"], C["amber"], C["purple"], C["teal"], C["gray"]]
TEXT = "#1a202c"
MUTED = "#4a5568"
FAINT = "#cbd5e0"

# 示意图用的底色（都很浅，打印后仍是白底观感）
BAND_FC = {"chain": "#f0f6fd", "decide": "#f4f7f2", "exec": "#fbf5ee"}
BAND_EC = {"chain": "#a9c4de", "decide": "#b3c9a4", "exec": "#dfc39a"}

# ---------------------------------------------------------------- 数学记号

RS = r"$\rho_{\mathrm{struct}}$"
RT = r"$\rho_{\mathrm{trans}}$"
RB = r"$\rho_{\mathrm{behav}}$"
RHO = r"$\rho$"
LAM = r"$\lambda$"
DT = r"$\Delta t$"
DTS = r"$\Delta t^{*}$"
GR = r"$g$"
TH = r"$\theta$"
TC = r"$t_c$"
TV = r"$t_v$"

# ---------------------------------------------------------------- 语言

LANG = "cn"


def set_lang(lang: str) -> None:
    """切换语言并重设字体。cn 用微软雅黑（含真粗体），en 用 DejaVu Sans。"""
    global LANG
    assert lang in ("cn", "en"), lang
    LANG = lang
    if lang == "cn":
        for f in CN_FONTS:
            if os.path.exists(f):
                try:
                    fm.fontManager.addfont(f)
                except Exception:
                    pass
        fam = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    else:
        fam = ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fam,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.labelsize": 8.0,
        "legend.fontsize": 6.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# 语言相关的图集目录名
def fig_dir(lang: str = None) -> str:
    return os.path.join(FIG_ROOT, lang or LANG)


def tr(cn: str, en: str) -> str:
    """就地取词：中文版返回 cn，英文版返回 en。"""
    return cn if LANG == "cn" else en


# ---------------------------------------------------------------- 数据读取

def load(name: str):
    """读 results/ 下的 CSV，空字段置 None，能转数的转数。"""
    with open(os.path.join(RES, name), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if not isinstance(v, str):
                # 取值是 list，说明该行字段数多于表头：某个字段含逗号却没加引号。
                # 直接崩在 .strip() 上会把病因指到读表的地方，这里点名到列。
                raise ValueError(
                    "%s 第 %d 行有未转义的含逗号字段（列 %r）；生成端必须用 "
                    "csv.writer 写出" % (name, len(out) + 2, k))
            v = v.strip()
            if v == "":
                d[k] = None
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        out.append(d)
    return out


def sel(rows, cond=None, **conds):
    """按等值条件筛行：sel(rows, 对照="基准")。

    列名与 Python 关键字冲突时用字典形式：sel(rows, {"lambda": 50.0, "g": 1})。
    数据表里确有一列叫 lambda，关键字实参写不出来。
    """
    if cond:
        conds = dict(cond, **conds)
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in conds.items()):
            out.append(r)
    return out


def col(rows, key):
    return [r[key] for r in rows]


# ---------------------------------------------------------------- 画布与装饰

def axes_fig(w: float, h: float):
    """普通数据图：单坐标轴，四周留出轴标签空间。"""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.085, 0.13, 0.895, 0.82])
    style_axes(ax)
    fig._panel_cursor = 0
    return fig, ax


def style_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=2.4, width=0.7, pad=1.8)
    if grid:
        ax.grid(True, which="major", color=FAINT, linewidth=0.45,
                linestyle=(0, (2.5, 2.5)), zorder=0)
        ax.set_axisbelow(True)


def grid_fig(w: float, h: float, nrow: int, ncol: int, **kw):
    """多面板数据图，返回 (fig, axs 二维数组)。"""
    fig = plt.figure(figsize=(w, h))
    gs = fig.add_gridspec(nrow, ncol, **kw)
    axs = fig.add_subplot  # 占位，避免静态检查误报
    axes = [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)]
    for row in axes:
        for a in row:
            style_axes(a)
    return fig, axes


def diag_fig(w: float, h: float):
    """示意图：坐标轴以英寸为单位铺满整张图，1 单位 = 1 英寸。"""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def panel(ax, label, dx=0.0, dy=1.04):
    """在轴的左上角外侧标 (a)/(b)/(c)。放到轴外，免得压住轴内的图例。"""
    ax.text(dx, dy, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, fontweight="bold", color=TEXT)


def legend(ax, ncol=1, loc="best", frame=False, **kw):
    """图例。默认无框，在示意图里更干净；数据曲线压到图例文字上时用
    frame=True 给白底不透明框----曲线从字缝里穿过比遮住一小段难看得多。"""
    common = dict(ncol=ncol, handlelength=1.7, handletextpad=0.5,
                  columnspacing=1.1, borderaxespad=0.25, labelspacing=0.32)
    if frame:
        lg = ax.legend(loc=loc, frameon=True, facecolor="white",
                       edgecolor="#c9c9c9", framealpha=0.96, borderpad=0.45,
                       **common, **kw)
    else:
        lg = ax.legend(loc=loc, frameon=False, **common, **kw)
    return lg


# ---------------------------------------------------------------- 示意图元件

def rbox(ax, x, y, w, h, text="", fc="white", ec=None, lw=0.8, fs=6.8,
         bold=False, tc=None, radius=0.05, z=2, ha="center", va="center",
         pad_x=0.5, linespacing=1.5):
    """圆角矩形 + 居中文字。坐标单位为英寸。"""
    ec = ec or C["gray"]
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=%.4f" % radius,
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z,
        mutation_aspect=1.0)
    ax.add_patch(b)
    if text:
        tx = x + w * pad_x if ha == "left" else x + w / 2.0
        tt = ax.text(tx, y + h / 2.0, text, ha=ha, va=va, fontsize=fs,
                     color=tc or TEXT, zorder=z + 1, linespacing=linespacing,
                     fontweight="bold" if bold else "normal")
        check_fit(ax, tt, x, y, w, h, what="rbox")
    return b


def band(ax, x, y, w, h, kind="chain", label="", fs=7.4):
    """示意图层级底框（很浅的填充，打印后接近白底）。"""
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.07",
                       linewidth=0.8, facecolor=BAND_FC[kind],
                       edgecolor=BAND_EC[kind], zorder=1)
    ax.add_patch(b)
    if label:
        tt = ax.text(x + 0.14, y + h - 0.13, label, ha="left", va="center",
                     fontsize=fs, fontweight="bold", color=TEXT, zorder=2)
        if x + 0.14 + (tt.get_window_extent(renderer=_renderer(ax.figure)).width
                       / float(ax.figure.dpi)) > x + w - 0.10:
            print("  ~ band 标题超出底框: [%s]" % label[:48])
    return b


def arrow(ax, p0, p1, color=None, lw=0.9, ls="-", style="-|>", rad=0.0,
          z=3, ms=8):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        linewidth=lw, color=color or C["gray"],
                        linestyle=ls, zorder=z, shrinkA=0.0, shrinkB=0.0,
                        connectionstyle="arc3,rad=%.3f" % rad)
    ax.add_patch(a)
    return a


def line(ax, p0, p1, color=None, lw=0.9, ls="-", z=3):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color or C["gray"],
            linewidth=lw, linestyle=ls, zorder=z, solid_capstyle="butt")


def tick_marks(ax, x, y, w, h, n, color=None, z=4):
    """在 [x, x+w] 上均匀画 n 条短竖线，用来表示记录流。"""
    for i in range(n):
        xx = x + w * (i + 0.5) / n
        line(ax, (xx, y), (xx, y + h), color=color or C["teal"], lw=0.7, z=z)


# ---------------------------------------------------------------- 文本量测与折行

_WCACHE = {}


def _renderer(fig):
    """Agg 画布不 draw() 也能拿到渲染器，量文字不必重画整张图。"""
    r = getattr(fig, "_wb_renderer", None)
    if r is None:
        r = fig.canvas.get_renderer()
        fig._wb_renderer = r
    return r


def measure_in(ax, s, fs, weight="normal"):
    """量一段文字在版面里的英寸宽度（不落笔、不残留）。"""
    fig = ax.figure
    t = ax.text(0.0, 0.0, s, fontsize=fs, fontweight=weight)
    bb = t.get_window_extent(renderer=_renderer(fig))
    t.remove()
    return (bb.x1 - bb.x0) / float(fig.dpi)


def _atoms(s):
    """切折行原子：$..$ 整段、CJK 单字、拉丁词、空格。

    $..$ 必须整段不可拆（拆开就成了两段数学式）；中文没有词间空格，
    只能逐字累加，所以 CJK 单字成原子。
    """
    out, buf, i = [], "", 0
    while i < len(s):
        ch = s[i]
        if ch == "$":
            j = s.find("$", i + 1)
            if j > i:
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(s[i:j + 1])
                i = j + 1
                continue
        if ch == " ":
            if buf:
                out.append(buf)
                buf = ""
            out.append(" ")
            i += 1
            continue
        if ord(ch) > 0x2E7F:          # CJK 与全角标点
            if buf:
                out.append(buf)
                buf = ""
            out.append(ch)
            i += 1
            continue
        buf += ch
        i += 1
    if buf:
        out.append(buf)
    return out


def _atom_w(ax, a, fs, weight):
    k = (a, fs, weight, LANG)
    w = _WCACHE.get(k)
    if w is None:
        w = measure_in(ax, a, fs, weight)
        _WCACHE[k] = w
    return w


def wrap_to(ax, s, width_in, fs, weight="normal"):
    """按「可用英寸」折行，返回含 \n 的字符串；原有 \n 视为强制换行。

    宽度用原子的宽度累加（每个原子只量一次并缓存），不做逐字试探，
    因此一张图里几百次折行也只是一次量测的开销。
    """
    out_lines = []
    for seg in s.split("\n"):
        cur, curw = "", 0.0
        for a in _atoms(seg):
            aw = _atom_w(ax, a, fs, weight)
            if cur.strip() and curw + aw > width_in:
                out_lines.append(cur.rstrip())
                if a == " ":
                    cur, curw = "", 0.0
                    continue
                cur, curw = a, aw
            else:
                cur += a
                curw += aw
        out_lines.append(cur.rstrip())
    return "\n".join(out_lines)


def check_fit(ax, t, x, y, w, h, what="框", tol=0.04):
    """文字是否撑破所在小框；超了就打印一行，供批量生成时定位。

    英文文案比中文长约 1.6 倍，同一个框中文装得下、英文装不下，
    这类问题肉眼看拼版图很容易漏（差一点点），必须量。
    """
    fig = ax.figure
    bb = t.get_window_extent(renderer=_renderer(fig))
    dpi = float(fig.dpi)
    tw, th = (bb.x1 - bb.x0) / dpi, (bb.y1 - bb.y0) / dpi
    if tw > w - tol or th > h:
        print("  ~ %s 文字超框: 文字 %.2f×%.2f in vs 框 %.2f×%.2f  [%s]"
              % (what, tw, th, w, h, t.get_text().replace("\n", "/")[:40]))
        return False
    return True


# ---------------------------------------------------------------- 绘制期自查


def _clip_seg(x0, y0, x1, y1, bx0, by0, bx1, by1):
    """线段落在矩形内的长度（像素）。不在矩形内返回 None。

    Liang-Barsky 裁剪：数据线往往只有几个顶点，两个端点都在文字框外、
    中间一段横穿文字的情况必须靠线段与矩形的交叠长度判断，不能只看顶点。
    """
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        rr = q / p
        if p < 0:
            if rr > t1:
                return None
            if rr > t0:
                t0 = rr
        else:
            if rr < t0:
                return None
            if rr < t1:
                t1 = rr
    if t0 > t1:
        return None
    f = t1 - t0
    return ((dx * f) ** 2 + (dy * f) ** 2) ** 0.5


def audit_texts(fig, num, slug, w_in, h_in, margin=0.012, overlap=0.22,
                pad_px=2.0, min_px=5.0, core=0.25):
    """量出每个文本的英寸占位，报越界与重叠。

    越界＝文本超出画布（打印或缩放后会被裁掉）；
    重叠＝两个文本的矩形交叠超过较小者面积的 overlap 倍（视觉上糊在一起）。
    返回 (越界列表, 重叠列表)，并直接打印，便于在批量生成时定位。
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    out_bad, out_over, out_cross = [], [], {}
    masked = set()   # 白底框盖住了背后的曲线，这类文字不参与「曲线压字」判定
    for ai, ax in enumerate(fig.axes):
        texts = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        # 刻度标签、轴标题与偏移文本都不在 ax.texts 里，但同样会越界、会互相压。
        # 只纳入「会真正画出来」的刻度：轴可见，且刻度位置在当前视野之内
        # （对数定位器会给视野外的刻度也造标签，位置远在画布外，属假越界）。
        if ax.axison:
            for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                lo, hi = min(lim), max(lim)
                for tk in axis.get_major_ticks():
                    loc = tk.get_loc()
                    if not (lo - 1e-12 <= loc <= hi + 1e-12):
                        continue
                    # y 轴画在右侧时用 label2，label1 被置为不可见；两份都要量
                    for lab in (tk.label1, tk.label2):
                        if lab.get_visible() and lab.get_text().strip():
                            texts.append(lab)
                for extra in (axis.get_label(), axis.get_offset_text()):
                    if extra.get_text().strip():
                        texts.append(extra)
        lg = ax.get_legend()
        if lg is not None:
            lgt = [t for t in lg.get_texts()
                   if t.get_visible() and t.get_text().strip()]
            texts += lgt
            # 带不透明框的图例压在数据上是有意为之，框挡住了曲线，
            # 不能再按「曲线压字」报错（否则 framed 图例永远过不了自查）
            if lg.get_frame_on():
                masked |= {id(t) for t in lgt}
        boxes = []
        for t in texts:
            # Annotation 的 get_window_extent 会把箭头包围盒并进来，
            # 斜箭头于是变成一块大矩形，凭空报出重叠。只量文字本体。
            if isinstance(t, Annotation):
                bb = Text.get_window_extent(t, renderer=r)
            else:
                bb = t.get_window_extent(renderer=r)
            x0, y0 = bb.x0 / dpi, bb.y0 / dpi
            x1, y1 = bb.x1 / dpi, bb.y1 / dpi
            boxes.append((t, x0, y0, x1, y1))
            if x0 < -margin or y0 < -margin or x1 > w_in + margin or y1 > h_in + margin:
                out_bad.append({
                    "ax": ai, "text": t.get_text().replace("\n", "/")[:60],
                    "x0": round(x0, 3), "x1": round(x1, 3),
                    "y0": round(y0, 3), "y1": round(y1, 3)})
        for i in range(len(boxes)):
            _, ax0, ay0, ax1, ay1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                _, bx0, by0, bx1, by1 = boxes[j]
                ix = min(ax1, bx1) - max(ax0, bx0)
                iy = min(ay1, by1) - max(ay0, by0)
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                a1 = max((ax1 - ax0) * (ay1 - ay0), 1e-9)
                a2 = max((bx1 - bx0) * (by1 - by0), 1e-9)
                if inter / min(a1, a2) >= overlap:
                    out_over.append({
                        "ax": ai,
                        "a": boxes[i][0].get_text().replace("\n", "/")[:34],
                        "b": boxes[j][0].get_text().replace("\n", "/")[:34],
                        "frac": round(inter / min(a1, a2), 2)})
        # 数据线压字：文字与文字的重叠能量，文字被曲线压住量不出，而后者
        # 在图上同样读不出来（实测 fig05(a) 的两处都是这一类）。判定分两级：
        # 先按包围盒内缩 core 得到「字形核心区」，曲线穿过核心区才算压字；
        # 只在包围盒边角上蹭一下就归入轻微。单纯用交叠长度会误判----陡线切
        # 角时盒内路径可以很长，实际只是擦过（实测 fig06 的 Δt 标签）。
        lines_px = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xd, yd = ln.get_xdata(), ln.get_ydata()
            if len(xd) < 2:
                continue
            lname = (ln.get_label() or "").strip()
            if lname.startswith("_"):
                lname = ""
            # 必须用折线自身的 transform：`axhline`/`axvline` 的 transform 是
            # blended（轴坐标 × 数据坐标），拿 ax.transData 去变换会把它们算成
            # 横跨整张画布的超长斜线，凭空报出「曲线压住刻度标签」。
            pts = ln.get_transform().transform(list(zip(xd, yd)))
            if len(pts):
                lines_px.append((lname, pts))
        for t, x0, y0, x1, y1 in boxes:
            if id(t) in masked:
                continue
            w_px, h_px = (x1 - x0) * dpi, (y1 - y0) * dpi
            cx, cy = w_px * core, h_px * core
            boxes_px = {
                "soft": (x0 * dpi + pad_px, y0 * dpi + pad_px,
                         x1 * dpi - pad_px, y1 * dpi - pad_px, 1.0),
                "hard": (x0 * dpi + cx, y0 * dpi + cy,
                         x1 * dpi - cx, y1 * dpi - cy, min_px),
            }
            for lname, pts in lines_px:
                best = {"soft": 0.0, "hard": 0.0}
                for lvl, (bx0, by0, bx1, by1, thr) in boxes_px.items():
                    if bx1 <= bx0 or by1 <= by0:
                        continue
                    for k in range(len(pts) - 1):
                        seg = _clip_seg(pts[k][0], pts[k][1],
                                        pts[k + 1][0], pts[k + 1][1],
                                        bx0, by0, bx1, by1)
                        if seg is not None and seg > best[lvl]:
                            best[lvl] = seg
                key = (ai, t.get_text().replace("\n", "/")[:44], lname)
                prev = out_cross.get(key, (0.0, 0.0, 0.0, 0.0))
                out_cross[key] = (max(prev[0], best["soft"]),
                                  max(prev[1], best["hard"]),
                                  w_px, h_px)
    # 贴边告警：不越界但离画布边缘过近，进版心后会紧贴正文，同样不合格
    out_tight = []
    for ax in fig.axes:
        for t in list(ax.texts) + ([t for t in ax.get_legend().get_texts()]
                                   if ax.get_legend() is not None else []):
            if not (t.get_visible() and t.get_text().strip()):
                continue
            bb = t.get_window_extent(renderer=r)
            d = min(bb.x0 / dpi, bb.y0 / dpi, w_in - bb.x1 / dpi, h_in - bb.y1 / dpi)
            if -margin < d < 0.03:
                out_tight.append((t.get_text().replace("\n", "/")[:40], round(d, 3)))
    if out_tight:
        print("  ~~ fig%02d_%s 贴边 %d 处:" % (num, slug, len(out_tight)))
        for txt, d in out_tight:
            print("     距边 %.3f in  %s" % (d, txt))

    tag = "fig%02d_%s" % (num, slug)
    if out_bad:
        print("  !! %s 越界 %d 处:" % (tag, len(out_bad)))
        for b in out_bad:
            print("     ax%d x[%.2f,%.2f] y[%.2f,%.2f]  %s"
                  % (b["ax"], b["x0"], b["x1"], b["y0"], b["y1"], b["text"]))
    if out_over:
        print("  !! %s 重叠 %d 处:" % (tag, len(out_over)))
        for o in out_over:
            print("     ax%d %.0f%%  [%s] x [%s]"
                  % (o["ax"], o["frac"] * 100, o["a"], o["b"]))
    hard = {k: v for k, v in out_cross.items() if v[1] >= min_px}
    soft = {k: v for k, v in out_cross.items() if k not in hard and v[0] >= min_px}
    if hard:
        # 曲线穿过字形核心区：真正读不出来的压字
        print("  !! %s 曲线压字 %d 处:" % (tag, len(hard)))
        for (ai_, txt, lname), (sp, hp, bw, bh) in sorted(
                hard.items(), key=lambda kv: -kv[1][1]):
            print("     ax%d 穿心 %.3f in（文字 %.2f×%.2f in）[%s] × 曲线[%s]"
                  % (ai_, hp / dpi, bw / dpi, bh / dpi, txt, lname or "无标签"))
    if soft:
        # 只在包围盒边角上蹭过：多半是数值标签紧贴数据点，字形未被动到
        print("  ~~ %s 曲线擦字 %d 处（轻微，未计入不通过）:" % (tag, len(soft)))
        for (ai_, txt, lname), (sp, hp, bw, bh) in sorted(
                soft.items(), key=lambda kv: -kv[1][0]):
            print("     ax%d %.3f in  [%s] × 曲线[%s]"
                  % (ai_, sp / dpi, txt, lname or "无标签"))
    if not out_bad and not out_over and not hard:
        print("  ok %s 版面自查通过" % tag)
    return out_bad, out_over, hard


# ---------------------------------------------------------------- 输出与清单

MANIFEST = []
PARTIAL = False   # 生成器用 --only 只重跑部分图时置 True，见 write_manifest


def save(fig, num: int, slug: str, cap_cn: str, cap_en: str,
         w_in: float, h_in: float):
    """存 PNG + PDF，并登记到 manifest。num 为论文里的图号。"""
    audit_texts(fig, num, slug, w_in, h_in)
    d = fig_dir()
    os.makedirs(d, exist_ok=True)
    base = "fig%02d_%s" % (num, slug)
    png = os.path.join(d, base + ".png")
    pdf = os.path.join(d, base + ".pdf")
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)
    MANIFEST.append({
        "num": num, "slug": slug, "lang": LANG,
        "png": os.path.relpath(png, HERE).replace("\\", "/"),
        "pdf": os.path.relpath(pdf, HERE).replace("\\", "/"),
        "caption_cn": cap_cn, "caption_en": cap_en,
        "width_in": round(w_in, 3), "height_in": round(h_in, 3),
        "width_px": int(round(w_in * 96)), "height_px": int(round(h_in * 96)),
    })
    print("  [%s] %-34s %.1f KB" % (LANG, base + ".png",
                                    os.path.getsize(png) / 1024.0))


def write_manifest(fname="manifest.json"):
    """把两个语言的登记合并成一份 manifest，供插图和投稿使用。

    PARTIAL 为 True（生成器只跑了 --only 的那几张）时**保留**同语言的旧条目，
    否则一次 `--only 7` 会把同一篇论文其余 11 张的登记抹掉 ---- 而 build_docx.py
    是按清单插图号的，删掉即静默少插，只报「插入 1/12 张」而不是报错。

    fname 让**同一图集的多套图**各写各的清单。论文一投 SR 时把 16 张原图两两
    合并成 8 张，合并图号 1..8 与原图号 1..16 大面积重叠；而本函数的去重键是
    (num, lang)、build_docx.load_figures 又返回 {num: m} ---- 同语言下 num 必须
    唯一，两套图**无法共存于一份清单**。早先只写一份时，跑 `--set sr` 会把图集
    里 full 的 16 图登记整批覆盖（非 PARTIAL 分支只保留另一个语言的旧条目），
    而 build_docx.py 里论文一从 v3.7.1 起的**全部历史作业**都指着
    figures/manifest.json，重建旧版 docx 会静默少插图。故拆成
    manifest.json（full 16 图）与 manifest_sr.json（SR 8 图）两份。
    """
    p = os.path.join(FIG_ROOT, fname)
    old = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            old = [m for m in json.load(f) if m.get("lang") != LANG or PARTIAL]
    merged = old + MANIFEST
    # MANIFEST 在一次运行里跨语言累积（--lang both 会先后 append 中英两套），
    # 而本函数每轮语言都写一次，不去重就会让中文条目翻倍
    uniq = {}
    for m in merged:
        uniq[(m["num"], m["lang"])] = m
    merged = sorted(uniq.values(), key=lambda m: (m["num"], m["lang"]))
    with open(p, "w", encoding="utf-8", newline="") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print("  manifest: %d 条 -> %s" % (len(merged), p))
    return merged
