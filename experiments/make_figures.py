# -*- coding: utf-8 -*-
"""论文一插图生成。

用法：
    python make_figures.py                # 中英双份
    python make_figures.py --lang cn      # 只出中文
    python make_figures.py --lang en      # 只出英文
    python make_figures.py --only 5 10    # 只出指定图号

数值全部现场读 results/*.csv，脚本里不写死任何实验结果 —— 与正文回填同一口径。
图 1 至图 4 为示意图（内容来自方法章的设计），图 5 至图 16 为数据图与对比图。
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *  # noqa: F401,F403
import figstyle as F


# ================================================================ 画布工厂
# 合并出图：同一段绘图代码既能单独成图，也能挂到外层画布的子区域上。
# _OUTER 非空时进入合并模式，_mk/_diag/_mk_axes 把「新建 figure」换成
# 「在外层画布划出的子区域里按原尺寸建坐标轴」，因此图内文字的绝对字号、
# 元素间距与单独出图时逐点一致，不存在缩放导致的压字。

from matplotlib.gridspec import GridSpec  # noqa: E402

_OUTER = None          # 合并模式：(figure, SubplotSpec)
_CUR = {}              # 独立模式下最近一次建的 figure
_PANEL_SEQ = []        # 当前图的面板标签序列，按绘制顺序 pop


def _panel_labels():
    return ["(%s)" % chr(ord("a") + i) for i in range(26)]


def pnext():
    """取下一个面板标签 (a)(b)(c)...，合并时跨原图连续编号。"""
    return _PANEL_SEQ.pop(0) if _PANEL_SEQ else "(?)"


def _block_bbox(fig, sub):
    """子区域在 figure 坐标下的 bbox。"""
    return sub.get_position(fig)


def _inner_grid(fig, sub, nrow, ncol, **kw):
    """在子区域内按原图的比例边距建 GridSpec，保证 1:1 复现原布局。"""
    bb = _block_bbox(fig, sub)
    w, h = bb.x1 - bb.x0, bb.y1 - bb.y0
    L = kw.get("left", 0.0)
    R = kw.get("right", 1.0)
    B = kw.get("bottom", 0.0)
    T = kw.get("top", 1.0)
    extra = {}
    if kw.get("height_ratios") is not None:
        extra["height_ratios"] = kw["height_ratios"]
    if kw.get("width_ratios") is not None:
        extra["width_ratios"] = kw["width_ratios"]
    return GridSpec(nrow, ncol, figure=fig,
                    left=bb.x0 + L * w, right=bb.x0 + R * w,
                    bottom=bb.y0 + B * h, top=bb.y0 + T * h,
                    hspace=kw.get("hspace", 0.2), wspace=kw.get("wspace", 0.2),
                    **extra)


def _mk(W, H, nrow, ncol, **kw):
    """建多面板画布。"""
    if _OUTER is not None:
        fig, sub = _OUTER
        return fig, _inner_grid(fig, sub, nrow, ncol, **kw)
    fig = plt.figure(figsize=(W, H))
    _CUR["fig"] = fig
    return fig, fig.add_gridspec(nrow, ncol, **kw)


def _diag(W, H):
    """建示意图画布（1 单位 = 1 英寸）。"""
    if _OUTER is not None:
        fig, sub = _OUTER
        ax = fig.add_subplot(sub)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax
    fig, ax = diag_fig(W, H)
    _CUR["fig"] = fig
    return fig, ax


def _rel_axes(fig, sub, rect):
    """在子区域内按相对矩形建坐标轴。"""
    bb = _block_bbox(fig, sub)
    w, h = bb.x1 - bb.x0, bb.y1 - bb.y0
    return fig.add_axes([bb.x0 + rect[0] * w, bb.y0 + rect[1] * h,
                         rect[2] * w, rect[3] * h])


def _mk_axes(W, H, rect):
    """建单坐标轴画布（按相对矩形定位）。"""
    if _OUTER is not None:
        fig, sub = _OUTER
        return fig, _rel_axes(fig, sub, rect)
    fig = plt.figure(figsize=(W, H))
    _CUR["fig"] = fig
    return fig, fig.add_axes(rect)


def _standalone(body, num, slug, cap_cn, cap_en, W, H):
    """按原样单独出一张图。"""
    global _PANEL_SEQ
    _PANEL_SEQ = _panel_labels()
    _CUR["fig"] = None
    body()
    save(_CUR["fig"], num, slug, cap_cn, cap_en, W, H)


def _stack(num, slug, cap_cn, cap_en, parts, pad_in=0.0, labels=None):
    """把若干 body 纵向堆叠成一张图。

    parts = [(body, W, H), ...]；各块高度按原始英寸分配，块内边距按原比例
    内缩，因此每个面板的物理尺寸与单独出图时完全相同 —— 内容零改动。

    pad_in > 0 时在每个块上方留一条空白带，用来放该块自己的面板标签
    （示意图内部没有 panel() 调用，合并后必须补 (a)(b)）。
    """
    global _OUTER, _PANEL_SEQ
    W = max(p[1] for p in parts)
    rows = []
    for i, (body, _w, h) in enumerate(parts):
        if pad_in > 0:
            rows.append((None, pad_in, i))
        rows.append((body, h, i))
    total = float(sum(r[1] for r in rows))
    fig = plt.figure(figsize=(W, total))
    outer = fig.add_gridspec(len(rows), 1, height_ratios=[r[1] for r in rows],
                             hspace=0.0, left=0.0, right=1.0, top=1.0,
                             bottom=0.0)
    _PANEL_SEQ = _panel_labels()
    for i, (body, _h, k) in enumerate(rows):
        if body is None:
            if labels and k < len(labels) and labels[k]:
                bb = outer[i].get_position(fig)
                fig.text(bb.x0 + 0.014, (bb.y0 + bb.y1) / 2.0, labels[k],
                         ha="left", va="center", fontsize=8.2,
                         fontweight="bold", color=TEXT)
            continue
        _OUTER = (fig, outer[i])
        body()
    _OUTER = None
    save(fig, num, slug, cap_cn, cap_en, W, total)



# ==================================================================== 图 1

def body01():
    """分层结构与信任边界。"""
    W, H = 5.9, 4.40
    fig, ax = _diag(W, H)

    # ---- 链上信任根 ----
    band(ax, 0.10, 2.95, 5.70, 1.39, "chain",
         tr("链上信任根（联盟链，各中心共同维护）",
            "On-chain trust root (consortium chain, jointly maintained)"))
    ax.text(0.24, 4.08,
            tr("五合约族 IDC / PESC / TSC / PDPC / GovC：只做证据与治理，"
               "不进入请求关键路径",
               "Five contract families IDC / PESC / TSC / PDPC / GovC: evidence "
               "and governance only, never on the request critical path"),
            ha="left", va="center", fontsize=6.0, color=MUTED, zorder=4)
    roots = [
        (tr("身份根 IDRoot", "Identity root IDRoot"),
         tr("身份注册与吊销\n（累加器）", "enrolment & revocation\n(accumulator)")),
        (tr("策略根 PolRoot", "Policy root PolRoot"),
         tr("策略版本哈希\n（含监管模板）", "policy version hash\n(with regulatory template)")),
        (tr("信任根 TrustRoot", "Trust root TrustRoot"),
         tr("信任等级快照\n摘要", "trust-level snapshot\ndigest")),
        (tr("审计根 AuditRoot", "Audit root AuditRoot"),
         tr("审计批次根\n（可验证留存）", "audit batch root\n(verifiable retention)")),
    ]
    for i, (t1, t2) in enumerate(roots):
        x = 0.26 + i * 1.365
        rbox(ax, x, 3.46, 1.30, 0.56, "", fc="white", ec=C["blue"], lw=0.9)
        ax.text(x + 0.65, 3.86, t1, ha="center", va="center", fontsize=6.9,
                fontweight="bold", color=TEXT, zorder=4)
        ax.text(x + 0.65, 3.63, t2, ha="center", va="center", fontsize=6.1,
                color=MUTED, zorder=4, linespacing=1.35)

    fam = ["IDC " + tr("身份", "identity"), "PESC " + tr("策略", "policy"),
           "TSC " + tr("信任", "trust"), "PDPC " + tr("决策", "decision"),
           "GovC " + tr("治理", "governance")]
    for i, t in enumerate(fam):
        rbox(ax, 0.20 + i * 1.11, 3.02, 1.06, 0.30, t, fc="#e8f0f9",
             ec=C["blue"], lw=0.6, fs=6.1, tc="#1d4e79")

    # ---- 链上 / 链下 往来 ----
    arrow(ax, (1.30, 2.62), (1.30, 2.95), color=C["amber"], ls=(0, (3.5, 2.2)), lw=1.0)
    ax.text(1.42, 2.855, tr("批量锚定：间隔 $\\Delta t$，粒度 $g$",
                            "Batch anchoring: interval $\\Delta t$, granularity $g$"),
            ha="left", va="center", fontsize=6.3, color="#8a5a00", zorder=4)
    arrow(ax, (4.70, 2.95), (4.70, 2.62), color=C["blue"], lw=1.0)
    ax.text(4.58, 2.755, tr("策略版本与信任证据加载（读缓存）",
                            "policy version & trust evidence load (cache read)"),
            ha="right", va="center", fontsize=6.3, color="#1d4e79", zorder=4)

    # ---- 链下决策面 ----
    band(ax, 0.10, 0.98, 5.70, 1.64, "decide",
         tr("链下决策面（逐请求判定，各中心本地执行）",
            "Off-chain decision plane (per-request, executed locally at each centre)"))
    steps = [
        (tr("① 基线过滤", "1 Baseline filter"),
         tr("角色基线硬边界\n＋归属中心背书\n（确定性）", "role baseline hard bound\n+ home-centre endorsement\n(deterministic)")),
        (tr("② 动态映射", "2 Dynamic mapping"),
         tr("ABAC 属性与\n上下文条件", "ABAC attributes and\ncontextual conditions")),
        (tr("③ 信任分级", "3 Trust grading"),
         tr("信任等级 ≥\n策略要求下界", "trust level $\\geq$\npolicy lower bound")),
        (tr("④ 风险干预", "4 Risk intervention"),
         tr("迟滞：升级 .5\n降级 .4", "hysteresis: up .5\ndown .4")),
    ]
    # 四个步骤框右移并略收窄：左端腾出的空隙给「访问请求」标注用，
    # 否则中文贴着画布边、底框边线还会从文字中间穿过，英文两行更放不下
    for i, (t1, t2) in enumerate(steps):
        x = 0.44 + i * 1.30
        rbox(ax, x, 1.72, 1.24, 0.56, "", fc="white", ec=C["green"], lw=0.9)
        ax.text(x + 0.62, 2.11, t1, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=TEXT, zorder=4)
        ax.text(x + 0.62, 1.90, t2, ha="center", va="center", fontsize=5.9,
                color=MUTED, zorder=4, linespacing=1.35)
        if i < 3:
            arrow(ax, (x + 1.24, 2.00), (x + 1.30, 2.00), color=C["green"], ms=7)
    arrow(ax, (0.12, 2.00), (0.44, 2.00), color=C["gray"], ms=7)
    ax.text(0.12, 2.06, tr("访问请求", "access\nrequest"), ha="left", va="bottom",
            fontsize=6.3, color=MUTED, zorder=4, linespacing=1.3)

    ax.text(0.44, 1.52,
            tr("风险分量：" + RS + " 客体可达性 ｜ " + RT + " 状态迁移 ｜ " + RB + " 请求级残差",
               "Risk components: " + RS + " object reachability | " + RT
               + " state migration | " + RB + " per-request residual"),
            ha="left", va="center", fontsize=6.4, color=TEXT, zorder=4)
    ax.text(0.44, 1.37,
            tr("三者并列，作用域互不重叠；合成为概率和，给出分级动作档位（B 节、算法 2）",
               "three parallel and scope-disjoint; combined by a probabilistic sum into "
               "the graded action level (Methods B, Algorithm 2)"),
            ha="left", va="center", fontsize=6.0, color=MUTED, zorder=4)

    arrow(ax, (4.92, 1.72), (4.92, 0.86), color=C["green"], lw=1.1)
    ax.text(4.80, 1.10, tr("决策结果（放行 / 收紧 / 降级 / 拒绝）",
                           "decision (permit / refine / degrade / deny)"),
            ha="right", va="center", fontsize=6.3, color="#256b3f", zorder=4)

    # ---- 执行点 ----
    band(ax, 0.10, 0.20, 5.70, 0.66, "exec",
         tr("执行点与跨中心互认", "Enforcement points and cross-centre mutual endorsement"))
    rbox(ax, 0.30, 0.28, 1.70, 0.34, tr("中心 B 执行点 PEP", "Centre B enforcement point PEP"),
         fc="white", ec=C["amber"], fs=6.4)
    rbox(ax, 3.90, 0.28, 1.70, 0.34, tr("中心 A 执行点 PEP", "Centre A enforcement point PEP"),
         fc="white", ec=C["amber"], fs=6.4)
    arrow(ax, (2.20, 0.45), (3.70, 0.45), color=C["amber"], style="<|-|>", lw=0.9)
    ax.text(2.95, 0.56, tr("跨中心互认（门限背书）", "cross-centre endorsement (threshold)"),
            ha="center", va="center", fontsize=6.2, color="#8a5a00", zorder=4)

    ax.text(0.10, 0.10,
            tr("实线位于请求关键路径内；虚线为批量锚定与证据加载，均在关键路径之外",
               "Solid arrows lie on the request critical path; dashed ones are batch "
               "anchoring and evidence loading, both off the critical path"),
            ha="left", va="center", fontsize=6.0, color=MUTED, zorder=4)



def fig01():
    _standalone(body01, 1, "architecture",
                tr("PGB-AC 的分层结构与信任边界",
            "Layered structure and trust boundary of PGB-AC"),
                "Layered structure and trust boundary of PGB-AC", 5.9, 4.40)



# ==================================================================== 图 2

def body02():
    """端到端处理时序与四步决策。"""
    W, H = 5.9, 3.00
    fig, ax = _diag(W, H)

    ax.text(0.10, 2.83, tr("请求关键路径（同步，逐请求）",
                           "Request critical path (synchronous, per request)"),
            ha="left", va="center", fontsize=7.3, fontweight="bold", color=TEXT)
    arrow(ax, (0.05, 2.35), (0.55, 2.35), color=C["gray"], ms=7)
    ax.text(0.05, 2.46, tr("请求到达", "request arrival"), ha="left", va="bottom",
            fontsize=6.2, color=MUTED)

    steps = [
        (tr("① 基线过滤", "1 Baseline"), tr("确定性", "deterministic")),
        (tr("② 动态映射", "2 Mapping"), tr("ABAC 条件", "ABAC cond.")),
        (tr("③ 信任分级", "3 Grading"), tr("$L \\geq L_{\\min}$", "$L \\geq L_{\\min}$")),
        (tr("④ 风险干预", "4 Intervention"), tr("迟滞＋阻尼", "hysteresis")),
    ]
    x0, seg = 0.55, 1.1625
    for i, (t1, t2) in enumerate(steps):
        x = x0 + i * seg
        rbox(ax, x, 2.08, seg, 0.54, "", fc="#f7fbf5", ec=C["green"], lw=0.9)
        ax.text(x + seg / 2, 2.44, t1, ha="center", va="center", fontsize=6.7,
                fontweight="bold", color=TEXT, zorder=4)
        ax.text(x + seg / 2, 2.24, t2, ha="center", va="center", fontsize=6.0,
                color=MUTED, zorder=4)
    arrow(ax, (x0 + 4 * seg, 2.35), (5.76, 2.35), color=C["green"], ms=7)
    ax.text(5.76, 2.46, tr("决策输出", "decision"), ha="right", va="bottom",
            fontsize=6.2, color="#256b3f")

    arrow(ax, (x0, 1.98), (x0 + 4 * seg, 1.98), color=C["blue"],
          style="<|-|>", lw=0.7)
    # 时延符号须与正文一致：D_det 是「证据确认时延」（决策→证据取得链上不可否认性），
    # 这里量的是判定本身，故另设 D_dec。D_det/D_dec 混用是第二轮审稿点出的图文矛盾。
    ax.text(x0 + 2 * seg, 1.87, tr("本次判定时延 " + r"$D_{\mathrm{dec}}$"
                                   + "（不含网络、TLS 与存储 I/O）",
                                   "per-decision evaluation time " + r"$D_{\mathrm{dec}}$"
                                   + " (excludes network, TLS and storage I/O)"),
            ha="center", va="center", fontsize=6.3, color="#1d4e79")

    ax.text(0.10, 1.62, tr("锚定旁路（异步，可跨请求合并）",
                           "Anchoring bypass (async, batchable)"),
            ha="left", va="center", fontsize=7.3, fontweight="bold", color=TEXT)
    arrow(ax, (1.16, 2.08), (1.16, 1.42), color=C["gray"], ls=(0, (3.5, 2.2)), lw=0.9)
    ax.text(1.26, 1.74, tr("决策伴随证据", "decision evidence"), ha="left",
            va="center", fontsize=6.0, color=MUTED)
    boxes = [
        (0.55, 1.35, tr("证据缓冲\n（随决策产出）", "Evidence buffer\n(produced per decision)")),
        # 符号与正文一致：t_v 是批量构建与验证时间，t_c 是共识确认时延
        (1.90, 1.35, tr("Merkle 建根＋签名上链\n代价 " + r"$c_g$" + "，时延 " + r"$t_v$",
                        "Merkle root + signed tx\ncost " + r"$c_g$" + ", latency " + r"$t_v$")),
        (4.05, 1.35, tr("链条确认\n时延 " + r"$t_c$", "Chain confirmation\nlatency " + r"$t_c$")),
    ]
    for i, (x, y, t) in enumerate(boxes):
        w = 1.20 if i == 0 else (1.95 if i == 1 else 1.15)
        rbox(ax, x, 1.05, w, 0.50, t, fc="#fdf8f0", ec=C["amber"], fs=6.0)
        if i < 2:
            x2 = boxes[i + 1][0]
            arrow(ax, (x + w, 1.30), (x2, 1.30), color=C["amber"], ms=7)
    ax.text(0.55, 0.94, tr("按间隔 $\\Delta t$ 或粒度 $g$ 批量触发",
                           "triggered in batches by interval $\\Delta t$ or granularity $g$"),
            ha="left", va="center", fontsize=6.0, color="#8a5a00")

    ax.text(0.10, 0.66,
            wrap_to(ax, tr("异步形态下锚定不在关键路径内，决策输出不等待上链确认；同步形态（对照）"
                           "则逐请求等待，实测 P99 由 1.2 μs 升至 508 ms。",
                           "With asynchronous anchoring the decision never waits for on-chain "
                           "confirmation; in the synchronous control the measured P99 rises "
                           "from 1.2 \u03bcs to 508 ms."), 5.68, 6.2),
            ha="left", va="center", fontsize=6.2, color=TEXT)
    ax.text(0.10, 0.40,
            wrap_to(ax, tr("证据确认时延 $D_{\\det}^{\\max}=\\Delta t+t_c+t_v$ 自决策发生计到证据取得"
                           "链上不可否认性；锚定间隔与粒度由方法 C 节的权衡模型给出，其时间关系"
                           "见图 2（b）。",
                           "The evidence-confirmation latency $D_{\\det}^{\\max}=\\Delta t+t_c+t_v$ "
                           "runs from the decision to the point where its evidence becomes "
                           "non-repudiable on chain; the interval and granularity come from the "
                           "trade-off model of Methods C, and their time relation is shown in "
                           "Fig. 2b."), 5.68, 6.2),
            ha="left", va="center", fontsize=6.2, color=MUTED)



def fig02():
    _standalone(body02, 2, "timing",
                tr("访问请求的端到端处理时序与四步决策",
            "End-to-end request processing timeline and the four-step decision"),
                "End-to-end request processing timeline and the four-step decision", 5.9, 3.00)



# ==================================================================== 图 3

def body03():
    """三个风险分量的判别对象与作用域。"""
    W, H = 5.9, 3.80
    fig, ax = _diag(W, H)
    rows = [
        (2.72, RS, tr("结构偏离度", "Structural deviation"),
         tr("状态空间＝客体集。判据＝目标客体在带重启随机游走平稳分布下的可达性偏离。",
            "State space is the object set. Criterion: reachability deviation of the target "
            "object under the stationary distribution of a restarted random walk."),
         tr("作用域：单条请求对客体的选择。", "Scope: the object chosen by a single request."),
         tr("归属：方法 B 节定义", "Defined in Methods B"), C["blue"]),
        (1.64, RT, tr("状态迁移偏离度", "State-migration deviation"),
         tr("对象＝相邻操作的次序。判据＝最近 $W$ 次迁移的平均对数不似然，按角色分位归一。",
            "Object: the ordering of adjacent operations. Criterion: mean log-unlikelihood of "
            "the last $W$ transitions, normalised by per-role quantiles."),
         tr("作用域：主体操作序列的次序。", "Scope: the ordering of the subject's operation sequence."),
         tr("归属：方法 B 节定义", "Defined in Methods B"), C["purple"]),
        (0.56, RB, tr("时序行为偏离度", "Temporal-behaviour deviation"),
         tr("对象＝单条请求的属性。判据＝请求级行为模型给出的属性残差。",
            "Object: the attributes of a single request. Criterion: attribute residual from a "
            "per-request behavioural model."),
         tr("作用域：请求自身的属性取值。", "Scope: the attribute values of the request itself."),
         tr("归属：信任评估引擎提供", "Provided by the trust evaluation engine"), C["teal"]),
    ]
    for y, sym, name, d1, d2, d3, cc in rows:
        rbox(ax, 0.18, y, 2.77, 0.94, "", fc="#fbfcfe", ec=cc, lw=0.8)
        ax.text(3.05, y + 0.80, sym + "  " + name, ha="left", va="center",
                fontsize=7.0, fontweight="bold", color=cc, zorder=4)
        ax.text(3.05, y + 0.53, wrap_to(ax, d1, 2.70, 6.2), ha="left",
                va="center", fontsize=6.2, color=TEXT, zorder=4)
        ax.text(3.05, y + 0.26, wrap_to(ax, d2, 2.70, 6.2), ha="left",
                va="center", fontsize=6.2, color=TEXT, zorder=4)
        ax.text(3.05, y + 0.07, wrap_to(ax, d3, 2.70, 5.9), ha="left",
                va="center", fontsize=5.9, color=MUTED, zorder=4)

    # 行 A：随机游走示意图
    y = 2.72
    ax.add_patch(plt.Circle((0.48, y + 0.30), 0.085, fc="#e8f0f9",
                            ec=C["blue"], lw=0.8, zorder=4))
    ax.text(0.48, y + 0.30, tr("角色", "role"), ha="center", va="center",
            fontsize=5.2, color="#1d4e79", zorder=5)
    for i in range(4):
        cx = 1.12 + i * 0.44
        hot = (i == 2)
        ax.add_patch(plt.Circle((cx, y + 0.72), 0.105,
                                fc="#dbe9f8" if hot else "white",
                                ec=C["red"] if hot else C["blue"],
                                lw=1.1 if hot else 0.7, zorder=4))
        ax.text(cx, y + 0.72, "$o_%d$" % (i + 1), ha="center", va="center",
                fontsize=5.8, color=TEXT, zorder=5)
    arrow(ax, (0.55, y + 0.36), (1.06, y + 0.66), color=C["blue"], lw=0.7, ms=6)
    arrow(ax, (1.23, y + 0.72), (1.52, y + 0.72), color=C["blue"], lw=0.7, ms=6,
          ls=(0, (2.5, 1.8)))
    arrow(ax, (1.67, y + 0.72), (1.96, y + 0.72), color=C["blue"], lw=0.7, ms=6,
          ls=(0, (2.5, 1.8)))
    arrow(ax, (2.11, y + 0.72), (2.40, y + 0.72), color=C["blue"], lw=0.7, ms=6,
          ls=(0, (2.5, 1.8)))
    ax.text(1.90, y + 0.24, tr("客体可达性", "object reachability"), ha="center",
            va="center", fontsize=5.9, color="#1d4e79")
    ax.text(1.55, y + 0.90, tr("平稳分布 $p$", "stationary $p$"), ha="center",
            va="center", fontsize=5.9, color=MUTED)

    # 行 B：状态迁移示意图
    y = 1.64
    chips = [(tr("常规", "plain"), "read"), (tr("常规", "plain"), "write"),
             (tr("高敏", "high"), "export"), (tr("高敏", "high"), "export"),
             (tr("常规", "plain"), "read")]
    for i, (lbl, op) in enumerate(chips):
        cx = 0.26 + i * 0.50
        hi = (lbl == tr("高敏", "high"))
        rbox(ax, cx, y + 0.42, 0.42, 0.34, "", fc="#f3e9fb" if hi else "white",
             ec=C["purple"], lw=0.8, fs=5.6, radius=0.04)
        ax.text(cx + 0.21, y + 0.62, lbl, ha="center", va="center", fontsize=5.4,
                color=TEXT, zorder=5)
        ax.text(cx + 0.21, y + 0.51, op, ha="center", va="center", fontsize=5.0,
                color=MUTED, zorder=5)
        if i < 4:
            arrow(ax, (cx + 0.42, y + 0.59), (cx + 0.50, y + 0.59),
                  color=C["purple"], lw=0.7, ms=6)
    arrow(ax, (0.26, y + 0.30), (2.42, y + 0.30), color=C["purple"],
          style="<|-|>", lw=0.65)
    ax.text(1.34, y + 0.19, tr("滑动窗口 $W = 5$ 次迁移", "sliding window of $W = 5$ transitions"),
            ha="center", va="center", fontsize=5.9, color="#5b3a8e")

    # 行 C：请求级残差示意图
    y = 0.56
    rbox(ax, 0.26, y + 0.42, 0.72, 0.34, tr("本次请求", "request"), fc="white",
         ec=C["teal"], lw=0.8, fs=5.8)
    bars = [0.55, 0.18, 0.62]
    for i, v in enumerate(bars):
        bx = 1.16 + i * 0.44
        ax.add_patch(Rectangle((bx, y + 0.42), 0.26, 0.34, fc="#e6f4f4",
                               ec=C["teal"], lw=0.6, zorder=4))
        ax.add_patch(Rectangle((bx, y + 0.42), 0.26, 0.34 * v, fc=C["teal"],
                               ec="none", zorder=5, alpha=0.65))
        ax.text(bx + 0.13, y + 0.34, "%.2f" % v, ha="center", va="center",
                fontsize=5.0, color=MUTED, zorder=6)
    ax.text(1.72, y + 0.80, tr("属性残差", "attribute residuals"), ha="center",
            va="center", fontsize=5.9, color="#1f5f5f")

    # ---- 底部：合成与分级 ----
    rbox(ax, 0.18, 0.06, 5.54, 0.40, "", fc="#f7f7fb", ec=C["gray"], lw=0.7)
    ax.text(0.30, 0.26,
            "$\\rho = 1-(1-$" + RS + "$)(1-$" + RT + "$)(1-$" + RB + "$)$",
            ha="left", va="center", fontsize=7.0, color=TEXT, zorder=4)
    gr = [(tr("低 [0,0.25)", "low\n[0,0.25)"), "#e8f2e8"),
          (tr("中 [0.25,0.5)", "medium\n[0.25,0.5)"), "#f3f0dd"),
          (tr("高 [0.5,0.8)", "high\n[0.5,0.8)"), "#f7e3dc"),
          (tr("极高 [0.8,1]", "very high\n[0.8,1]"), "#efd2cc")]
    for i, (t, fc) in enumerate(gr):
        rbox(ax, 2.72 + i * 0.74, 0.10, 0.70, 0.32, t, fc=fc, ec=C["gray"],
             lw=0.5, fs=5.6, radius=0.04, linespacing=1.15)
    ax.text(2.66, 0.26, tr("分级：", "grades:"), ha="right", va="center",
            fontsize=6.0, color=MUTED, zorder=4)



def fig03():
    _standalone(body03, 3, "risk_scope",
                tr("三个风险分量的判别对象与作用域划分",
            "Discriminative objects and disjoint scopes of the three risk components"),
                "Discriminative objects and disjoint scopes of the three risk components", 5.9, 3.80)



# ==================================================================== 图 4

def body04():
    """锚定间隔、粒度与暴露窗口的时间关系。"""
    W, H = 5.9, 2.75
    fig, ax = _diag(W, H)

    x0, x1 = 0.86, 5.62
    span = 6
    up = (x1 - x0) / float(span)

    def X(t):
        return x0 + t * up

    # 暴露窗口（保守计量：整窗口）
    ax.add_patch(Rectangle((X(1.0), 0.86), X(2.0) - X(1.0), 1.72,
                           fc="#fdf3e3", ec="none", zorder=0))
    ax.text(X(1.5), 2.48, tr("暴露窗口（按整个 $\\Delta t$ 计，保守上界）",
                             "exposure window (a full $\\Delta t$ is charged, conservative bound)"),
            ha="center", va="center", fontsize=6.0, color="#8a5a00", zorder=5)

    for t in range(span + 1):
        line(ax, (X(t), 0.90), (X(t), 2.30), color=FAINT, lw=0.6, ls=(0, (2.5, 2.0)))
        ax.text(X(t), 0.80, tr(("%d$\\Delta t$" % t) if t else "0",
                               ("%d$\\Delta t$" % t) if t else "0"),
                ha="center", va="top", fontsize=6.0, color=MUTED)

    tick_marks(ax, X(0.0), 2.16, X(6.0) - X(0.0), 0.16, 54, color=C["teal"])
    ax.text(0.10, 2.24, tr("记录流", "record\nstream"), ha="left", va="center",
            fontsize=6.3, color=TEXT, linespacing=1.3)

    rows = [(1.94, tr("$g=2$ 分层", "$g=2$ layered"),
             [tr("业务域→批次", "domain to batch"), tr("批次→根", "batch to root")]),
            (1.55, tr("$g=1$ 每批", "$g=1$ per batch"), [tr("每批一个叶", "one leaf per batch")]),
            (1.16, tr("$g=0$ 每记录", "$g=0$ per record"), [tr("每条记录一个叶", "one leaf per record")])]
    for y, lbl, notes in rows:
        ax.text(0.10, y, lbl, ha="left", va="center", fontsize=6.3, color=TEXT)
        if len(notes) == 2:
            for k in range(span // 3):
                xa, xb = X(k * 3), X(k * 3 + 3)
                ax.text((xa + xb) / 2, y + 0.16, notes[k], ha="center", va="center",
                        fontsize=5.6, color=MUTED)
                line(ax, (xa + 0.03, y + 0.09), (xb - 0.03, y + 0.09),
                     color=C["gray"], lw=0.7)
                line(ax, (xa + 0.03, y + 0.09), (xa + 0.03, y - 0.03), color=C["gray"], lw=0.7)
                line(ax, (xb - 0.03, y + 0.09), (xb - 0.03, y - 0.03), color=C["gray"], lw=0.7)
        elif tr("每批", "per batch") in lbl:
            for k in range(span):
                xa, xb = X(k), X(k + 1)
                line(ax, (xa + 0.04, y + 0.09), (xb - 0.04, y + 0.09),
                     color=C["gray"], lw=0.7)
                line(ax, (xa + 0.04, y + 0.09), (xa + 0.04, y - 0.03), color=C["gray"], lw=0.7)
                line(ax, (xb - 0.04, y + 0.09), (xb - 0.04, y - 0.03), color=C["gray"], lw=0.7)
        else:
            for k in range(span * 6):
                xx = X(k / 6.0)
                line(ax, (xx, y - 0.03), (xx, y + 0.09), color=C["gray"], lw=0.6)

    # 事件与锚定
    ax.add_patch(plt.Circle((X(1.35), 2.24), 0.045, fc=C["red"], ec="none", zorder=6))
    arrow(ax, (X(2.0), 1.02), (X(2.0), 2.34), color=C["amber"], lw=1.2, ms=9)
    # g=2 那一层的括号线在 y=2.03 横贯全图，标注放在该高度必被穿过（实测
    # 中文版被压住 0.39 in）。改放到 g=1 括号（1.64）与 g=2 括号（2.03）之间的
    # 空档，并折成两行，使宽度不超过相邻两条竖虚线之间 0.79 in 的间距。
    ax.text(X(2.0) + 0.06, 1.78,
            tr("锚定上链\n（代价 $c_g$）", "anchor on chain\n(cost $c_g$)"),
            ha="left", va="center", fontsize=6.0, color="#8a5a00", zorder=6,
            linespacing=1.25)
    arrow(ax, (X(1.35), 2.36), (X(2.0) - 0.04, 2.36), color=C["red"],
          style="<|-|>", lw=0.8, ms=7)
    ax.text((X(1.35) + X(2.0)) / 2, 2.62,
            tr("滞后 $\\leq \\Delta t$", "lag $\\leq \\Delta t$"),
            ha="center", va="center", fontsize=6.2, color=C["red"], zorder=6)

    ax.text(0.10, 0.48,
            wrap_to(ax, tr("$D_{det}^{max} = \\Delta t + t_c + t_v$：一条记录从产生到可被链上证据"
                           "确认的最长间隔。同一记录无论在窗口内何处产生，暴露都按整个 "
                           "$\\Delta t$ 计量。",
                           "$D_{det}^{max} = \\Delta t + t_c + t_v$: the longest interval from a "
                           "record's birth to its being confirmable by on-chain evidence. "
                           "Whatever its position in the window, a record is charged a full "
                           "$\\Delta t$."), 5.68, 6.2),
            ha="left", va="center", fontsize=6.2, color=TEXT)
    ax.text(0.10, 0.20,
            wrap_to(ax, tr("粒度 $g$ 决定一棵批次树内的分层深度：$g=0$ 时每条记录一个叶，$g=1$ 时"
                           "每批一个叶，$g=2$ 时按业务域先聚一层再入根。",
                           "Granularity $g$ sets the layering inside a batch tree: $g=0$ gives "
                           "one leaf per record, $g=1$ one leaf per batch, and $g=2$ an extra "
                           "level per business domain."), 5.68, 6.2),
            ha="left", va="center", fontsize=6.2, color=MUTED)



def fig04():
    _standalone(body04, 4, "anchoring_timeline",
                tr("锚定间隔、锚定粒度与暴露窗口的时间关系",
            "Time relation among anchoring interval, granularity and exposure window"),
                "Time relation among anchoring interval, granularity and exposure window", 5.9, 2.75)



# ==================================================================== 图 5

def body05():
    """决策路径时延与判定容量。"""
    W, H = 5.9, 4.05
    rows = load("E1_决策时延与吞吐.csv")
    a_rows = [r for r in rows if str(r["模式"]).startswith("异步")]
    s_rows = [r for r in rows if str(r["模式"]).startswith("同步")]
    lam = [r["lambda"] for r in a_rows]

    fig, gs = _mk(W, H, 2, 2, left=0.088, right=0.945, top=0.935, bottom=0.10, hspace=0.62, wspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)
    ax_c = fig.add_subplot(gs[1, :]); style_axes(ax_c)

    # (a) 时延分位。最大值在 λ=3200 时达 146.7 ms，一并画出会把 1 μs 量级的
    # 四条分位线压成一条，故只画分位，最大值另以文字交代。
    qs = [("P50_us", tr("P50", "P50"), C["blue"], "o"),
          ("P90_us", tr("P90", "P90"), C["teal"], "d"),
          ("P99_us", tr("P99", "P99"), C["red"], "s"),
          ("P99.9_us", tr("P99.9", "P99.9"), C["green"], "^")]
    for key, lbl, cc, mk in qs:
        ax_a.plot(lam, [r[key] for r in a_rows], marker=mk, ms=3.2, lw=0.9,
                  color=cc, label=lbl)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    # 注意：2026-10-18：主表改五次运行中位数后 P99.9 降到 80 至 100 μs，图例
    # （四行，落在左上）会与 P99.9 线相撞，故把纵轴上界抬到 1200 μs 留出空带。
    ax_a.set_ylim(0.5, 1200)
    ax_a.set_xlabel(tr("请求率 $\\lambda$ (req/s)", "request rate $\\lambda$ (req/s)"))
    ax_a.set_ylabel(tr("判定时延 (μs)", "decision latency (\u03bcs)"))
    legend(ax_a, ncol=2, loc="upper left")
    panel(ax_a, pnext())
    # 两条分位线在 1 μs 量级，注释只能落在 P99（最高 1.8）与 P99.9（最低 14.2）
    # 之间的空带里；原先放在 y=40 会被 P99.9 线横穿。最大值 146.7 ms 超出纵轴
    # （上限 300 μs），画不出来，改为写进图题。
    p50_rng = (min(r["P50_us"] for r in a_rows), max(r["P50_us"] for r in a_rows))
    p99_rng = (min(r["P99_us"] for r in a_rows), max(r["P99_us"] for r in a_rows))
    ax_a.annotate(tr("P50 %.1f 至 %.1f μs、P99 %.1f 至 %.1f μs"
                     % (p50_rng[0], p50_rng[1], p99_rng[0], p99_rng[1]),
                     "P50 %.1f to %.1f \u03bcs, P99 %.1f to %.1f \u03bcs"
                     % (p50_rng[0], p50_rng[1], p99_rng[0], p99_rng[1])),
                  xy=(1600.0, 1.2), xytext=(60.0, 12.0),
                  fontsize=6.0, color=MUTED, va="bottom",
                  arrowprops=dict(arrowstyle="->", lw=0.6, color=FAINT))

    # (b) 判定容量与单核利用率。同步对照的容量只有 1.97 req/s，画成横线会把纵轴
    # 拉到六个数量级、把本文曲线压到边框上，故只用文字标注该对照值。
    cap_s = s_rows[0]["判定容量_req_s"]
    ax_b.plot(lam, [r["判定容量_req_s"] for r in a_rows], marker="o", ms=3.4,
              lw=1.1, color=C["blue"])
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    # 注意：2026-10-18：最终实现的判定容量降到 3.78e5 至 6.08e5 req/s，原下界 4e5
    # 会把最低那一档压出画面，且注释放到 0.70 倍处会掉到轴外、与 x 轴标题重叠。
    ax_b.set_ylim(2.5e5, 5e6)
    ax_b.set_xlabel(tr("请求率 $\\lambda$ (req/s)", "request rate $\\lambda$ (req/s)"))
    ax_b.set_ylabel(tr("判定容量 (req/s)", "decision capacity (req/s)"))
    # 标注放在曲线上方：放到下方会掉进 x 轴刻度带（与 10^3 刻度叠字）。
    ax_b.text(lam[-1], a_rows[-1]["判定容量_req_s"] * 1.18,
              tr("判定容量（本文）", "capacity (ours)"), ha="right", va="bottom",
              fontsize=6.2, color=C["blue"])
    ax_b.text(lam[0] * 0.93, 2.8e6,
              tr("同步锚定（对照）仅 %.2f req/s" % cap_s,
                 "synchronous (control) only %.2f req/s" % cap_s),
              ha="left", va="top", fontsize=6.0, color=C["red"])
    ax2 = ax_b.twinx()
    ax2.plot(lam, [r["单核利用率"] for r in a_rows], marker="s", ms=3.0, lw=1.0,
             color=C["green"])
    ax2.set_yscale("log")
    # 不设轴标题：绿色序列已在图内用同名字样标注，版心也放不下第二个标题
    ax2.tick_params(direction="in", length=2.4, width=0.7, colors=MUTED,
                    labelsize=6.8)
    ax2.spines["top"].set_visible(False)
    ax2.text(lam[-1], a_rows[-1]["单核利用率"] * 1.25,
             tr("单核利用率", "utilisation"), ha="right", va="bottom", fontsize=6.0,
             color=C["green"])
    panel(ax_b, pnext())

    # (c) 同步与异步 P99 对比
    ypos = np.arange(len(lam))
    hp = 0.34
    ax_c.barh(ypos + hp / 1.7, [r["P99_us"] for r in a_rows], height=hp,
              color=C["blue"], label=tr("异步锚定（本文）", "asynchronous (ours)"))
    ax_c.barh(ypos - hp / 1.7, [r["P99_us"] for r in s_rows], height=hp,
              color=C["red"], label=tr("同步锚定（对照）", "synchronous (control)"))
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.7, 4e6)
    for i, (ra, rs) in enumerate(zip(a_rows, s_rows)):
        ax_c.text(ra["P99_us"] * 1.35, i + hp / 1.7, "%.1f μs" % ra["P99_us"],
                  va="center", fontsize=5.9, color=TEXT)
        ax_c.text(rs["P99_us"] * 1.35, i - hp / 1.7, "%.1f ms" % (rs["P99_us"] / 1000.0),
                  va="center", fontsize=5.9, color=TEXT)
    # 两条序列量的不是同一件事：异步侧是判定本身的求值时间（μs），同步侧是请求等到
    # 上链确认的阻塞时间（ms）。此前用一条双箭头标注「相差约 5.6 个数量级」，等于把
    # 两个测量对象当成同一指标求比值，正文已撤回该比较（第三轮审稿 R1），图内同步撤除，
    # 改为在轴上分段并各自标注对象名。
    ax_c.axvline(1e3, lw=0.8, ls=(0, (3, 2.2)), color=FAINT, zorder=2)
    ax_c.text(1.6e0, 3.52, tr("判定求值（μs 量级）", "evaluation (μs scale)"),
              ha="left", va="bottom", fontsize=6.0, color="#1d4e79")
    ax_c.text(3.0e3, 3.52, tr("等待链上确认（ms 量级）", "wait for confirmation (ms scale)"),
              ha="left", va="bottom", fontsize=6.0, color="#8f3a2e")
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels(["$\\lambda=%d$" % v for v in lam])
    ax_c.set_ylim(-0.7, len(lam) - 0.05)
    ax_c.set_xlabel(tr("P99 时延（对数刻度；两侧测量对象不同，不作倍率比较）",
                       "P99 latency (log scale; the two sides measure different objects, "
                       "so no ratio is formed)"))
    # 图例放到面板上方：柱值标签与图例在面板内必然相撞
    legend(ax_c, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.015))
    panel(ax_c, pnext())



def fig05():
    _standalone(body05, 5, "latency",
                tr("决策路径的时延分位与判定容量（最大时延在 $\\lambda=3200$ 时达 146.7 ms，"
            "超出 (a) 的纵轴范围，未画出）",
            "Latency quantiles and decision capacity of the decision path (the maximum "
            "reaches 146.7 ms at $\\lambda=3200$, beyond the axis range of (a) and therefore "
            "not drawn)"),
                "Latency quantiles and decision capacity of the decision path (the maximum "
         "reaches 146.7 ms at $\\lambda=3200$, beyond the axis range of (a) and therefore "
         "not drawn)", 5.9, 4.05)



# ==================================================================== 图 6

def body06():
    """锚定建根与策略版本校验的开销。"""
    W, H = 5.9, 2.85
    ank = load("E1_锚定批次成本.csv")
    ver = load("E3_一致性校验开销.csv")

    fig, gs = _mk(W, H, 1, 2, left=0.098, right=0.965, top=0.905, bottom=0.165, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)

    leaves = [r["每批叶数"] for r in ank]
    cost = [r["建根耗时_ms"] for r in ank]
    ax_a.plot(leaves, cost, marker="o", ms=3.6, lw=1.1, color=C["blue"])
    ax_a.plot([leaves[0], leaves[-1]], [cost[0], cost[-1] * 1.0], lw=0.8,
              ls=(0, (4, 2)), color=FAINT)
    for r in ank:
        # 标注一律放在点的左上方：曲线在对数坐标下斜率≈1，标注向右延伸必被
        # 曲线追上（实测四个标签的下缘都被压住），向左延伸则始终在曲线下方。
        # 最左点靠 xlim 的左留白腾地方，否则标签会顶到纵轴刻度上。
        ax_a.annotate("$\\Delta t=%d$" % r["dt"], (r["每批叶数"], r["建根耗时_ms"]),
                      textcoords="offset points", xytext=(-4, 3),
                      ha="right", va="bottom", fontsize=5.6, color=MUTED)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(left=leaves[0] / 3.0)
    ax_a.set_xlabel(tr("每批叶数", "leaves per batch"))
    ax_a.set_ylabel(tr("Merkle 建根耗时 (ms)", "Merkle root time (ms)"))
    # 箭头指向最后两点在对数坐标下的几何中点：每个数据点上方都有 Δt 标注，
    # 指到点上会让箭头尖压住标注。中点由数据算得，不写死坐标。
    x_mid = (leaves[-2] * leaves[-1]) ** 0.5
    y_mid = (cost[-2] * cost[-1]) ** 0.5
    ax_a.annotate(tr("斜率≈1：与叶数同阶", "slope ~1: linear in leaves"),
                  xy=(x_mid, y_mid), xytext=(leaves[0], cost[-1] * 0.62),
                  fontsize=6.2, color=MUTED, ha="left",
                  arrowprops=dict(arrowstyle="->", lw=0.6, color=MUTED))
    panel(ax_a, pnext())

    n = [r["策略条数"] for r in ver]
    t = [r["校验耗时_ms"] for r in ver]
    ax_b.plot(n, t, marker="o", ms=3.6, lw=1.1, color=C["red"],
              label=tr("逐条 Merkle 路径校验", "per-entry Merkle path check"))
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel(tr("策略条数", "policy entries"))
    ax_b.set_ylabel(tr("校验耗时 (ms)", "verification time (ms)"))
    axb2 = ax_b.twinx()
    axb2.plot(n, [r["平均证明节点数"] for r in ver], marker="s", ms=3.2, lw=1.0,
              color=C["blue"], ls="-", label=tr("平均证明节点数", "mean proof nodes"))
    axb2.plot(n, [r["理论log2(n)"] for r in ver], lw=0.9, color=C["gray"],
              ls=(0, (4, 2)), label=tr("理论 $\\log_2 N_P$", "theoretical $\\log_2 N_P$"))
    # 不设轴标题：两条序列已在图例中给出
    axb2.tick_params(direction="in", length=2.4, width=0.7, colors=MUTED, labelsize=6.8)
    axb2.spines["top"].set_visible(False)
    h1, l1 = ax_b.get_legend_handles_labels()
    h2, l2 = axb2.get_legend_handles_labels()
    ax_b.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=6.1,
                handlelength=1.6, labelspacing=0.3)
    panel(ax_b, pnext())



def fig06():
    _standalone(body06, 6, "anchor_verify_cost",
                tr("锚定建根与策略版本校验的开销随规模的变化",
            "Anchoring and policy-version verification cost versus scale"),
                "Anchoring and policy-version verification cost versus scale", 5.9, 2.85)



# ==================================================================== 图 7

def body07():
    """总成本率随锚定间隔的变化与成本分解。"""
    W, H = 5.9, 4.10
    abl = sel(load("E2_锚定粒度消融.csv"), 对照="基准")

    fig, gs = _mk(W, H, 2, 2, left=0.088, right=0.975, top=0.935, bottom=0.105, hspace=0.60, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)
    ax_c = fig.add_subplot(gs[1, :]); style_axes(ax_c)

    # (a) 不同请求率
    for i, lam_v in enumerate([50.0, 200.0, 800.0]):
        rs = sel(abl, {"lambda": lam_v, "g": 1.0})
        rs.sort(key=lambda r: r["dt"])
        ax_a.plot([r["dt"] for r in rs], [r["总成本率J"] for r in rs],
                  marker="", lw=1.2, color=SERIES[i],
                  label="$\\lambda=%d$" % lam_v)
        j = min(rs, key=lambda r: r["总成本率J"])
        ax_a.plot([j["dt"]], [j["总成本率J"]], marker="*", ms=7,
                  color=SERIES[i], zorder=5)
    ax_a.set_xscale("log")
    ax_a.set_xlabel(tr("锚定间隔 $\\Delta t$（相对单位）", "anchoring interval $\\Delta t$ (relative)"))
    ax_a.set_ylabel(tr("总成本率 $J$", "total cost rate $J$"))
    # 图例放上中：三条曲线都是 U 形，最小值在中段，顶部中段才是空的；
    # 放右上会被 λ=800 曲线右端穿过。
    legend(ax_a, ncol=1, loc="upper center")
    panel(ax_a, pnext())
    ax_a.text(0.03, 0.05, tr("注意：为数值最优", "* numerical optimum"),
              transform=ax_a.transAxes, ha="left", va="bottom", fontsize=6.0,
              color=MUTED)

    # (b) 成本分解
    rs = sel(abl, {"lambda": 200.0, "g": 1.0})
    rs.sort(key=lambda r: r["dt"])
    xs = [r["dt"] for r in rs]
    ax_b.plot(xs, [r["链上成本率"] for r in rs], lw=1.1, color=C["blue"],
              label=tr("链上成本率 $c_g/\\Delta t$", "on-chain rate $c_g/\\Delta t$"))
    ax_b.plot(xs, [r["暴露风险率_计量"] for r in rs], lw=1.1, color=C["red"],
              label=tr("暴露风险率（计量）", "exposure rate (metered)"))
    lo = [r["暴露风险率_加速抽样"] - r["抽样95%半宽"] for r in rs]
    hi = [r["暴露风险率_加速抽样"] + r["抽样95%半宽"] for r in rs]
    ax_b.fill_between(xs, lo, hi, color=C["red"], alpha=0.16, lw=0)
    ax_b.plot(xs, [r["暴露风险率_加速抽样"] for r in rs], lw=0.9, color=C["red"],
              ls=(0, (4, 2)), label=tr("暴露风险率（加速抽样）", "exposure rate (accelerated)"))
    ax_b.plot(xs, [r["总成本率J"] for r in rs], lw=1.3, color=C["purple"],
              label=tr("总成本率 $J$", "total cost rate $J$"))
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    # 固定参数并入轴标题：放在图内左下会被加速抽样曲线穿过
    ax_b.set_xlabel(tr("锚定间隔 $\\Delta t$（$\\lambda=200$、$g=1$）",
                       "anchoring interval $\\Delta t$ ($\\lambda=200$, $g=1$)"))
    ax_b.set_ylabel(tr("成本率（对数刻度）", "cost rate (log scale)"))
    # log 轴下「顶部乘 2.4」只换来 0.38 个数量级，四条曲线上端仍够到图例下沿、
    # 把「总成本率 J」压在中间，故留出 0.9 个数量级并给图例加白底框
    legend(ax_b, ncol=1, loc="upper center", frame=True)
    ax_b.set_ylim(top=ax_b.get_ylim()[1] * 10 ** 0.9)
    panel(ax_b, pnext())

    # (c) 不同粒度
    stars = []
    for i, gv in enumerate([0.0, 1.0, 2.0]):
        rs = sel(abl, {"lambda": 200.0, "g": gv})
        rs.sort(key=lambda r: r["dt"])
        ax_c.plot([r["dt"] for r in rs], [r["总成本率J"] for r in rs], lw=1.2,
                  color=SERIES[i], label="$g=%d$" % int(gv))
        j = min(rs, key=lambda r: r["总成本率J"])
        # 闭式解参考线只画注记带以上的一段：满高竖虚线会穿过底部注记
        # （英文文案宽 4.27 in，整条带都被压住，左右挪躲不开）。
        ax_c.plot([j["理论dt_star"]] * 2, [0.12, 1.0],
                  transform=ax_c.get_xaxis_transform(),
                  color=SERIES[i], lw=0.7, ls=(0, (3, 2)))
        stars.append(j["理论dt_star"])
    ax_c.set_xscale("log")
    ax_c.set_xlabel(tr("锚定间隔 $\\Delta t$（相对单位）", "anchoring interval $\\Delta t$ (relative)"))
    ax_c.set_ylabel(tr("总成本率 $J$", "total cost rate $J$"))
    legend(ax_c, ncol=3, loc="upper right")
    panel(ax_c, pnext())
    # 三条竖虚线彼此只隔约 0.16 in，而「Δt*=x.xx」标签宽 0.43 in，标在哪一侧
    # 都会被相邻那条线穿过（实测 g=0 与 g=1 两处）。改为把数值写进底部注记——
    # 曲线的最小值在面板高度的 14%~38%，其下方整条带都是空的。
    star_txt = " / ".join("%.2f" % v for v in stars)
    # 线性纵轴的自动下限只到 min-5%，曲线最低点因此落在面板高度约 5% 处，
    # 注记必须先把下界压下去才有位置（英文文案宽 4.27 in，整条带都会被压住）
    ax_c.set_ylim(bottom=-0.7)
    ax_c.text(0.01, 0.03,
              tr("$\\lambda=200$；虚线为闭式解 $\\Delta t^{*}$：$g=0/1/2$ 依次为 "
                 + star_txt + "（自左向右）",
                 "$\\lambda=200$; dashed lines are the closed form $\\Delta t^{*}$: "
                 + star_txt + " for $g=0/1/2$ (increasing left to right)"),
              transform=ax_c.transAxes, ha="left", va="bottom", fontsize=6.0, color=MUTED)



def fig07():
    _standalone(body07, 11, "tradeoff",
                tr("总成本率随锚定间隔的变化与成本分解",
            "Total cost rate versus anchoring interval and its decomposition"),
                "Total cost rate versus anchoring interval and its decomposition", 5.9, 4.10)



# ==================================================================== 图 8

def body08():
    """闭式锚定间隔与实测最优的吻合度及缩放律。"""
    W, H = 5.9, 4.20
    opt = load("E2_实测最优与闭式解对照.csv")
    sens = sel(load("E2_参数敏感度.csv"), p=1e-08)

    fig, gs = _mk(W, H, 2, 2, left=0.085, right=0.975, top=0.94, bottom=0.10, hspace=0.58, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)
    ax_c = fig.add_subplot(gs[1, 0]); style_axes(ax_c)
    ax_d = fig.add_subplot(gs[1, 1]); style_axes(ax_d)

    # (a) 实测 vs 闭式
    lo, hi = 3.5, 24.0
    ax_a.plot([lo, hi], [lo, hi], lw=0.8, ls=(0, (4, 2)), color=C["gray"],
              label=tr("$y=x$", "$y=x$"))
    for gv, mk in zip([0.0, 1.0, 2.0], ["o", "s", "^"]):
        rs = sel(opt, g=gv)
        ax_a.scatter([r["理论dt_star"] for r in rs], [r["实测最优dt"] for r in rs],
                     marker=mk, s=17, color=SERIES[int(gv)], zorder=5,
                     label="$g=%d$" % int(gv), edgecolor="white", linewidth=0.4)
    ax_a.set_xlim(lo, hi); ax_a.set_ylim(lo, hi)
    ax_a.set_xlabel(tr("闭式解 $\\Delta t^{*}$", "closed-form $\\Delta t^{*}$"))
    ax_a.set_ylabel(tr("仿真实测最优 $\\Delta t$", "measured optimum $\\Delta t$"))
    legend(ax_a, ncol=2, loc="upper left")
    panel(ax_a, pnext())
    errs = [abs(r["相对误差"]) * 100 for r in opt]
    ax_a.text(0.97, 0.06, tr("中位误差 %.2f%%，最大 %.2f%%" % (float(np.median(errs)), max(errs)),
                             "median %.2f%%, max %.2f%%" % (float(np.median(errs)), max(errs))),
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.1,
              color=MUTED)

    # (b) 相对误差
    labels = ["$\\lambda=%d$, $g=%d$" % (r["lambda"], r["g"]) for r in opt]
    xs = np.arange(len(opt))
    ax_b.bar(xs, [abs(r["相对误差"]) * 100 for r in opt],
             color=[SERIES[int(r["g"])] for r in opt], width=0.66)
    ax_b.axhline(float(np.median(errs)), lw=0.9, ls=(0, (4, 2)), color=C["gray"])
    ax_b.text(len(opt) - 0.5, float(np.median(errs)) * 1.25,
              tr("中位 %.2f%%" % float(np.median(errs)), "median %.2f%%" % float(np.median(errs))),
              ha="right", fontsize=6.0, color=MUTED)
    ax_b.set_xticks(xs)
    # 注意：2026-10-18 改（第五轮审稿 R1.2）：原写法只删掉 "\lambda=" 与 "$g="，
    # 尾部的 "$" 留在标签里（如 "50/0$"），图上会看到美元符。
    ax_b.set_xticklabels([l.replace("$\\lambda=", "").replace("$, $g=", "/").rstrip("$")
                          for l in labels], fontsize=5.6, rotation=90)
    ax_b.set_xlabel(tr("参数组合（$\\lambda$ / $g$）", "parameter setting ($\\lambda$ / $g$)"))
    ax_b.set_ylabel(tr("相对误差 (%)", "relative error (%)"))
    panel(ax_b, pnext())

    # (c) 缩放律：Δt* 与 λ
    for i, cr in enumerate([90.0, 900.0, 9000.0]):
        rs = sel(sens, c_r=cr, g=1.0)
        rs.sort(key=lambda r: r["lambda"])
        ax_c.plot([r["lambda"] for r in rs], [r["dt_star"] for r in rs], marker="o",
                  ms=3.2, lw=1.0, color=SERIES[i],
                  label="$c_r=%g$" % cr)
    rs = sel(sens, c_r=900.0, g=1.0)
    rs.sort(key=lambda r: r["lambda"])
    xm = float(np.sqrt(rs[0]["lambda"] * rs[-1]["lambda"]))
    ym = float(np.sqrt(rs[0]["dt_star"] * rs[-1]["dt_star"]))
    xx = np.array([rs[0]["lambda"], rs[-1]["lambda"]])
    ax_c.plot(xx, ym * (xx / xm) ** -0.5, lw=0.9, ls=(0, (4, 2)), color=C["gray"],
              label=tr("斜率 $-1/2$", "slope $-1/2$"))
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xticks([50, 200, 800])
    ax_c.set_xticklabels(["50", "200", "800"])
    # 对数轴的次刻度带小数标签，横轴只有三个主刻度，次刻度标签会互相压住
    ax_c.xaxis.set_minor_locator(mticker.NullLocator())
    ax_c.yaxis.set_minor_locator(mticker.NullLocator())
    ax_c.set_xlabel(tr("请求率 $\\lambda$", "request rate $\\lambda$"))
    ax_c.set_ylabel(tr("$\\Delta t^{*}$（相对单位）", "$\\Delta t^{*}$ (relative)"))
    legend(ax_c, ncol=2, loc="upper right")
    panel(ax_c, pnext())

    # (d) 缩放律：Δt* 与 c_r
    for i, lam_v in enumerate([50.0, 200.0, 800.0]):
        rs = sel(sens, {"lambda": lam_v, "g": 1.0})
        rs.sort(key=lambda r: r["c_r"])
        ax_d.plot([r["c_r"] for r in rs], [r["dt_star"] for r in rs], marker="s",
                  ms=3.2, lw=1.0, color=SERIES[i], label="$\\lambda=%d$" % lam_v)
    rs = sel(sens, {"lambda": 200.0, "g": 1.0})
    rs.sort(key=lambda r: r["c_r"])
    xm = float(np.sqrt(rs[0]["c_r"] * rs[-1]["c_r"]))
    ym = float(np.sqrt(rs[0]["dt_star"] * rs[-1]["dt_star"]))
    xx = np.array([rs[0]["c_r"], rs[-1]["c_r"]])
    # 注意：2026-10-18 改（第五轮审稿 R1.2）：横轴是 c_r，而 dt* = sqrt(c_g/(c_r p lam))
    # 对 c_r 的弹性是 **-1/2**；+1/2 属于 c_g。原图画的是正幂，参考线向上，
    # 与实线下降的方向相反，等于把 c_r 扫描读成了 c_g 扫描。
    ax_d.plot(xx, ym * (xx / xm) ** -0.5, lw=0.9, ls=(0, (4, 2)), color=C["gray"],
              label=tr("斜率 $-1/2$", "slope $-1/2$"))
    ax_d.set_xscale("log"); ax_d.set_yscale("log")
    ax_d.set_xticks([90, 900, 9000])
    ax_d.set_xticklabels(["90", "900", "9000"])
    ax_d.set_xlabel(tr("风险成本系数 $c_r$", "risk cost coefficient $c_r$"))
    ax_d.set_ylabel(tr("$\\Delta t^{*}$（相对单位）", "$\\Delta t^{*}$ (relative)"))
    # 曲线随 c_r 增大而下降（左上到右下），图例避开曲线放右上
    legend(ax_d, ncol=1, loc="upper right", frame=True)
    panel(ax_d, pnext())



def fig08():
    _standalone(body08, 12, "closed_form",
                tr("闭式锚定间隔与仿真实测最优的吻合度及其缩放律",
            "Agreement between the closed-form and measured optimum and the scaling law"),
                "Agreement between the closed-form and measured optimum and the scaling law", 5.9, 4.20)



# ==================================================================== 图 9

def body09():
    """跨中心互认的成功条件与对照结构下的权衡曲线。"""
    W, H = 5.9, 2.90
    e4 = load("E4_跨中心互认.csv")
    ctrl = load("E2_对照实验.csv")
    abl = sel(load("E2_锚定粒度消融.csv"), 对照="基准")

    fig, gs = _mk(W, H, 1, 2, left=0.092, right=0.975, top=0.905, bottom=0.175, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)

    x = [r["失效中心数"] for r in e4]
    y = [r["互认成功率"] for r in e4]
    ax_a.axvspan(-0.35, 1.5, color="#eaf3ea", zorder=0)
    ax_a.axvspan(1.5, 3.45, color="#f9ece9", zorder=0)
    ax_a.plot(x, y, marker="o", ms=4.6, lw=1.2, color=C["blue"], ls="-")
    for xi, yi, r in zip(x, y, e4):
        # 放在点的右上方而非正上方：成功率在 x=1→2 段从 0.8 陡降到 0，
        # 正上方无论抬高多少都会被这段曲线切到，右侧则整段都是空的。
        ax_a.annotate(tr("份额 %d" % r["可用签名份额"], "shares %d" % r["可用签名份额"]),
                      (xi, yi), textcoords="offset points", xytext=(5, 4),
                      ha="left", va="bottom", fontsize=5.9, color=MUTED)
    ax_a.text(0.55, 0.52, tr("份额充足\n可完成门限背书", "enough shares\nthreshold met"),
              ha="center", va="center", fontsize=6.0, color="#2f6b3f")
    ax_a.text(2.5, 0.52, tr("份额不足\n无法产生有效背书", "shares insufficient\nno endorsement"),
              ha="center", va="center", fontsize=6.0, color="#8f3a2e")
    ax_a.set_ylim(-0.22, 1.25)
    ax_a.set_xticks(x)
    ax_a.set_xlabel(tr("失效中心数（共 3 个中心）", "failed centres (3 centres in total)"))
    ax_a.set_ylabel(tr("跨中心互认成功率", "mutual endorsement success rate"))
    panel(ax_a, pnext())
    # 门限说明不进图内：面板 (a) 的下缘已被 shares 标注与数据点占满，任何位置都会压字。
    # 「至多 $N_M-t$ 个成员不可用」改在图题里交代，图中只留绿/红两区的能力说明。

    base = sel(abl, {"lambda": 200.0, "g": 1.0})
    base.sort(key=lambda r: r["dt"])
    ax_b.plot([r["dt"] for r in base], [r["总成本率J"] for r in base], lw=1.4,
              color=C["gray"], label=tr("基准（三条假设成立）", "baseline (assumptions hold)"))
    names = []
    for r in ctrl:
        if r["对照"] not in names:
            names.append(r["对照"])
    for i, nm in enumerate(names):
        rs = sel(ctrl, {"对照": nm, "lambda": 200.0, "g": 1.0})
        rs.sort(key=lambda r: r["dt"])
        ax_b.plot([r["dt"] for r in rs], [r["总成本率J"] for r in rs], lw=1.1,
                  color=SERIES[i], label={"择时攻击": tr("择时攻击", "timing attack"),
                                          "c_r 分段": tr("$c_r$ 分段", "$c_r$ piecewise"),
                                          "分级锚定": tr("分级锚定", "graded anchoring"),
                                          "批次元数据成本": tr("批次元数据成本", "batch metadata cost")}[nm])
    ax_b.set_xscale("log")
    # 固定参数并入轴标题：放在图内左下会被批次元数据成本曲线穿过
    ax_b.set_xlabel(tr("锚定间隔 $\\Delta t$（$\\lambda=200$、$g=1$）",
                       "anchoring interval $\\Delta t$ ($\\lambda=200$, $g=1$)"))
    ax_b.set_ylabel(tr("总成本率 $J$", "total cost rate $J$"))
    legend(ax_b, ncol=1, loc="upper right")
    ax_b.set_ylim(top=ax_b.get_ylim()[1] * 1.8)
    panel(ax_b, pnext())



def fig09():
    _standalone(body09, 7, "interop",
                tr("跨中心互认的成功条件与对照结构下的权衡曲线",
            "Success condition of cross-centre endorsement and trade-off curves under controls"),
                "Success condition of cross-centre endorsement and trade-off curves under controls", 5.9, 2.90)



# ==================================================================== 图 10

def body10():
    """各检测通道的检出能力与误报率。"""
    W, H = 5.9, 3.55
    ch = load("E5_通道检出率.csv")
    atk = ["A1", "A2", "A3", "A4", "A7"]
    rows = [r for r in ch if r["TPR_A1"] is not None]
    # 文献 [11] 式单标量风险基线（由 lit11_baseline.py 独立产出，样本与 E5 逐条一致）：
    # 插到「合成 ρ」之后、「三通道合并」之前，与另一条单标量评分并排比较。
    _lit = [r for r in load("E11_工作点通道行.csv") if r["类型"] == "文献基线"]
    if _lit:
        _i = next((i for i, r in enumerate(rows) if r["通道"].startswith("指定确定性")), len(rows))
        rows = rows[:_i] + _lit + rows[_i:]
    name_map = {
        "第 1 步 角色基线硬边界": tr("① 角色基线硬边界", "1 role baseline hard bound"),
        "第 1 步 归属中心背书": tr("① 归属中心背书", "1 home-centre endorsement"),
        "第 2 步 上下文条件（θ=0.60）": tr("② 上下文条件", "2 contextual conditions"),
        "统计通道 ρ_struct（按角色标定归一化），θ=0.30": RS + " " + tr("统计通道", "statistical"),
        "统计通道 ρ_trans（按角色标定归一化），θ=0.30": RT + " " + tr("统计通道", "statistical"),
        "统计通道 合成 ρ（含 ρ_behav 占位），θ=0.45": tr("合成", "combined") + " " + RHO,
        "指定确定性与告警通道的并集": tr("指定通道并集", "specified-channel union"),
    }

    def _label(nm: str) -> str:
        # 注意：基线行的阈值随扫描网格变化（2026-10-18 由 0.30 改为其真实极大 0.12），
        # 不能再写死进 name_map 的键里——否则匹配不上、整行退回中文原串。
        if nm.startswith("文献[11]"):
            lab = tr("文献 [11] 式单标量", "single scalar [11]")
            if "θ=" in nm:
                th = nm.split("θ=")[1].rstrip("）")
                lab += tr("（θ=%s）", " ($\\theta$=%s)") % th
            return lab
        return name_map.get(nm, nm)

    labels = [_label(r["通道"]) for r in rows]
    M = np.array([[r["TPR_%s" % a] for a in atk] for r in rows], dtype=float)

    fig, gs = _mk(W, H, 1, 2, width_ratios=[2.05, 1.0], left=0.245, right=0.975, top=0.885, bottom=0.155, wspace=0.55)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f4f4f4")
    Mm = np.ma.masked_invalid(M)
    im = ax_a.imshow(Mm, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto",
                     origin="upper")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax_a.text(j, i, "—", ha="center", va="center", fontsize=6.0,
                          color=MUTED)
            else:
                ax_a.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=5.9,
                          color="white" if v > 0.55 else TEXT)
    ax_a.set_xticks(range(len(atk)))
    ax_a.set_xticklabels(atk, fontsize=6.6)
    ax_a.set_yticks(range(len(labels)))
    ax_a.set_yticklabels(labels, fontsize=6.1)
    ax_a.set_xlabel(tr("攻击剧本", "attack script"))
    ax_a.tick_params(length=0)
    for s in ax_a.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.032, pad=0.02)
    cb.ax.tick_params(labelsize=6.0, length=2.0, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.set_label(tr("检出率 TPR", "detection rate TPR"), fontsize=6.4)
    panel(ax_a, pnext())

    fpr = [r["误报率"] for r in rows]
    ypos = np.arange(len(rows))
    ax_b.barh(ypos, fpr, height=0.62, color=[C["blue"]] * (len(rows) - 1) + [C["red"]])
    for i, v in enumerate(fpr):
        ax_b.text(v + 0.006, i, "%.4f" % v if v < 0.01 else "%.3f" % v,
                  va="center", fontsize=5.7, color=TEXT)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 0.325)
    ax_b.set_xlabel(tr("误报率 FPR", "false-positive rate"))
    panel(ax_b, pnext())
    ax_b.text(0.98, 0.965, tr("单通道工作点上的误报", "FPR at each channel's working point"),
              transform=ax_b.transAxes, ha="right", va="top", fontsize=5.9,
              color=MUTED)



def fig10():
    _standalone(body10, 8, "channel_detection",
                tr("各检测通道对不同攻击剧本的检出能力与误报率",
            "Per-channel detection capability and false-positive rate across attack scripts"),
                "Per-channel detection capability and false-positive rate across attack scripts", 5.9, 3.55)



# ==================================================================== 图 11

def body11():
    """风险分量的判别力消融与阈值工作点。"""
    W, H = 5.9, 4.15
    abl = load("E5_风险分量消融.csv")
    sc = load("E5_阈值扫描.csv")

    fig, gs = _mk(W, H, 2, 2, left=0.085, right=0.975, top=0.935, bottom=0.10, hspace=0.62, wspace=0.30, height_ratios=[1.0, 1.12])
    ax_a = fig.add_subplot(gs[0, :]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[1, 0]); style_axes(ax_b)
    ax_c = fig.add_subplot(gs[1, 1]); style_axes(ax_c)

    # (a) 分量消融
    scripts = [r["剧本"] for r in abl]
    series = [("AUC_ρ_struct", RS, C["blue"]), ("AUC_ρ_trans", RT, C["purple"]),
              ("AUC_ρ_behav", RB, C["teal"]),
              ("AUC_二分量", tr("二分量", "two-comp."), C["amber"]),
              ("AUC_三分量", tr("三分量", "three-comp."), C["red"])]
    xs = np.arange(len(scripts))
    bw = 0.155
    for i, (key, lbl, cc) in enumerate(series):
        off = (i - 2) * bw
        ax_a.bar(xs + off, [r[key] for r in abl], width=bw * 0.92, color=cc,
                 label=lbl)
    ihot = scripts.index("A7")
    ax_a.axvspan(ihot - 0.46, ihot + 0.46, color="#fdf3e3", zorder=0)
    ax_a.axhline(0.5, lw=0.8, ls=(0, (4, 2)), color=C["gray"])
    ax_a.text(-0.44, 0.518, tr("随机判别", "chance"), fontsize=5.9, color=MUTED)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(scripts, fontsize=6.6)
    ax_a.set_ylim(0, 1.52)
    ax_a.set_ylabel(tr("AUC", "AUC"))
    ax_a.set_xlabel(tr("攻击剧本", "attack script"))
    legend(ax_a, ncol=3, loc="upper left")
    panel(ax_a, pnext())
    ax_a.annotate(tr("A7：客体分布与正常流一致，\n$\\rho_{struct}$ 失效，判别力归于 $\\rho_{trans}$",
                     "A7: object distribution matches normal traffic,\n"
                     "$\\rho_{struct}$ fails, power comes from $\\rho_{trans}$"),
                  xy=(ihot, 0.78), xytext=(2.05, 1.24), fontsize=5.9,
                  color="#8a5a00",
                  arrowprops=dict(arrowstyle="->", lw=0.6, color="#8a5a00"))

    # (b) 阈值扫描
    styles = {"ρ_struct": (C["blue"], "A3"), "ρ_trans": (C["purple"], "A7"),
              "合成ρ": (C["red"], "TPR_A1UA3")}
    keymap = {"ρ_struct": "TPR_A3", "ρ_trans": "TPR_A7", "合成ρ": "TPR_A1UA3"}
    lab = {"ρ_struct": tr("在 A3 上", "on A3"), "ρ_trans": tr("在 A7 上", "on A7"),
           "合成ρ": tr("在 A1∪A3 上", "on A1 u A3")}
    # 图例名：数据表的评分列里有一档叫「合成ρ」，直接拿来当 label 会在英文版
    # 里落成 CJK 字形（DejaVu Sans 缺字，渲染成方框），这里按语言取词。
    nm_show = {k: tr(k, "combined $\\rho$" if k == "合成ρ" else k) for k in styles}
    for nm, (cc, _) in styles.items():
        rs = [r for r in sc if r["评分"] == nm]
        rs.sort(key=lambda r: r["阈值θ"])
        ax_b.plot([r["阈值θ"] for r in rs], [r[keymap[nm]] for r in rs],
                  marker="o", ms=2.6, lw=1.1, color=cc,
                  label=nm_show[nm] + " " + lab[nm])
        ax_b.plot([r["阈值θ"] for r in rs], [r["FPR"] for r in rs], lw=0.8,
                  ls=(0, (3, 2)), color=cc, alpha=0.65)
    ax_b.set_xlabel(tr("阈值 $\\theta$", "threshold $\\theta$"))
    ax_b.set_ylabel(tr("检出率（实线）与误报率（虚线）", "TPR (solid) and FPR (dashed)"))
    ax_b.set_ylim(0, 1.12)
    legend(ax_b, ncol=1, loc="upper right")
    panel(ax_b, pnext())

    # (c) 检出—误报权衡
    for nm, (cc, _) in styles.items():
        rs = [r for r in sc if r["评分"] == nm]
        rs.sort(key=lambda r: r["阈值θ"])
        ax_c.plot([r["FPR"] for r in rs], [r[keymap[nm]] for r in rs], marker="o",
                  ms=3.0, lw=1.2, color=cc, label=nm_show[nm])
        # 注意：2026-10-18：网格已由 0.30 起扩到 0.05 起，rs[0] 不再是最初的端点。
        # 这里标的是**工作点** θ=0.30（正文实际取用的点），不是曲线端点。
        lo = next((r for r in rs if abs(r["阈值θ"] - 0.30) < 1e-9), rs[0])
        _bb = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.70)
        # 2026-10-18（第六轮 R6）：文字只写工作点，坐标靠箭头钉到散点上，
        # 三条线的偏移各不相同以免互压；箭头从文字端指向该 θ 的真实散点。
        _off = {"合成ρ": (-30, -18), "ρ_struct": (16, 14),
                "ρ_trans": (26, -20)}[nm]
        ax_c.annotate("$\\theta=%.2f$" % lo["阈值θ"], (lo["FPR"], lo[keymap[nm]]),
                      textcoords="offset points", xytext=_off, fontsize=5.7,
                      color=cc, bbox=_bb,
                      arrowprops=dict(arrowstyle="-|>", color=cc, lw=0.6,
                                      shrinkA=1.5, shrinkB=2.5,
                                      mutation_scale=4.5))
    ax_c.set_xlabel(tr("误报率 FPR", "false-positive rate"))
    ax_c.set_ylabel(tr("该分量目标检出率", "TPR on its target script"))
    ax_c.set_xlim(-0.01, 0.44)
    ax_c.set_ylim(0, 1.12)
    legend(ax_c, ncol=1, loc="lower right")
    panel(ax_c, pnext())



def fig11():
    _standalone(body11, 9, "ablation_threshold",
                tr("风险分量的判别力消融与阈值工作点的选取",
            "Discriminative-power ablation of the risk components and threshold selection"),
                "Discriminative-power ablation of the risk components and threshold selection", 5.9, 4.15)



# ==================================================================== 图 12

def body12():
    """处置分布、聚合方式对照与检出滞后。"""
    W, H = 5.9, 4.05
    disp3 = load("E5_处置分布.csv")
    disp2 = load("E5_处置分布_二分量对照.csv")
    aggr = load("E5_聚合方式对照.csv")
    lag = load("E5_检测滞后.csv")

    fig, gs = _mk(W, H, 2, 2, left=0.088, right=0.955, top=0.935, bottom=0.115, hspace=0.58, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[1, 0]); style_axes(ax_b)
    ax_c = fig.add_subplot(gs[1, 1]); style_axes(ax_c)

    # (a) 处置分布
    order = [r["标签"] for r in disp3]
    cols = [("放行", tr("放行", "permit"), "#3f8f5f"),
            ("收紧或降级", tr("收紧或降级", "refine/degrade"), "#d5a03a"),
            ("拒绝", tr("拒绝", "deny"), "#b4453a")]
    bw, gap = 0.32, 0.075
    for k, lbl in enumerate(order):
        r3 = [r for r in disp3 if r["标签"] == lbl][0]
        r2 = [r for r in disp2 if r["标签"] == lbl][0]
        for si, (r, hatch, base) in enumerate([(r2, "", -1), (r3, "///", 1)]):
            x = k + si * (bw + gap) - (bw + gap) / 2.0
            bottom = 0.0
            for key, _, cc in cols:
                v = r[key]
                ax_a.bar(x, v, width=bw, bottom=bottom, fc=cc, ec="white", lw=0.5,
                         hatch=hatch, zorder=3)
                if v > 0.075:
                    ax_a.text(x, bottom + v / 2, "%.2f" % v, ha="center", va="center",
                              fontsize=5.4, color="white", zorder=4)
                bottom += v
    # 注：原先在每个柱组下方另画一行「二分量 / 三分量」手写标签，位置与
    # matplotlib 的 x 刻度标签同处轴下一条带，实测两套文字互相压字（中英版
    # 均如此）。该信息已由 x 轴标题「左柱为二分量合成，右柱为三分量合成」与
    # 图例「三分量（斜纹）」承担，故删去这行重复标签，不再另画。
    ax_a.set_xticks(range(len(order)))
    # 刻度名来自数据表的「标签」列，其中「正常」一档是中文，直接画出来在
    # 英文版里会缺字形，故按语言取名。
    ax_a.set_xticklabels([tr(o, "normal" if o == "正常" else o) for o in order],
                         fontsize=6.6)
    ax_a.set_ylim(0, 1.14)
    ax_a.set_ylabel(tr("处置占比", "proportion of dispositions"))
    ax_a.set_xlabel(tr("剧本（左柱为二分量合成，右柱为三分量合成）",
                       "script (left bar: two components; right bar: three)"))
    hnd = [Rectangle((0, 0), 1, 1, fc=cc) for _, _, cc in cols]
    hnd += [Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="///", lw=0.5)]
    ax_a.legend(hnd, [lbl for _, lbl, _ in cols] + [tr("三分量（斜纹）", "three-component (hatched)")],
                ncol=4, loc="upper center", frameon=False, fontsize=6.0,
                handlelength=1.3, columnspacing=0.9, borderaxespad=0.15)
    panel(ax_a, pnext())

    # (b) 聚合方式
    agg_names = [r["聚合方式"] for r in aggr]
    segs = [("占比_低", tr("低", "low"), "#dfeae0"),
            ("占比_中", tr("中", "medium"), "#efeada"),
            ("占比_高", tr("高", "high"), "#f5ddd6"),
            ("占比_极高", tr("极高", "very high"), "#e6bdb4")]
    xs = np.arange(len(agg_names))
    bottoms = np.zeros(len(agg_names))
    for key, lbl, cc in segs:
        vals = np.array([r[key] for r in aggr])
        ax_b.bar(xs, vals, width=0.56, bottom=bottoms, fc=cc, ec=C["gray"], lw=0.5,
                 label=lbl)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.06:
                ax_b.text(xi, b + v / 2, "%.3f" % v, ha="center", va="center",
                          fontsize=5.6, color=TEXT)
        bottoms += vals
    ax_b.set_xticks(xs)
    # 刻度名取自数据表的「聚合方式」列，本身是中文。逐词 replace 只换得掉
    # 后缀，「二分量 / 三分量」前缀会原样落到英文版里，渲染成 CJK 方框；
    # 这里整串按语言取词。
    agg_lab = {"二分量概率和": tr("二分量概率和", "two-comp. noisy-OR"),
               "三分量概率和": tr("三分量概率和", "three-comp. noisy-OR"),
               "三分量取最大": tr("三分量取最大", "max of three")}
    ax_b.set_xticklabels([agg_lab.get(n, n) for n in agg_names], fontsize=5.8)
    ax_b.set_ylim(0, 1.16)
    ax_b.set_ylabel(tr("正常请求的等级占比", "grade share of normal traffic"))
    legend(ax_b, ncol=4, loc="upper center")
    panel(ax_b, pnext())
    fx = {"二分量概率和": 0.5015, "三分量概率和": 0.6137, "三分量取最大": 0.5931}
    txt = tr("A7 的 AUC：%.3f → %.3f → %.3f",
             "AUC on A7: %.3f -> %.3f -> %.3f") % tuple(fx[n] for n in agg_names)
    ax_b.text(0.5, -0.22, txt, transform=ax_b.transAxes, ha="center", va="top",
              fontsize=5.9, color=MUTED)

    # (c) 检测滞后
    seg = [r["进度区间"] for r in lag]
    ypos = np.arange(len(lag))
    ax_c.bar(ypos - 0.19, [r["高敏操作占比"] for r in lag], width=0.34,
             color="#cbb7e4", label=tr("高敏操作占比", "high-sensitivity share"))
    ax_c.bar(ypos + 0.19, [r["ρ_trans均值"] for r in lag], width=0.34,
             color=C["purple"], label=RT + tr(" 均值", " mean"))
    axc2 = ax_c.twinx()
    axc2.plot(ypos, [r["检出率"] for r in lag], marker="s", ms=3.4, lw=1.2,
              color=C["red"], label=tr("检出率", "detection rate"))
    # 不设轴标题：该序列已在面板 (c) 图例中给出
    axc2.set_ylim(0, 0.86)
    axc2.tick_params(direction="in", length=2.4, width=0.7, colors=MUTED, labelsize=6.8)
    axc2.spines["top"].set_visible(False)
    ax_c.set_xticks(ypos)
    ax_c.set_xticklabels([s.strip('"') for s in seg], fontsize=5.5, rotation=18)
    ax_c.set_xlabel(tr("A7 暴露窗口内的进度区间", "progress bucket within the A7 window"))
    ax_c.set_ylabel(tr("分量均值与占比", "mean value and share"))
    ax_c.set_ylim(0, 0.80)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = axc2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=5.9,
                handlelength=1.5)
    panel(ax_c, pnext())
    axc2.text(0.97, 0.06, tr("均值仅作趋势判读", "read as a trend only"),
              transform=axc2.transAxes, ha="right", va="bottom", fontsize=5.8,
              color=MUTED)



def fig12():
    _standalone(body12, 10, "disposition_aggregation_lag",
                tr("不同剧本的处置分布、聚合方式对照与检出滞后",
            "Disposition distribution, aggregation comparison and detection lag"),
                "Disposition distribution, aggregation comparison and detection lag", 5.9, 4.05)



# ==================================================================== 入口

# ==================================================================== 图 13

def body13():
    """链上路线的成本对照与端到端锚定的确认时延（真实链实测）。"""
    W, H = 5.9, 2.86
    loadc = load("E8_链上负载率对照.csv")
    tp = load("E8_吞吐上界.csv")
    loop = load("E9_端到端锚定.csv")

    fig, gs = _mk(W, H, 1, 2, left=0.088, right=0.978, top=0.885, bottom=0.155, wspace=0.265)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    style_axes(ax_a)
    style_axes(ax_b)

    # ---- (a) 链上成本率随请求率的变化 ----
    lam = col(loadc, "请求率_req_s")
    ax_a.plot(lam, col(loadc, "B3_gas_per_s"), marker="o", ms=3.2, lw=1.5,
              color=C["red"], zorder=4, label=tr("B3 逐条记录上链", "B3 per-record"))
    ax_a.plot(lam, col(loadc, "本文Δt=1s_gas_per_s"), marker="s", ms=3.0, lw=1.4,
              color=C["blue"], zorder=4, label=tr("本文 $\\Delta t$ = 1 s", "ours, $\\Delta t$ = 1 s"))
    ax_a.plot(lam, col(loadc, "本文Δt=60s_gas_per_s"), marker="^", ms=3.0, lw=1.4,
              color=C["teal"], zorder=4, label=tr("本文 $\\Delta t$ = 60 s", "ours, $\\Delta t$ = 60 s"))
    # B3 路线的请求率上界即链的事务吞吐上界（1 笔交易只能承载 1 条记录）
    b3tps = None
    for r in tp:
        if str(r["路线"]).startswith("B3"):
            b3tps = float(r["吞吐_TPS"])
    if b3tps:
        ax_a.axvline(b3tps, color=C["red"], lw=0.9, ls=(0, (3, 2.4)), zorder=2)
        ax_a.text(b3tps * 0.9, 2.5e8, tr("链吞吐上界\n%d TPS" % int(b3tps),
                                          "chain throughput\nceiling %d TPS" % int(b3tps)),
                  ha="right", va="top", fontsize=5.3, color=C["red"], linespacing=1.25)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(tr("请求到达率 $\\lambda$ (req/s)", "request rate $\\lambda$ (req/s)"))
    ax_a.set_ylabel(tr("链上成本率 (gas/s)", "on-chain cost rate (gas/s)"))
    ax_a.set_ylim(1e3, 1e9)
    ax_a.legend(loc="upper left", frameon=False, fontsize=5.6, handlelength=1.5,
                borderaxespad=0.2, labelspacing=0.32)
    panel(ax_a, pnext())

    # ---- (b) 端到端锚定的确认时延随锚定间隔的变化 ----
    dts = col(loop, "Δt_ms")
    ax_b.plot(dts, col(loop, "确认时延P50_ms"), marker="o", ms=3.2, lw=1.5,
              color=C["blue"], zorder=4, label=tr("P50", "P50"))
    ax_b.plot(dts, col(loop, "确认时延P95_ms"), marker="s", ms=3.0, lw=1.4,
              color=C["amber"], zorder=4, label=tr("P95", "P95"))
    ax_b.plot(dts, col(loop, "确认时延最大_ms"), marker="x", ms=3.4, lw=0, mew=1.1,
              color=C["red"], zorder=4, label=tr("最大", "max"))
    ax_b.axhline(500, color=MUTED, lw=0.9, ls=(0, (3, 2.4)), zorder=2)
    ax_b.text(dts[-1] * 0.95, 512, tr("最小出块间隔 500 ms", "min block time 500 ms"),
              ha="right", va="bottom", fontsize=5.3, color=MUTED)
    # Merkle 建根耗时（对应模型里的 $t_v$）；计时分辨率 1 ms，多数批次低于分辨率，
    # 绘图取下限 0.5 ms 表示「在计时分辨率以下」
    mk = [max(v, 0.5) for v in col(loop, "Merkle建根最大_ms")]
    ax_b.plot(dts, mk, marker="d", ms=2.8, lw=1.2, color=C["green"], zorder=3,
              label=tr("建根 $t_v$", "tree build $t_v$"))
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(tr("锚定间隔 $\\Delta t$ (ms)", "anchoring interval $\\Delta t$ (ms)"))
    ax_b.set_ylabel(tr("单批耗时 (ms)", "per-batch time (ms)"))
    ax_b.set_ylim(0.05, 3000)
    ax_b.legend(loc="upper left", frameon=False, fontsize=5.6, ncol=2,
                handlelength=1.4, columnspacing=0.8, borderaxespad=0.2,
                labelspacing=0.32)
    panel(ax_b, pnext())



def fig13():
    _standalone(body13, 13, "chain_route_cost_and_loop",
                "链上路线的成本对照与端到端锚定的单批耗时",
                "On-chain route cost comparison and per-batch time of the end-to-end anchoring loop", 5.9, 2.86)



def body16():
    """三条架构路线的方案级对比。

    纵轴统一为一个可对齐的量：**从决策发生到结果可释放**（ms，对数）。三条路线在
    这条轴上口径相同，不同之处在于该时段由什么构成——集中式与本文只有判定求值，
    逐条记录上链则在求值之外还要等该条记录获得共识确认。图内直接把构成写在柱顶，
    不再把 0.60 μs 与 508 ms 并称「决策时延」求倍率（第三轮审稿 R1）；容量与吞吐
    另行给出，因为二者是另外两个测量对象，不能与时段同轴比较。

    编号说明：D 节内两张数据图（跨机部署时延、条款映射与落实查询）在正文中
    均出现于本图（E 节）之前，故本图由第 14 号顺延为第 16 号，编号与出现次序
    保持一致。
    """
    W, H = 5.9, 2.86
    e1 = load("E1_决策时延与吞吐.csv")
    e8 = load("E8_路线成本对照.csv")
    tp = load("E8_吞吐上界.csv")

    # λ=200 代表点：异步锚定 = 本文/集中式的判定路径；同步锚定 = 逐条上链的确认等待
    lam = 200.0
    async_row = next(r for r in e1 if float(r["lambda"]) == lam and "异步" in r["模式"])
    sync_row = next(r for r in e1 if float(r["lambda"]) == lam and "同步" in r["模式"])

    per_record = next(r for r in e8 if str(r["路线"]).startswith("B3"))
    batch = next(r for r in e8 if "本文" in str(r["路线"]))
    b3_tps = next(r for r in tp if str(r["路线"]).startswith("B3"))
    ours_tps = next(r for r in tp if "本文" in str(r["路线"]))

    def cap_s(v):
        return ("%.1f" % v) if v < 100 else ("%.2f$\\times$10$^{%d}$"
                                             % (v / 10 ** int(np.log10(v)),
                                                int(np.log10(v))))

    names = [tr("集中式判定", "centralized"),
             tr("逐条记录上链", "per-record\non-chain"),
             tr("PGB-AC\n批次锚定", "PGB-AC\nbatch anchoring")]
    eval_us = float(async_row["P50_us"])              # 判定求值本身
    rel_ms = [eval_us / 1000.0, float(sync_row["P50_us"]) / 1000.0, eval_us / 1000.0]
    rel_lab = [tr("%.2f μs 求值" % eval_us, "%.2f μs evaluation" % eval_us),
               tr("%.2f μs 求值 ＋ 508 ms 确认等待" % eval_us,
                  "%.2f μs evaluation + 508 ms confirmation wait" % eval_us),
               tr("%.2f μs 求值" % eval_us, "%.2f μs evaluation" % eval_us)]
    cost = [None, float(per_record["单条记录平均gas"]), float(batch["单条记录平均gas"])]
    cost_plot = [np.nan if v is None else v for v in cost]

    fig, gs = _mk(W, H, 1, 2, width_ratios=[1.42, 1.0],
                  left=0.085, right=0.935, top=0.875, bottom=0.255, wspace=0.46)
    ax = fig.add_subplot(gs[0, 0]); style_axes(ax)
    ax2 = fig.add_subplot(gs[0, 1]); style_axes(ax2)
    panel(ax, pnext())

    # 左：决策→可释放的时段
    x = np.arange(3)
    colors = [C["gray"], C["red"], C["blue"]]
    for xi, v, c in zip(x, rel_ms, colors):
        ax.bar(xi, v, width=0.55, color=c, zorder=3)
        ax.text(xi, v * 1.45, rel_lab[xi], ha="center", va="bottom",
                fontsize=5.7, color=TEXT, linespacing=1.35)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 6e3)
    ax.set_xlim(-0.75, 2.75)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.9, linespacing=1.45)
    ax.set_ylabel(tr("求值 + 可选确认等待 (ms，对数)", "evaluation + optional confirm wait (ms, log)"),
                  fontsize=7.2)
    ax.tick_params(axis="x", pad=3.5)
    # 容量与吞吐不进这张轴：它们是另外两个测量对象，同轴比较正是第三轮审稿点出的问题。
    # 数值改在图题里交代。

    # 右：单记录链上成本
    for xi, v, c in zip(x, cost_plot, colors):
        if np.isnan(v):
            # 集中式无链上写入：在对数轴上不落柱，只标 0，避免用 10^0 的假高度占位
            ax2.text(xi, 1.15, "0", ha="center", va="bottom", fontsize=5.8,
                     color=MUTED)
        else:
            ax2.bar(xi, v, width=0.55, color=c, zorder=3)
            ax2.text(xi, v * 1.45, ("%.0f" % v if v >= 100 else "%.2f" % v),
                     ha="center", va="bottom", fontsize=5.8, color=TEXT)
    ax2.set_yscale("log")
    ax2.set_ylim(1, 4e5)
    ax2.set_xlim(-0.75, 2.75)
    ax2.set_xticks(x)
    # 右面板较窄，两行全名会互相挤压，用短名；全名在左面板已给出
    ax2.set_xticklabels([tr("集中式", "centralized"), tr("逐条上链", "per-record"),
                         tr("PGB-AC", "PGB-AC")], fontsize=5.9)
    ax2.set_ylabel(tr("单记录链上成本 (gas，对数)", "on-chain cost per record (gas, log)"),
                   fontsize=7.2)
    ax2.tick_params(axis="x", pad=3.5)



def fig16():
    _standalone(body16, 16, "baseline_comparison",
                "三条架构路线在决策时延与单记录链上成本上的对比",
                "Comparison of the three architectural routes on decision latency and per-record on-chain cost", 5.9, 2.86)



# ==================================================================== 图 15

def body15():
    """条款到策略模板的映射一致率与条款落实查询的开销。"""
    W, H = 5.9, 2.72
    mq = load("E12_条款映射质量.csv")
    q = load("E12_条款落实查询.csv")

    fig, gs = _mk(W, H, 1, 2, width_ratios=[1.0, 1.08], left=0.088, right=0.982, top=0.885, bottom=0.185, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)

    # ---- (a) 两组语料上的综合一致率：本文三阶段映射 vs 文本检索
    corpora = [("建模组", tr("建模组", "modelling set")),
               ("留出组", tr("留出组", "held-out set"))]
    series = [("本文 三阶段映射", tr("本文三阶段映射", "three-stage mapping"), C["blue"]),
              ("文本检索", tr("文本检索", "textual retrieval"), C["gray"])]
    x = np.arange(len(corpora))
    bw = 0.34
    for k, (key, lbl, colr) in enumerate(series):
        vals, txts = [], []
        for tag, _ in corpora:
            r = next(rr for rr in mq if rr["语料"] == tag and key in rr["方法"])
            vals.append(float(r["综合一致率"]))
            txts.append("%d/%d" % (int(r["一致条数"]), int(r["总条数"])))
        ax_a.bar(x + (k - 0.5) * bw, vals, width=bw, color=colr, zorder=3, label=lbl)
        for xi, v, t in zip(x + (k - 0.5) * bw, vals, txts):
            ax_a.text(xi, v + 0.024, t, ha="center", va="bottom", fontsize=5.9,
                      color=TEXT)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([c[1] for c in corpora], fontsize=6.6)
    ax_a.set_ylim(0, 1.34)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_ylabel(tr("综合一致率", "overall agreement rate"), fontsize=7.6)
    legend(ax_a, ncol=2, loc="upper center", fontsize=6.0)
    panel(ax_a, pnext())

    # ---- (b) 单条款落实查询与全量逐条核验的耗时（双对数）
    ns = [int(r["策略条数"]) for r in q]
    ax_b.plot(ns, [float(r["单条款查询耗时_ms"]) for r in q], "o-", color=C["blue"],
              lw=1.4, ms=4.0, zorder=4,
              label=tr("单条款落实查询", "per-clause query"))
    ax_b.plot(ns, [float(r["全量核验耗时_ms"]) for r in q], "s-", color=C["red"],
              lw=1.4, ms=4.0, zorder=4,
              label=tr("全量逐条核验", "full re-verification"))
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlim(24, 1500)
    ax_b.set_ylim(2.2e-3, 1.9)
    ax_b.set_xlabel(tr("策略条数 $|\\Pi|$", "number of policies $|\\Pi|$"), fontsize=7.6)
    ax_b.set_ylabel(tr("耗时 (ms，对数)", "time (ms, log)"), fontsize=7.6)
    legend(ax_b, ncol=1, loc="upper left", fontsize=6.0)
    r_last = float(q[-1]["开销比"])
    lo, hi = float(q[-1]["单条款查询耗时_ms"]), float(q[-1]["全量核验耗时_ms"])
    ax_b.annotate("", xy=(1000, hi), xytext=(1000, lo),
                  arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=0.8,
                                  shrinkA=2.0, shrinkB=2.0))
    ax_b.text(930, np.sqrt(lo * hi), "%.1f×" % r_last, ha="right", va="center",
              fontsize=6.4, color=MUTED)
    ax_b.text(0.97, 0.055, tr("四类查询场景判定均正确", "all four query cases correct"),
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.9,
              color=MUTED)
    panel(ax_b, pnext())



def fig15():
    _standalone(body15, 15, "clause_mapping_and_query",
                tr("条款映射的一致率与条款落实查询的开销",
            "Agreement rate of clause mapping and cost of clause-compliance queries"),
                "Agreement rate of clause mapping and cost of clause-compliance queries", 5.9, 2.72)



# ==================================================================== 图 14

def body14():
    """跨机部署下确认时延随节点间往返时延的变化，以及链上成本的无关性对照。

    横轴一律取每档实测的 ICMP 往返时延（不是 netem 的名义设定值），因此
    「确认时延—往返时延」这一关系与测试床的具体拓扑无关。
    """
    W, H = 5.9, 2.72
    d = load("E13_跨机时延.csv")

    def fv(r, key):
        v = r.get(key, "")
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # 无注入档（rtt0）的单程延时为 0，其横坐标取同一洁净状态下实测的往返基线
    def rtt_of(r):
        return fv(r, "rtt_avg") or 0.301

    fig, gs = _mk(W, H, 1, 2, width_ratios=[1.0, 1.0], left=0.094, right=0.978, top=0.878, bottom=0.192, wspace=0.31)
    ax_a = fig.add_subplot(gs[0, 0]); style_axes(ax_a)
    ax_b = fig.add_subplot(gs[0, 1]); style_axes(ax_b)

    # 纯延时档（无注入与五个单程延时档）与叠加抖动/丢包档分开画：
    # 后者名义往返同为 20 ms，混进折线只会把横坐标重复的点连成竖线。
    PURE = ["rtt0", "rtt1", "rtt5", "rtt20", "rtt50", "rtt200"]
    BIAS = ["jit5", "loss01", "loss1"]

    def series(dt, tags, key="p50"):
        rs = sorted((r for r in d if int(r["dt"]) == dt and r["tag"] in tags),
                    key=rtt_of)
        return [rtt_of(r) for r in rs], [fv(r, key) for r in rs]

    # ---- (a) 稳态档随节点间往返时延上升；小于出块间隔的档位由队列相位主导
    xs, ys = series(1000, PURE)
    ax_a.plot(xs, ys, "o-", color=C["blue"], lw=1.5, ms=4.2, zorder=6,
              label=tr("Δt = 1 s（稳态）", "Δt = 1 s (steady)"))
    xb, yb = series(1000, BIAS)
    if xb:
        ax_a.plot(xb, yb, "v", color=C["amber"], ms=4.0, lw=0, zorder=7,
                  alpha=0.9,
                  label=tr("叠加抖动/丢包（名义往返 20 ms）",
                           "with jitter/loss (nominal RTT 20 ms)"))
    xq, yq = series(100, PURE)
    ax_a.plot(xq, yq, "s--", color=C["gray"], lw=1.2, ms=3.6, zorder=4,
              label=tr("Δt = 100 ms（队列相位）", "Δt = 100 ms (queue phase)"))

    # 最小出块间隔为约束段（往返 50 ms 以内）的最小二乘拟合与外推。
    # 拟合只覆盖拐点之前；外推段用浅色虚线画到 200 ms，让实测点落在其上方的
    # 偏离量直接可读——这一偏离即共识轮次越过 500 ms 最小出块间隔的位置。
    # 下界取 1 ms：无注入档（0.31 ms）落在出块节拍与固定开销把曲线压平的一端，
    # 与 experiments/lan_rtt.py 的拟合口径一致（只用五个纯延时档）。
    ps = [(x, y) for x, y in zip(xs, ys) if 1.0 <= x <= 60 and y is not None]
    n = len(ps)
    mx = sum(p[0] for p in ps) / n
    my = sum(p[1] for p in ps) / n
    fk = sum((p[0] - mx) * (p[1] - my) for p in ps) / \
        sum((p[0] - mx) ** 2 for p in ps)
    fb = my - fk * mx
    gseg = np.logspace(np.log10(1.0), np.log10(50.5), 60)
    ax_a.plot(gseg, fb + fk * gseg, "-", color=C["red"], lw=1.3, zorder=5,
              label=tr("约束段拟合 $%.2f+%.2f\\times$RTT" % (fb, fk),
                       "fit for the constrained range "
                       "$%.2f+%.2f\\times$RTT" % (fb, fk)))
    gext = np.logspace(np.log10(50.5), np.log10(205), 40)
    ax_a.plot(gext, fb + fk * gext, ":", color=C["red"], lw=1.2, alpha=0.75,
              zorder=3)
    ax_a.axhline(500, ls="-.", lw=1.0, color=C["teal"], zorder=2)
    ax_a.text(0.24, 512, tr("最小出块间隔 500 ms", "min block interval 500 ms"),
              fontsize=5.8, color=C["teal"], ha="left", va="bottom")

    ax_a.set_xscale("log")
    ax_a.set_xlim(0.22, 300)
    ax_a.set_ylim(0, 1360)
    ax_a.set_xticks([0.3, 1, 5, 20, 50, 200])
    ax_a.set_xticklabels(["0.3", "1", "5", "20", "50", "200"], fontsize=6.4)
    ax_a.set_yticks([0, 250, 500, 750, 1000, 1250])
    ax_a.set_xlabel(tr("节点间实测往返时延 (ms，对数)",
                       "measured inter-node RTT (ms, log)"), fontsize=7.6)
    ax_a.set_ylabel(tr("确认时延中位 (ms)", "median confirmation latency (ms)"),
                    fontsize=7.6)
    top = max(v for v in ys if v is not None)
    for xi, yi in zip(xs, ys):
        if yi is None or yi < 100:
            continue
        if yi == top:
            ax_a.annotate(tr("%.0f\n共识轮次越过出块间隔" % yi,
                             "%.0f\nconsensus round exceeds\nthe interval" % yi),
                          xy=(xi * 0.90, yi * 0.97), xytext=(0.30, 0.66),
                          textcoords="axes fraction", fontsize=5.7,
                          color=C["blue"], ha="left", va="top",
                          arrowprops=dict(arrowstyle="-", lw=0.7,
                                          color=C["blue"]))
        else:
            ax_a.text(xi * 0.86, yi + 26, "%.0f" % yi, ha="right", va="bottom",
                      fontsize=6.0, color=C["blue"])
    legend(ax_a, ncol=1, loc="upper left", fontsize=5.5)
    panel(ax_a, pnext())

    # ---- (b) 阴性对照：各往返时延档叠在同一批内记录数上，gas 只随 k 变
    tags = []
    for r in d:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    marks = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    cols = [C["blue"], C["teal"], C["red"], C["gray"], C["blue"], C["teal"],
            C["red"], C["gray"], C["blue"]]
    n = len(tags)
    for i, tg in enumerate(tags):
        rs = sorted((r for r in d if r["tag"] == tg), key=lambda r: int(r["k"]))
        off = 1 + 0.05 * (i - (n - 1) / 2.0)
        ax_b.plot([int(r["k"]) * off for r in rs], [fv(r, "gas") for r in rs],
                  marks[i % len(marks)], color=cols[i % len(cols)], ms=3.5,
                  lw=0, zorder=4,
                  label=tr("往返 %.2f ms" % rtt_of(rs[0]),
                           "RTT %.2f ms" % rtt_of(rs[0])))
    ax_b.set_xscale("log")
    ax_b.set_xlim(12, 26000)
    ax_b.set_xticks([20, 100, 1000, 12000])
    ax_b.set_xticklabels(["20", "100", "1000", "12000"], fontsize=6.4)
    ax_b.set_xlabel(tr("批内记录数 $k$（对数）", "records per batch $k$ (log)"),
                    fontsize=7.6)
    ax_b.set_ylabel(tr("每批平均 gas", "mean gas per batch"), fontsize=7.6)
    lo = min(fv(r, "gas") for r in d)
    hi = max(fv(r, "gas") for r in d)
    pad = max(6.0, (hi - lo) * 0.45)
    ax_b.set_ylim(lo - pad, hi + pad)

    # 阴性对照的判据是「同一批内记录数 k 下、各往返时延档的 gas 相同」，
    # 所以这里取的是同 k 的组内极差，而不是全图的极差（全图极差由 k 引起，
    # 正是要说明的那条编码长度关系）。
    gby = {}
    for r in d:
        gby.setdefault(int(r["k"]), []).append(fv(r, "gas"))
    inner = max(max(v) - min(v) for v in gby.values() if len(v) > 1)
    ax_b.text(0.03, 0.94, tr("同 $k$ 下各档极差 %.0f gas" % inner,
                             "spread across RTT levels at equal $k$: %.0f gas"
                             % inner),
              transform=ax_b.transAxes, ha="left", va="top", fontsize=5.9,
              color=MUTED)
    legend(ax_b, ncol=3, loc="lower center", fontsize=5.3)
    panel(ax_b, pnext())



def fig14():
    _standalone(body14, 14, "lan_rtt",
                tr("跨机部署下确认时延随节点间往返时延的变化与链上成本的无关性",
            "Confirmation latency versus inter-node RTT in cross-host deployment "
            "and the RTT-independence of on-chain cost"),
                "Confirmation latency versus inter-node RTT in cross-host deployment "
         "and the RTT-independence of on-chain cost", 5.9, 2.72)



# 图号＝论文里的图号，键与上方各 save() 的第二个参数必须一致。
# 图 7~12 于 2026-09-19 按五章制的正文顺序重排：互认并入 IV-B 提到 7~10，
# 成本模型随 IV-D 后移到 11、12。键是图号、值是绘图函数，函数名 figNN 是
# 历史名（fig07 画的是图 11），判据只看 save() 的第二个参数与这里的键。
FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06,
        7: fig09, 8: fig10, 9: fig11, 10: fig12, 11: fig07, 12: fig08,
        13: fig13, 14: fig14, 15: fig15, 16: fig16}


# ==================================================== Scientific Reports 版
# SR 的 display items 上限是 8（图 + 表合计）。论文一原有 16 图，故按内容
# 相关性把原图两两纵向堆叠成 8 张；每块的高度取原图英寸高度，块内边距按原
# 比例内缩，因此每个面板的物理尺寸、字号与元素间距与单独出图时逐点相同
# （见 _stack），不存在缩放导致的压字。5 张表全部移入 Supplementary
# Information —— 表内的数值本来就已经画进这些图里，不进正文不丢信息。
# 各原图编号沿用历史名：body07 画的是原图 11、body08 画的是原图 12。
_SR_H = {1: 4.40, 2: 3.00, 3: 3.80, 4: 2.75, 5: 4.05, 6: 2.85, 7: 2.90,
         8: 3.55, 9: 4.15, 10: 4.05, 11: 4.10, 12: 4.20, 13: 2.86,
         14: 2.72, 15: 2.72, 16: 2.86}


def _sr(num, slug, cap_cn, cap_en, src_nums, pad_in=0.0, labels=None):
    """按原图编号组装一张合并图。src_nums 是原图的 save() 编号。

    原图编号与函数名序号并不一致（fig09 画的是原图 7），故由 FIGS 反查函数名
    再取 bodyNN；高度表 _SR_H 的键是原图编号。
    """
    parts = []
    for n in src_nums:
        k = int(FIGS[n].__name__[3:])
        parts.append((globals()["body%02d" % k], 5.9, _SR_H[n]))
    _stack(num, slug, cap_cn, cap_en, parts, pad_in=pad_in, labels=labels)


def sr_fig01():
    """图 1：分层架构与端到端时序。"""
    _sr(1, "sr_arch_timing",
        "PGB-AC 的分层结构与信任边界（a），以及端到端请求处理时序与四步决策（b）",
        "Layered structure and trust boundary of PGB-AC (a), and the end-to-end "
        "request processing timeline with the four-step decision (b)",
        [1, 2], pad_in=0.32, labels=["(a)", "(b)"])


def sr_fig02():
    """图 2：风险分量作用域与锚定时间关系。"""
    _sr(2, "sr_risk_anchoring",
        "三个风险分量的判别对象与互不重叠的作用域（a），以及锚定间隔、锚定粒度"
        "与暴露窗口之间的时间关系（b）",
        "Discriminative objects and disjoint scopes of the three risk components (a), "
        "and the time relation among the anchoring interval, the anchoring granularity "
        "and the exposure window (b)",
        [3, 4], pad_in=0.32, labels=["(a)", "(b)"])


def sr_fig03():
    """图 3：决策路径性能与锚定开销。"""
    _sr(3, "sr_latency_overhead",
        "决策路径的时延分位与判定容量（a-c），以及锚定建根与策略版本校验开销随"
        "规模的变化（d, e）",
        "Latency quantiles and decision capacity of the decision path (a-c), and the "
        "anchoring and policy-version verification cost versus scale (d, e)",
        [5, 6])


def sr_fig04():
    """图 4：跨中心互认与各通道检出能力。"""
    _sr(4, "sr_interop_detection",
        "跨中心互认的成功条件与对照结构下的权衡曲线（a, b），以及各检测通道的检出"
        "能力与误报率（c, d）",
        "Success condition of cross-centre endorsement and the trade-off curves under "
        "controls (a, b), and the per-channel detection capability and false-positive "
        "rate across attack scripts (c, d)",
        [7, 8])


def sr_fig05():
    """图 5：判别力消融、阈值工作点与处置分布。"""
    _sr(5, "sr_ablation_disposition",
        "风险分量的判别力消融与阈值工作点选择（a-c），以及处置分布、聚合方式对照"
        "与检测滞后（d-f）",
        "Discriminative-power ablation of the risk components and the selection of the "
        "threshold operating point (a-c), and the disposition distribution, aggregation "
        "comparison and detection lag (d-f)",
        [9, 10])


def sr_fig06():
    """图 6：锚定成本与链上路线。"""
    _sr(6, "sr_cost_chain_route",
        "总成本率随锚定间隔的变化与成本分解（a-c），以及链上路线的成本对照与端到端"
        "锚定的单批耗时（d, e）",
        "Total cost rate versus anchoring interval and its decomposition (a-c), and the "
        "on-chain route cost comparison together with the per-batch time of the "
        "end-to-end anchoring loop (d, e)",
        [11, 13])


def sr_fig07():
    """图 7：闭式解吻合度与跨机确认时延。"""
    _sr(7, "sr_closedform_lanrtt",
        "闭式锚定间隔与实测最优的吻合度及其标度律（a-d），以及跨机部署下确认时延"
        "随节点间往返时延的变化与链上成本的无关性（e, f）",
        "Agreement between the closed-form and measured optimum of the anchoring "
        "interval and its scaling law (a-d), and the confirmation latency versus "
        "inter-node RTT in cross-host deployment together with the RTT-independence of "
        "on-chain cost (e, f)",
        [12, 14])


def sr_fig08():
    """图 8：条款映射开销与三条架构路线对比。"""
    _sr(8, "sr_clause_route",
        "条款到策略模板的映射一致率与条款落实查询开销（a, b），以及三条架构路线在"
        "决策时延与单记录链上成本上的对比（c）",
        "Agreement rate of clause mapping and the cost of clause-compliance queries "
        "(a, b), and the comparison of the three architectural routes on decision "
        "latency and per-record on-chain cost (c)",
        [15, 16])


SR_FIGS = {1: sr_fig01, 2: sr_fig02, 3: sr_fig03, 4: sr_fig04,
           5: sr_fig05, 6: sr_fig06, 7: sr_fig07, 8: sr_fig08}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["cn", "en", "both"], default="both")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--set", choices=["full", "sr"], default="full",
                    help="full=原 16 图（IEEE 版，清单 figures/manifest.json）；"
                         "sr=SR 合并后的 8 张（清单 figures/manifest_sr.json）")
    a = ap.parse_args()
    table = SR_FIGS if a.set == "sr" else FIGS
    langs = ["cn", "en"] if a.lang == "both" else [a.lang]
    nums = a.only or sorted(table)
    F.PARTIAL = bool(a.only)   # 只重跑部分图时不要把清单里其余图删掉
    for lg in langs:
        set_lang(lg)
        os.makedirs(fig_dir(lg), exist_ok=True)
        print("== %s (%s) ==" % (lg, a.set))
        for n in nums:
            table[n]()
        # 两套图各写各的清单：合并图号 1..8 与原图号 1..16 重叠，写进同一份会
        # 互相覆盖，而 build_docx 的历史作业都读 manifest.json。见 figstyle.write_manifest。
        write_manifest("manifest_sr.json" if a.set == "sr" else "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
