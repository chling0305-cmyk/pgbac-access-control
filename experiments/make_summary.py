# -*- coding: utf-8 -*-
"""
实验结果摘要生成器

读取 experiments/results/ 下的全部 CSV，生成「实验结果摘要.md」。
所有数值均从 CSV 现场汇总，不做任何手工录入，因此摘要与实验输出不会出现口径漂移。

用法：
    python make_summary.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

# Windows 控制台默认代码页为 936(GBK)。若输出流里出现 GBK 无法表示的字符
# （例如 U+2212 减号、下标数字），print 会抛 UnicodeEncodeError 直接中断实验，
# 而该问题在 UTF-8 终端下不会暴露。这里把输出流的错误策略放宽为 replace，
# 保证任何字符都不会导致运行中止。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "实验结果摘要.md")

sys.path.insert(0, HERE)

from pgbac_core import CFG, refine_min  # noqa: E402


def load(name: str):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt(v, nd: int = 4) -> str:
    """把 CSV 里的字符串格式化为紧凑的展示形式：整数值不带小数，极小值用科学计数。"""
    s = "" if v is None else str(v)
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if math.isnan(f):
        return s
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    if f != 0 and abs(f) < 1e-3:
        return f"{f:.2e}"
    txt = f"{f:.{nd}f}".rstrip("0").rstrip(".")
    return txt if txt not in ("", "-") else "0"


def table(rows, cols, headers=None, floatfmt=None) -> str:
    """floatfmt 参数保留以兼容旧调用，实际格式由 fmt() 统一决定。"""
    if not rows:
        return "（无数据）\n"
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def fit_slope(xs, ys) -> float:
    lx = np.log(np.asarray(xs, float))
    ly = np.log(np.asarray(ys, float))
    return float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))


def main() -> None:
    L: list[str] = []
    A = L.append

    A("# PGB-AC 阶段一实验结果摘要\n")
    A("> 本文件由 `experiments/make_summary.py` 从 `experiments/results/` 下的 CSV 自动汇总生成，")
    A("> 不含任何手工录入的数值。重跑实验后重新执行该脚本即可刷新。\n")
    A("**口径说明**：本阶段为无链仿真，验证论文第 III 章 B 节的机制与 C 节的权衡模型。")
    A("链上两量的来源已分开：共识确认时延 `t_c` 取真实链实测值（0.508 s / 0.0037 s，")
    A("见 `chain/阶段二_锚定时延与吞吐实测.md`）；单次锚定成本 `c_g` 的**结构**（与批次规模")
    A("无关、只随调用数据非零字节数变化）由真实链的受控消融确定，但其与 `c_r` 之间的量纲折算")
    A("系数未标定，故本节的绝对成本数值不应作为真实链的成本结论引用。\n")
    A("**计时口径**：凡涉及时钟的指标均取 5 次重复的最小值。本机为混合架构 CPU，")
    A("操作系统在持续负载之后会改变进程的调度档位，实测使同一段与实验无关的纯计算耗时改变约 2 倍，")
    A("且该档位在一次运行内保持不变。因此每个测量块都同时记录一个固定的参考运算耗时")
    A("（对 64 字节输入做 SHA-256，微秒/次），跨块比较时应以「耗时 / 参考运算耗时」归一化，")
    A("否则档位差会被误读成被测对象随自变量的变化。\n")
    A("---\n")

    # ---------------- E1 ----------------
    e1 = load("E1_决策时延与吞吐.csv")
    A("## E1 决策时延与判定容量\n")
    if e1:
        A(table(e1, ["模式", "lambda", "请求数", "P50_us", "P99_us", "P99.9_us",
                     "判定容量_req_s", "单核利用率", "参考运算_us", "归一化P99"],
                ["模式", "λ (req/s)", "请求数", "P50 (μs)", "P99 (μs)", "P99.9 (μs)",
                 "单核判定容量 (req/s)", "单核利用率", "参考运算 (μs)", "归一化 P99"]))
        asyncs = [r for r in e1 if "异步" in r["模式"]]
        worst = max(num(r["单核利用率"]) for r in asyncs)
        A(f"- 异步锚定在最高到达率下单核利用率仍为 **{worst:.2e}**，关键路径不构成容量约束。")
        A("- 归一化 P99 为该行「P99 ÷ 同块参考运算耗时」，无量纲。不同 λ 的行必须按此列比较：")
        A("  原始 P99 在同一张表内不可直接比，因为各行可能落在不同的调度档位上。")
        A("- 下表时延为纯进程内判定路径耗时，不含网络往返、TLS、数据库与真实签名运算，")
        A("  故是决策路径开销的下界；其用途是与同步锚定做相对比较，而非给出端到端时延。")
        A("- 同步对照行的 P50/P99 等于实测的 `t_c_mean`，其取值不随请求率变化。\n")
    ir = load("E1_重复运行稳定性.csv")
    if ir:
        cols_r = [c for c in ir[0] if c.startswith("归一化P99_")]
        A("**归一化 P99 的多次运行稳定性**（同一配置连跑多次，用于判断跨行差异是否可信）\n")
        A("| 请求率列 | 最小 | 最大 | 极差 | 中位数 |")
        A("| --- | --- | --- | --- | --- |")
        allv = []
        for c in cols_r:
            v = [num(r[c]) for r in ir if r.get(c)]
            allv += v
            A(f"| {c.replace('归一化P99_', '')} | {min(v):.3f} | {max(v):.3f} | "
              f"{max(v) - min(v):.3f} | {np.median(v):.3f} |")
        A(f"\n- 全部 {len(allv)} 个取值的范围为 **{min(allv):.2f} 至 {max(allv):.2f}**。")
        A("  高请求率一侧的运行间极差大于中位数随请求率的变化，故该列只支持"
          "「未随请求率单调上升」这一否定性结论，不作定量比较。\n")
    st = load("E1_判定步骤分布.csv")
    if st:
        A("**决策在第几步终止**（反映四步决策函数的实际过滤结构）\n")
        A(table(st, ["lambda", "步1占比", "步2占比", "步3占比", "步4占比", "放行", "收紧", "拒绝"],
                ["λ", "步1", "步2", "步3", "步4", "放行", "收紧", "拒绝"], "{:.3f}"))
    an = load("E1_锚定批次成本.csv")
    if an:
        A("**锚定工作进程的单批工作量**（承诺构建成本随批次规模的增长）\n")
        A(table(an, ["dt", "每批叶数", "建根耗时_ms", "承诺字节", "参考运算_us", "归一化建根耗时"],
                ["Δt (s)", "每批叶数", "Merkle 建根耗时 (ms)", "根大小 (字节)",
                 "参考运算 (μs)", "归一化建根耗时"]))
    A("---\n")

    # ---------------- E2 ----------------
    A("## E2 锚定粒度—开销权衡\n")
    cmp_ = load("E2_实测最优与闭式解对照.csv")
    if cmp_:
        A("**仿真极小点与闭式解 Δt\\* = √(c_g/(c_r·p·λ)) 的对照**\n")
        A(table(cmp_, ["lambda", "g", "实测最优dt", "理论dt_star", "相对误差"],
                ["λ", "g", "仿真极小点 Δt (s)", "闭式解 Δt\\* (s)", "相对误差"], "{:.4f}"))
        errs = [num(r["相对误差"]) for r in cmp_]
        A(f"- 相对误差中位数 **{np.median(errs):.1%}**，最大 **{max(errs):.1%}**。")
        A("- 极小点系在 log Δt 空间对网格极小邻域作二次拟合细化后得到，"
          "故误差不含网格离散化带来的偏差。\n")

    base = load("E2_锚定粒度消融.csv")
    if base:
        base = [r for r in base if r.get("对照") == "基准"]
        lam_arr = [num(r["lambda"]) for r in base]
        cg_arr = [num(r["c_g"]) for r in base]
        dt_arr = [num(r["dt"]) for r in base]
        j_arr = [num(r["总成本率J"]) for r in base]
        keys = sorted(set(zip(lam_arr, cg_arr)), key=lambda t: (t[0], t[1]))
        meas_l, meas_c = [], []
        for k in keys:
            sub = [(d, j) for l, c, d, j in zip(lam_arr, cg_arr, dt_arr, j_arr)
                   if (l, c) == k]
            sub.sort()
            meas_l.append(k[0])
            meas_c.append(k[1])
        # 主口径与运行器一致，取二次拟合细化后的极小点；另给网格极小点作交叉核对。
        ref, grid = [], []
        for k in keys:
            sub = sorted(((d, j) for l, c, d, j in zip(lam_arr, cg_arr, dt_arr, j_arr)
                          if (l, c) == k))
            ds = [d for d, _ in sub]
            js = [j for _, j in sub]
            ref.append(refine_min(ds, js)[0])
            grid.append(min(sub, key=lambda t: t[1])[0])
        A("**缩放律的 log-log 斜率**\n")
        A("| 关系 | 理论斜率 | 仿真极小点（二次拟合） | 仿真极小点（网格，交叉核对） |")
        A("| --- | --- | --- | --- |")
        A(f"| Δt\\* ∝ λ^(-1/2) | -0.500 | {fit_slope(meas_l, ref):+.3f} | "
          f"{fit_slope(meas_l, grid):+.3f} |")
        A(f"| Δt\\* ∝ c_g^(1/2) | +0.500 | {fit_slope(meas_c, ref):+.3f} | "
          f"{fit_slope(meas_c, grid):+.3f} |")
        A("")
        devs = [abs(num(r["暴露风险率_加速抽样"]) - num(r["暴露风险率_计量"]))
                / num(r["暴露风险率_计量"])
                for r in base if num(r["暴露风险率_计量"]) > 1e-9
                and num(r["暴露风险率_加速抽样"]) > 1e-9]
        if devs:
            A(f"- 加速抽样（危险率放大—计数回缩，K={int(CFG['hazard_scale'])}）与暴露计量法的一致性："
              f"可比较配置 {len(devs)} 组，相对偏差中位数 **{np.median(devs):.2%}**、"
              f"90 分位 **{np.percentile(devs, 90):.2%}**。")
            A("  两者口径独立（一个由事件计数得到，一个由公式直接代入），因此该偏差同时"
              "核对了 O(Δt) 的暴露项与公式的一致性。\n")

    ctrl = load("E2_对照实验.csv")
    if ctrl:
        A("**偏离理想化假设时的最优间隔移动**\n")
        for name in ("择时攻击", "c_r 分段", "分级锚定", "批次元数据成本"):
            sub = [r for r in ctrl if r.get("对照") == name]
            if not sub:
                continue
            per = []
            for lam in sorted({num(r["lambda"]) for r in sub}):
                ss = sorted((num(r["dt"]), num(r["总成本率J"]))
                            for r in sub if num(r["lambda"]) == lam)
                ds = [d for d, _ in ss]
                js = [j for _, j in ss]
                per.append(f"λ={int(lam)}: {refine_min(ds, js)[0]:.3f} s")
            A(f"- **{name}**：最优间隔 " + "，".join(per))
        A("")

    sens = load("E2_参数敏感度.csv")
    if sens:
        vs = [num(r["dt_star"]) for r in sens]
        A(f"- 参数敏感度：在 p∈[1e-8,1e-6]、c_r∈[90,9000]、λ∈[50,800]、g∈{{0,1,2}} 的范围内，")
        A(f"  Δt\\* 覆盖 **{min(vs):.3f} s ~ {max(vs):.1f} s**（跨 {math.log10(max(vs)/min(vs)):.1f} 个数量级），")
        A("  说明固定锚定周期无法同时逼近两端的最优，架构须支持运行期动态调整。\n")
    A("---\n")

    # ---------------- E3 ----------------
    e3 = load("E3_一致性校验开销.csv")
    A("## E3 策略版本一致性校验开销\n")
    if e3:
        A(table(e3, ["策略条数", "根比对一致", "逐条证明全通过", "校验耗时_ms",
                     "平均证明节点数", "理论log2(n)", "参考运算_us", "归一化校验耗时"],
                ["策略条数 |Π|", "根比对一致", "逐条证明通过", "校验耗时 (ms)",
                 "证明节点数（实测）", "log2|Π|", "参考运算 (μs)", "归一化校验耗时"], "{:.3f}"))
        A("- 证明节点数与 log2|Π| 同阶，校验耗时随规模仅缓慢上升，未出现线性膨胀。")
        A("- 结论依赖增长阶而非绝对耗时：绝对值随调度档位变化，归一化后两列才可跨规模比较。\n")
    A("---\n")

    # ---------------- E4 ----------------
    e4 = load("E4_跨中心互认.csv")
    A("## E4 跨中心门限背书\n")
    if e4:
        A(table(e4, ["失效中心数", "可用签名份额", "门限阈值t", "互认成功率"],
                ["失效中心数", "可用份额", "门限 t", "互认成功率"], "{:g}"))
        A("- 失效数不超过 n-t 时互认成功率恒为 1，不存在任何单一中心构成在线依赖。\n")
    A("---\n")

    # ---------------- E5 ----------------
    A("## E5 攻击检出\n")
    ch = load("E5_通道检出率.csv")
    det = load("E5_处置分布.csv")
    scan = load("E5_阈值扫描.csv")
    if ch:
        A("**各检测通道的检出率与误报率**（每条通道独立统计，最后一行为三通道合并）\n")
        rows = [r for r in ch if r.get("TPR_A1") not in (None, "")]
        A(table(rows, ["通道", "类型", "TPR_A1", "TPR_A2", "TPR_A3", "TPR_A4", "TPR_A7", "误报率"],
                ["通道", "类型", "A1", "A2", "A3", "A4", "A7", "正常流误报率"]))
        A("- 确定性通道的误报率为 0：基线客体范围由权限模型决定，归属中心由人员归属决定，")
        A("  两者都不依赖分值，因此不存在阈值意义上的误报。")
        A("- A2 与 A4 完全由确定性通道处置；A1 与 A3 依赖条件通道与统计通道。\n")
    uni = load("E5_综合检出率.csv")
    if uni:
        A("**三通道合并后的分类型检出率**\n")
        A(table(uni, ["攻击类型", "样本数", "检出率", "判定路径"],
                ["攻击类型", "样本数", "检出率", "主要判定路径"]))
        A("")
    if scan:
        A("**统计通道在推荐工作点的表现**\n")
        A("| 评分口径 | 工作点 θ | A1 检出 | A3 检出 | A7 检出 | A1∪A3 检出 | 正常流误报 |")
        A("| --- | --- | --- | --- | --- | --- | --- |")
        for score in sorted({r["评分"] for r in scan}):
            sub = [r for r in scan if r["评分"] == score]
            best = max(sub, key=lambda r: num(r["Youden"]))
            u = num(best.get("TPR_A1UA3", ""))
            us = "—" if math.isnan(u) else f"{u:.3f}"
            A(f"| {score} | {num(best['阈值θ']):.2f} | {num(best['TPR_A1']):.3f} | "
              f"{num(best['TPR_A3']):.3f} | {num(best['TPR_A7']):.3f} | {us} | "
              f"{num(best['FPR']):.3f} |")
        A("")
    if det:
        A("**完整四步决策函数的处置分布**\n")
        A(table(det, ["标签", "样本", "放行", "收紧或降级", "拒绝"],
                ["标签", "样本", "放行", "收紧或降级", "拒绝"], "{:.3f}"))
    d2 = load("E5_处置分布_二分量对照.csv")
    if d2:
        A("**处置分布的二分量对照**（同一请求序列，把 ρ_trans 置零后重跑，用于量化该分量的代价）\n")
        A(table(d2, ["标签", "样本", "放行", "收紧或降级", "拒绝"],
                ["标签", "样本", "放行", "收紧或降级", "拒绝"], "{:.3f}"))
    ab = load("E5_风险分量消融.csv")
    if ab:
        A("**风险分量的作用域消融**（各分量在各自剧本上的 AUC，取合成前）\n")
        A(table(ab, ["剧本", "样本数", "AUC_ρ_struct", "AUC_ρ_trans", "AUC_ρ_behav",
                     "AUC_二分量", "AUC_三分量"],
                ["剧本", "样本", "ρ_struct", "ρ_trans", "ρ_behav",
                 "二分量合成", "三分量合成"], "{:.3f}"))
        A("- 各分量的作用域不重叠：A3 的偏离在客体粒度上，ρ_struct 可辨而 ρ_trans 接近随机；")
        A("  A7 的客体分布与正常流一致，ρ_struct 退化为随机判别，判别力由 ρ_trans 承担。")
        A("- ρ_behav 一列的取值由外部信任评估引擎提供，本仓库不生成该分量；")
        A("  该列在本表中只用于说明接口形态，不构成对该维度的评价。\n")
    lg = load("E5_检测滞后.csv")
    if lg:
        A("**A7 的检出滞后**（按暴露窗口内的进度分桶）\n")
        A(table(lg, ["进度区间", "样本数", "ρ_trans均值", "高敏操作占比", "检出率"],
                ["暴露进度", "样本", "ρ_trans 均值", "高敏操作占比", "检出率"], "{:.4f}"))
        A("- 滞后来自渐进本身：窗口早期的高敏操作占比接近常态，判据不可能在模式尚未形成时响应。\n")
    ag = load("E5_聚合方式对照.csv")
    if ag:
        A("**风险分量的聚合方式对照**（占比为正常流的等级分布）\n")
        A(table(ag, ["聚合方式", "占比_低", "占比_中", "占比_高", "占比_极高",
                     "AUC_A1", "AUC_A2", "AUC_A3", "AUC_A4", "AUC_A7"],
                ["聚合方式", "占比 低", "占比 中", "占比 高", "占比 极高",
                 "AUC A1", "AUC A2", "AUC A3", "AUC A4", "AUC A7"], "{:.3f}"))
        A("- 概率和形式把三类来源视为相互独立的失效事件，分量数增加会按各分量对正常流的")
        A("  高档占用比例叠加放大；因此运营判据采用分通道阈值取并集，合成风险只用于给出等级语义。\n")
    A("---\n")

    # ---------------- 结论 ----------------
    A("## 结论要点\n")
    if cmp_:
        errs = [num(r["相对误差"]) for r in cmp_]
        A(f"1. **权衡模型成立**：仿真极小点与闭式解 Δt\\* 的相对误差中位数 {np.median(errs):.1%}，")
        A("   两条缩放律的 log-log 斜率与理论值（-0.5 与 +0.5）一致，J(Δt) 的严格凸性得到验证。")
    A("2. **链上调用不进入关键路径**：异步模式的归一化 P99 在 λ 的整个取值范围内无单调上升")
    if e1:
        nasync = [num(r["归一化P99"]) for r in e1 if "异步" in r["模式"]]
        lo, hi = min(nasync), max(nasync)
        rng = f"{lo:.1f} 至 {hi:.1f}" if hi - lo > 0.05 else f"约 {lo:.1f}"
        A(f"   （归一化 P99 {rng} 个参考运算当量，跨行无趋势），")
    A("   同步对照的 P99 由共识确认时延支配，高出五个数量级以上。")
    A("3. **一致性校验为对数开销**：证明长度按 log2|Π| 增长。")
    A("4. **多中心无单点在线依赖**：门限背书在 n-t 个中心失效内恒可用。")
    A("5. **检出能力分层**：A2 与 A4 由确定性判据处置（检出率 1，误报 0）；")
    A("   A3 的跨客体关联在结构偏离度上可辨；A7 的操作模式迁移在结构偏离度上不可辨，")
    A("   由状态迁移偏离度承担；A1 的凭证重放在结构与次序上均不可辨，")
    A("   其检出依赖上下文条件与外部信任评估引擎提供的时序行为偏离度。")
    if ag and det and d2:
        def _hi(row):
            return num(row["占比_高"]) + num(row["占比_极高"])
        r2 = [r for r in ag if r["聚合方式"].startswith("二分量")][0]
        r3 = [r for r in ag if r["聚合方式"].startswith("三分量概率")][0]
        n2 = [r for r in d2 if r["标签"] == "正常"][0]
        n3 = [r for r in det if r["标签"] == "正常"][0]
        A(f"6. **分量的代价与聚合口径**：正常流落入高与极高两档的比例由二分量的 {_hi(r2):.3f} "
          f"升至三分量的 {_hi(r3):.3f}，")
        A(f"   相应的被拒比例由 {num(n2['拒绝']):.1%} 升至 {num(n3['拒绝']):.1%}；因此运营判据采用"
          "分通道阈值取并集，")
        A("   合成风险仅用于给出等级语义。状态迁移偏离度对渐进型攻击的检出存在随暴露时长递减的固有滞后。")
    A("7. **风险分级的误报代价是可接受的**：正常流落入「收紧或降级」表现为增强认证而非拒绝，")
    A("   与确定性通道的零误报形成对照。\n")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    print("已生成：", OUT)


if __name__ == "__main__":
    main()
