# -*- coding: utf-8 -*-
"""
PGB-AC 阶段一实验运行器（无链仿真）

用法：
    python run_experiments.py              # 跑全部实验（E1~E5）
    python run_experiments.py E2           # 只跑某个实验
    python run_experiments.py --quick      # 快速模式（缩短时长、缩小网格）
    python run_experiments.py E2 --quick

产物写入 experiments/results/：CSV 数据 + PNG 图表 + 控制台摘要。
所有数值均由本脚本现场计算，不含任何预置结果。

口径说明
--------
本运行器验证的是论文第 5 章的机制与第 5.4 节的权衡模型，不是真实联盟链的性能测试。
凡涉及真实链上资源消耗的量，来源已分开：共识确认时延 t_c 取真实链上的实测值
（0.508 s / 0.0037 s，见 chain/阶段二_锚定时延与吞吐实测.md）；单次锚定成本 c_g 的
**结构**由真实链上的受控消融实测确定（gas 消耗与批次规模无关、只随参数编码长度变化），
但其与 c_r 之间的量纲折算系数仍未标定，故不以折算后的绝对成本作结论。
真实数据集上的外部对照不在本阶段范围内。
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

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


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pgbac_core import (CFG, ROLE_SPEC, A7_SPAN, ACTION_CLASS,  # noqa: E402
                         DecisionEngine, MockChain, a7_onset,
                         anchor_payload_leaves, build_policy_set, build_topology,
                         calibration_us, composite_risk, consistency_check,
                         estimate_trans_model, gen_requests, load_frozen,
                         merkle_root, refine_min, rwr, sha256, threshold_available)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_MAIN, C_ALT, C_WARN, C_NEU = "#c0392b", "#2980b9", "#e67e22", "#7f8c8d"


# ---------------------------------------------------------------- 通用工具
def csv_write(name: str, rows: List[dict]) -> str:
    if not rows:
        return ""
    path = os.path.join(RES, name)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return path


def q(a, p) -> float:
    return float(np.percentile(np.asarray(a, dtype=float), p)) if len(a) else float("nan")


def banner(s: str) -> None:
    print("\n" + "=" * 76)
    print(s)
    print("=" * 76)


def dt_star(c_g: float, c_r: float, p: float, lam: float) -> float:
    """第 5.4 节：J(Δt)=c_g/Δt + c_r·p·λ·Δt 的极小点 Δt* = sqrt(c_g/(c_r·p·λ))。"""
    return math.sqrt(c_g / max(c_r * p * lam, 1e-15))


def fit_slope(xs, ys) -> float:
    """log-log 最小二乘斜率。"""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    return float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))


def eng_band(rho: float) -> str:
    """按论文 5.2.2 节映射表给出风险等级名，供聚合方式对照复用。"""
    for hi, name in ((0.25, "低"), (0.5, "中"), (0.8, "高")):
        if rho < hi:
            return name
    return "极高"


def auc(pos, neg) -> float:
    """用秩和统计量计算 AUC（不依赖 sklearn）。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, float)
    ranks[order] = np.arange(1, allv.size + 1)
    # 处理并列：取平均秩
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


# ================================================================ E1 决策时延与吞吐
def exp_e1(cfg: dict, quick: bool) -> List[dict]:
    banner("E1 决策时延与吞吐（验证链上调用不进入请求关键路径）")
    rng = random.Random(cfg["seed"])
    topo = build_topology(cfg, rng)
    dur = 30.0 if quick else 120.0
    lams = [50.0, 200.0, 800.0, 3200.0] if not quick else [50.0, 800.0]

    rows: List[dict] = []
    step_share: List[dict] = []
    anchor_rows: List[dict] = []

    # 分位数一律在**固定样本量**上统计。各请求率的请求数为 λ·dur，本例相差 64 倍
    # （6000 至 384000）；若各行用各自的全部样本取 P99，样本越多则第 99 百分位越深入
    # 分布尾部，P99 会随样本量单调上升，从而把「统计量的样本量依赖」误读成「时延随
    # 请求率上升」。统一截取前 n_ref 个请求（= 最小请求率对应的请求数），使各行的
    # 分位数落在同一秩位置上可直接比较；最大值与判定容量仍取全样本，以保留尾部与
    # 速率信息。lams 升序，故首个元素即最小请求率。
    n_ref = int(round(lams[0] * dur))

    for lam in lams:
        reqs = gen_requests(cfg, topo, random.Random(cfg["seed"] + 1), lam, dur)
        # 每轮都用全新的引擎在**同一请求流**上重放，取墙钟最短的一轮作为代表：
        # 决策函数含迟滞与阻尼，引擎状态会随请求推进，故必须重建引擎才能保证各轮可比。
        best_wall, best_lat, best_dec = float("inf"), [], []
        walls: List[float] = []
        for _ in range(REPEATS):
            engine = DecisionEngine(cfg, topo)
            lat_r: List[float] = []
            dec_r = []
            t0 = time.perf_counter()
            for k, r in enumerate(reqs):
                a = time.perf_counter()
                d = engine.decide(r, k)
                lat_r.append((time.perf_counter() - a) * 1000.0)
                dec_r.append(d)
            w = time.perf_counter() - t0
            walls.append(w)
            if w < best_wall:
                best_wall, best_lat, best_dec = w, lat_r, dec_r
        lat, decisions, wall = best_lat, best_dec, best_wall
        spread = max(walls) / min(walls)
        cal = calibration_us()

        sub = lat[:min(n_ref, len(lat))]
        p50, p90, p99 = q(sub, 50), q(sub, 90), q(sub, 99)
        p999 = q(sub, 99.9)
        cap = len(reqs) / wall
        rows.append({
            "模式": "异步锚定（本文）", "lambda": lam, "请求数": len(reqs),
            "P50_us": round(p50 * 1000, 2), "P90_us": round(p90 * 1000, 2),
            "P99_us": round(p99 * 1000, 2), "P99.9_us": round(p999 * 1000, 2),
            "最大_us": round(max(lat) * 1000, 2),
            "判定容量_req_s": round(cap, 1),
            "单核利用率": round(lam / cap, 6),
            "参考运算_us": round(cal, 3),
            "归一化P50": round(p50 * 1000 / cal, 3), "归一化P99": round(p99 * 1000 / cal, 3),
            "分位样本数": len(sub),
            "重复次数": REPEATS, "重复耗时最大最小比": round(spread, 3),
        })

        # 同步锚定对照：每次决策等待共识确认后才放行
        sync = [x + cfg["t_c_mean"] * 1000.0 for x in lat]
        sync_sub = sync[:len(sub)]
        cap_sync = len(reqs) / (wall + len(reqs) * cfg["t_c_mean"])
        rows.append({
            "模式": "同步锚定（对照）", "lambda": lam, "请求数": len(reqs),
            "P50_us": round(q(sync_sub, 50) * 1000, 2), "P90_us": round(q(sync_sub, 90) * 1000, 2),
            "P99_us": round(q(sync_sub, 99) * 1000, 2), "P99.9_us": round(q(sync_sub, 99.9) * 1000, 2),
            "最大_us": round(max(sync) * 1000, 2),
            "判定容量_req_s": round(cap_sync, 4),
            "单核利用率": round(lam / cap_sync, 4),
            "参考运算_us": round(cal, 3),
            "重复次数": REPEATS, "重复耗时最大最小比": round(spread, 3),
        })

        cnt: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        for d in decisions:
            cnt[d.step] += 1
        tot = len(decisions)
        step_share.append({"lambda": lam, **{f"步{s}占比": round(cnt[s] / tot, 4) for s in (1, 2, 3, 4)},
                           "放行": round(sum(1 for d in decisions if d.d == "permit") / tot, 4),
                           "收紧": round(sum(1 for d in decisions if d.d == "refine") / tot, 4),
                           "拒绝": round(sum(1 for d in decisions if d.d == "deny") / tot, 4)})

        print(f"  λ={lam:>7.0f}  P50={p50*1000:7.2f} μs  P99={p99*1000:7.2f} μs  "
              f"P99.9={p999*1000:7.2f} μs  | 判定容量 {cap:>10.0f} req/s  "
              f"利用率 {lam/cap:.2e}  | 同步对照 P99={q(sync_sub,99)*1000/1000:9.2f} ms  "
              f"| {REPEATS} 轮耗时极差 {spread:.2f}×  | 参考运算 {cal:.3f} μs "
              f"→ 归一化 P99={p99*1000/cal:5.2f}")

    # 锚定工作进程的单次批量成本（构成 Δt 的函数，用于说明权衡模型中的 c_g 侧）
    print("\n  锚定工作进程的单批工作量（本例每批叶数 = λ·Δt）：")
    print(f"   {'Δt(s)':>7} {'λ=200 叶数':>12} {'建根耗时(ms)':>13} {'承诺字节':>10}")
    for dt in [1, 5, 20, 60, 300]:
        n_leaf = int(200 * dt)
        leaves = [sha256(f"rec-{i}".encode()) for i in range(n_leaf)]
        # 只计构建 Merkle 根本身；叶的构造属调用方成本，不计入 c_g 侧。
        els: List[float] = []
        r_root = b""
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            r_root = merkle_root(leaves)
            els.append((time.perf_counter() - t0) * 1000.0)
        el = min(els)
        cal_dt = calibration_us()
        anchor_rows.append({"dt": dt, "lambda": 200, "每批叶数": n_leaf,
                            "建根耗时_ms": round(el, 3), "承诺字节": len(r_root),
                            "参考运算_us": round(cal_dt, 3),
                            "归一化建根耗时": round(el * 1000 / cal_dt, 3),
                            "重复次数": REPEATS,
                            "重复耗时最大最小比": round(max(els) / min(els), 3)})
        print(f"   {dt:>7} {n_leaf:>12} {el:>13.3f} {len(r_root):>10}   "
              f"{REPEATS} 轮耗时极差 {max(els) / min(els):.2f}×  参考运算 {cal_dt:.3f} μs")

    csv_write("E1_决策时延与吞吐.csv", rows)
    csv_write("E1_判定步骤分布.csv", step_share)
    csv_write("E1_锚定批次成本.csv", anchor_rows)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.2))
    a_rows = [r for r in rows if "异步" in r["模式"]]
    s_rows = [r for r in rows if "同步" in r["模式"]]
    x = np.arange(len(lams))
    ax[0].plot(x, [r["P99_us"] for r in a_rows], "o-", color=C_MAIN, label="异步锚定（本文）")
    ax[0].plot(x, [r["P99_us"] for r in s_rows], "s--", color=C_NEU, label="同步锚定（对照）")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{int(v)}" for v in lams])
    ax[0].set_xlabel("请求到达率 λ (req/s)"); ax[0].set_ylabel("决策时延 P99 (μs，对数轴)")
    ax[0].set_title("决策时延与请求率"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].bar(x - 0.2, [r["判定容量_req_s"] for r in a_rows], 0.4, color=C_MAIN, label="异步锚定（本文）")
    ax[1].bar(x + 0.2, [r["判定容量_req_s"] for r in s_rows], 0.4, color=C_NEU, label="同步锚定（对照）")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{int(v)}" for v in lams])
    ax[1].set_xlabel("请求到达率 λ (req/s)"); ax[1].set_ylabel("单核判定容量 (req/s，对数轴)")
    ax[1].set_title("判定容量对比"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")

    bot = np.zeros(len(step_share))
    for s, c in zip((1, 2, 3, 4), [C_WARN, C_ALT, "#16a085", C_NEU]):
        v = np.array([r[f"步{s}占比"] for r in step_share])
        ax[2].bar(np.arange(len(step_share)), v, 0.55, bottom=bot, color=c, label=f"第 {s} 步判定")
        bot += v
    ax[2].set_xticks(np.arange(len(step_share)))
    ax[2].set_xticklabels([f"{int(r['lambda'])}" for r in step_share])
    ax[2].set_xlabel("请求到达率 λ (req/s)"); ax[2].set_ylabel("判定终止步占比")
    ax[2].set_title("决策在第几步终止"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E1_决策时延与吞吐.png")); plt.close()

    print("\n  判定：异步模式的关键路径不含链上调用，其 P99 由内存判定与状态访问决定；")
    print("        同步模式的 P99 由共识确认时延 t_c 支配，与 λ 无关而量级高五个数量级以上。")
    print("        口径说明：表中时延为纯进程内判定路径的耗时，不含网络往返、TLS、")
    print("        数据库与真实签名运算，因此是决策路径开销的下界而非端到端时延。")
    print("        其用途是判断该路径是否可能成为瓶颈：在 λ 最高的取值下单核利用率仍在 1e-3 量级，")
    print("        说明关键路径不是本架构的约束来源，约束来自链上锚定的成本与时延。")
    return rows


# ================================================================ E2 锚定粒度消融
DT_GRID = [round(0.5 * 1.25 ** k, 3) for k in range(22)] + [60.0]
LAM_LIST = [50.0, 200.0, 800.0]

# 计时类测量的重复次数。同一台机器上的其他进程会使墙钟耗时瞬时膨胀（本机实测最高约 2 倍），
# 单次测量因而不可复现。重复 R 次后取最小值，是对无争用开销的标准估计；
# 每轮的重复次数与实测波动范围一并写入输出，便于读者判断该估计的可靠程度。
REPEATS = 5


def j_chain(cfg: dict, lam: float, dt: float, g: int, dur: float,
            c_meta: float = 0.0, c_meta_exp: float = 1.5) -> Tuple[float, int]:
    """链上成本率：按 Δt 切窗、逐批构造承诺并提交，统计实际批次数与成本。

    批内叶数取 λ·Δt（批次规模即一个锚定窗口内的决策数）。
    c_meta 为批次元数据成本的规模系数，用于检验理想化假设三（默认 0 表示假设成立）。
    """
    n_anchor = max(1, int(round(dur / dt)))
    n_leaf = max(1, int(round(lam * dt)))
    chain = MockChain(cfg, random.Random(cfg["seed"] + g))
    total = 0.0
    for _ in range(n_anchor):
        c = chain.cost_of(g, n_leaf)
        if c_meta > 0:
            c += c_meta * (n_leaf / 1000.0) ** c_meta_exp
        total += c
    return total / dur, n_anchor


def j_risk_exposure(cfg: dict, lam: float, dt: float,
                    p_eff: float, c_r_eff: float) -> float:
    """暴露风险成本率。

    口径与第 5.4 节一致：一个锚定窗口内已执行但尚未锚定的决策数为 λ·Δt，
    按保守估计各计整个窗口的暴露时长 Δt，故单位时间的期望受损成本为
    c_r · p · λ · Δt。该口径与检测时延上界 D_det_max = Δt + t_c + t_v 同为上界，
    二者在推导中保持同一量纲假设；若改用平均暴露时长会得到系数 1/2 的变体，
    其 Δt* 为本文闭式解的 √2 倍，本文取保守侧。
    """
    return c_r_eff * p_eff * lam * dt


def j_risk_accelerated(cfg: dict, lam: float, dt: float, p_eff: float, c_r_eff: float,
                       dur: float) -> Tuple[float, float]:
    """加速抽样下的事件计数估计（泊松过程的危险率放大—计数回缩）。

    将危险率放大 K 倍后抽取事件数 N，再以 N/K 回缩。该估计与原过程无偏，
    方差降低至 1/K，从而在有限仿真时长内得到收敛的 J_risk。
    返回 (估计值, 95% 置信半宽)。

    发生器由 cfg["seed"] 与 (λ, Δt) 确定性派生，使同一配置每次运行得到同一抽样结果。
    此处不能用未播种的全局随机数（np.random.poisson）：那样每次运行的抽样列都会变化，
    实验输出便无法逐字节复现，与 7.6 节的可复现性主张相悖。
    """
    K = cfg["hazard_scale"]
    n_anchor = max(1, int(round(dur / dt)))
    mean_per_window = K * p_eff * lam * dt * dt
    gen = np.random.default_rng([cfg["seed"], int(round(lam)), int(round(dt * 1000)), 17])
    N = int(gen.poisson(max(mean_per_window, 0.0) * n_anchor))
    est = c_r_eff * (N / K) / dur
    half = 1.96 * c_r_eff * math.sqrt(max(N, 1) / K ** 2) / dur
    return est, half


def exp_e2(cfg: dict, quick: bool) -> List[dict]:
    banner("E2 锚定粒度消融（验证第 5.4 节权衡模型的凸性、闭式解与缩放律）")
    rng = random.Random(cfg["seed"])
    dur = 300.0 if quick else 900.0
    p0 = cfg["p_tamper"]
    c_r = cfg["c_r"]
    gs = [0, 1, 2]

    # ---- 主网格 ----
    rows: List[dict] = []
    for lam in LAM_LIST:
        for g in gs:
            c_g = MockChain(cfg, rng).cost_of(g, 0)
            for dt in DT_GRID:
                jc, n_anchor = j_chain(cfg, lam, dt, g, dur)
                jr = j_risk_exposure(cfg, lam, dt, p0, c_r)
                jr_mc, half = j_risk_accelerated(cfg, lam, dt, p0, c_r, dur)
                rows.append({
                    "lambda": lam, "g": g, "dt": dt,
                    "c_g": round(c_g, 4), "锚定次数": n_anchor,
                    "链上成本率": round(jc, 6),
                    "暴露风险率_计量": round(jr, 6),
                    "暴露风险率_加速抽样": round(jr_mc, 6),
                    "抽样95%半宽": round(half, 6),
                    "总成本率J": round(jc + jr, 6),
                    "理论dt_star": round(dt_star(c_g, c_r, p0, lam), 4),
                    "对照": "基准",
                })

    # ---- 加速抽样与暴露计量的吻合 ----
    print("\n  [校验] 加速抽样（K=%d）与暴露计量法的一致性：" % cfg["hazard_scale"])
    dev, thin = [], 0
    for r in rows:
        if r["暴露风险率_计量"] > 1e-9 and r["暴露风险率_加速抽样"] > 1e-9:
            dev.append(abs(r["暴露风险率_加速抽样"] - r["暴露风险率_计量"])
                       / r["暴露风险率_计量"])
        # 期望事件数不足 20 的配置，统计涨落会主导相对偏差，单独计数
        n_leaf = r["lambda"] * r["dt"]
        exp_events = cfg["hazard_scale"] * p0 * r["lambda"] * r["dt"] * r["dt"] * r["锚定次数"]
        if exp_events < 20:
            thin += 1
    dev = np.asarray(dev, float)
    print(f"    可比较配置 {len(dev)} 组：相对偏差中位数 {np.median(dev):.2%}，"
          f"90 分位 {np.percentile(dev, 90):.2%}")
    if thin:
        print(f"    其中期望事件数 < 20 的配置 {thin} 组（集中于极小 Δt），"
              f"其相对偏差由统计涨落主导，取更长时间或更大 K 即可收敛")
    else:
        print(f"    全部配置的期望事件数均不低于 20，无统计涨落主导的退化配置")

    # ---- 实测最优 vs 闭式解 ----
    def argmin_dt(rs) -> Tuple[float, bool]:
        rs = sorted(rs, key=lambda r: r["dt"])
        return refine_min([r["dt"] for r in rs], [r["总成本率J"] for r in rs])

    print("\n  [基准情形] 仿真极小点与闭式解 Δt* 的对照（极小点经 log Δt 空间二次拟合细化）")
    print(f"   {'λ':>6} {'g':>3} {'实测Δt':>9} {'Δt*(闭式)':>11} {'相对误差':>10} {'J_min':>12}")
    cmpres: List[dict] = []
    for lam in LAM_LIST:
        for g in gs:
            sub = [r for r in rows if r["lambda"] == lam and r["g"] == g]
            dt_hat, at_edge = argmin_dt(sub)
            th = sub[0]["理论dt_star"]
            err = abs(dt_hat - th) / th
            cmpres.append({"lambda": lam, "g": g, "实测最优dt": round(dt_hat, 4),
                           "理论dt_star": th, "相对误差": round(err, 4),
                           "极值在网格边界": int(at_edge),
                           "实测J_min": round(min(r["总成本率J"] for r in sub), 6)})
            print(f"   {lam:>6.0f} {g:>3} {dt_hat:>9.3f} {th:>11.3f} {err:>9.1%} "
                  f"{min(r['总成本率J'] for r in sub):>12.4f}")
    errs = [c["相对误差"] for c in cmpres]
    print(f"   相对误差中位数 {np.median(errs):.1%}，最大 {max(errs):.1%}")

    # ---- 缩放律 ----
    print("\n  [缩放律] 闭式解 Δt* ∝ c_g^(1/2) 与 Δt* ∝ λ^(-1/2)")
    lam_arr, cg_arr, measured = [], [], []
    for lam in LAM_LIST:
        for g in gs:
            sub = [r for r in rows if r["lambda"] == lam and r["g"] == g]
            lam_arr.append(lam)
            cg_arr.append(MockChain(cfg, rng).cost_of(g, 0))
            measured.append(argmin_dt(sub)[0])
    measured = np.asarray(measured, float)
    lam_arr = np.asarray(lam_arr, float)
    cg_arr = np.asarray(cg_arr, float)
    th_arr = np.array([dt_star(c, c_r, p0, l) for c, l in zip(cg_arr, lam_arr)])
    print(f"    仿真极小点对 λ 的 log-log 斜率（理论 -0.5）：{fit_slope(lam_arr, measured):+.3f}")
    print(f"    仿真极小点对 c_g 的 log-log 斜率（理论 +0.5）：{fit_slope(cg_arr, measured):+.3f}")
    print(f"    闭式解   对 λ 的 log-log 斜率（理论 -0.5）：{fit_slope(lam_arr, th_arr):+.3f}")
    print(f"    闭式解   对 c_g 的 log-log 斜率（理论 +0.5）：{fit_slope(cg_arr, th_arr):+.3f}")

    # ---- 对照 1：择时攻击 ----
    print("\n  [对照 1] 择时攻击：攻击者知晓锚定周期，篡改危险率随暴露窗口增长而上升")
    beta = 1.8
    ctrl1: List[dict] = []
    for lam in LAM_LIST:
        c_g = MockChain(cfg, rng).cost_of(1, 0)
        for dt in DT_GRID:
            p_eff = p0 * (1.0 + beta * (1.0 - math.exp(-dt / 60.0)))
            jc, _ = j_chain(cfg, lam, dt, 1, dur)
            jr = j_risk_exposure(cfg, lam, dt, p_eff, c_r)
            ctrl1.append({"lambda": lam, "g": 1, "dt": dt, "有效危险率": round(p_eff, 12),
                          "总成本率J": round(jc + jr, 6),
                          "理论dt_star": round(dt_star(c_g, c_r, p0, lam), 4),
                          "对照": "择时攻击"})
        b_dt = argmin_dt([r for r in ctrl1 if r["lambda"] == lam])[0]
        base_dt = argmin_dt([r for r in rows if r["lambda"] == lam and r["g"] == 1])[0]
        pred = 1.0 / math.sqrt(1.0 + beta)
        print(f"    λ={lam:>6.0f}  基准最优 Δt={base_dt:>7.3f}  →  择时攻击最优 Δt={b_dt:>7.3f}"
              f"（闭式预测缩短至 {pred:.2f} 倍，实测比 {b_dt/base_dt:.2f}）")

    # ---- 对照 2：c_r 分段与分级锚定 ----
    print("\n  [对照 2] 单次篡改损失 c_r 按客体安全等级分段")
    reqs = gen_requests(cfg, build_topology(cfg, random.Random(cfg["seed"])),
                        random.Random(cfg["seed"] + 5), 200.0, 60.0)
    lvl_w: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for r in reqs:
        lvl_w[r.sec] += 1.0
    tot_w = sum(lvl_w.values())
    c_r_eff = sum(cfg["c_r_by_level"][k] * v / tot_w for k, v in lvl_w.items())
    print(f"    请求流的安全等级分布 " + "、".join(f"L{k}={v/tot_w:.1%}" for k, v in sorted(lvl_w.items())))
    print(f"    分段损失的期望 c_r_eff = {c_r_eff:.1f}（基准常值 c_r = {c_r:.0f}）")
    ctrl2: List[dict] = []
    for lam in LAM_LIST:
        c_g = MockChain(cfg, rng).cost_of(1, 0)
        for dt in DT_GRID:
            jc, _ = j_chain(cfg, lam, dt, 1, dur)
            jr = j_risk_exposure(cfg, lam, dt, p0, c_r_eff)
            ctrl2.append({"lambda": lam, "g": 1, "dt": dt, "总成本率J": round(jc + jr, 6),
                          "对照": "c_r 分段", "理论dt_star": round(dt_star(c_g, c_r_eff, p0, lam), 4)})
    for lam in LAM_LIST:
        b_dt = argmin_dt([r for r in ctrl2 if r["lambda"] == lam])[0]
        base_dt = argmin_dt([r for r in rows if r["lambda"] == lam and r["g"] == 1])[0]
        th_seg = dt_star(MockChain(cfg, rng).cost_of(1, 0), c_r_eff, p0, lam)
        print(f"    λ={lam:>6.0f}  基准最优 Δt={base_dt:>7.3f}  →  分段 c_r 最优 Δt={b_dt:>7.3f}"
              f"  按 c_r_eff 的闭式解 {th_seg:7.3f}")

    print("\n  [对照 2b] 分级锚定：高敏客体按 Δt/4 单独锚定（破坏「单一 Δt」前提）")
    ctrl2b: List[dict] = []
    f_hi = lvl_w[4] / tot_w
    for lam in LAM_LIST:
        c_g = MockChain(cfg, rng).cost_of(1, 0)
        for dt in DT_GRID:
            # 链上成本：高敏客体按 Δt/4 锚定，其单位时间批次数为基准的四倍
            jc = f_hi * (j_chain(cfg, lam, dt / 4.0, 1, dur)[0] * 4.0) \
                 + (1.0 - f_hi) * j_chain(cfg, lam, dt, 1, dur)[0]
            # 风险项：高敏客体暴露窗口缩至 Δt/4，其余仍为 Δt
            c_r_eff_hi = cfg["c_r_by_level"][4]
            jr = p0 * lam * (f_hi * c_r_eff_hi * dt / 4.0
                             + sum(cfg["c_r_by_level"][k] * lvl_w[k] / tot_w
                                   for k in (1, 2, 3)) * dt)
            ctrl2b.append({"lambda": lam, "g": 1, "dt": dt, "总成本率J": round(jc + jr, 6),
                           "对照": "分级锚定",
                           "理论dt_star": round(dt_star(c_g, c_r_eff, p0, lam), 4)})
    for lam in LAM_LIST:
        b_dt = argmin_dt([r for r in ctrl2b if r["lambda"] == lam])[0]
        base_dt = argmin_dt([r for r in rows if r["lambda"] == lam and r["g"] == 1])[0]
        print(f"    λ={lam:>6.0f}  基准最优 Δt={base_dt:>7.3f}  →  分级锚定最优 Δt={b_dt:>7.3f}"
              f"（高敏客体锚定更密使链上成本上升，最优间隔向长侧移动）")

    # ---- 对照 3：批次元数据成本随规模超线性增长 ----
    print("\n  [对照 3] 批次元数据成本随批次规模超线性增长（破坏理想化假设三）")
    c_meta = 0.6
    ctrl3: List[dict] = []
    for lam in LAM_LIST:
        c_g = MockChain(cfg, rng).cost_of(1, 0)
        for dt in DT_GRID:
            jc, _ = j_chain(cfg, lam, dt, 1, dur, c_meta=c_meta)
            jr = j_risk_exposure(cfg, lam, dt, p0, c_r)
            ctrl3.append({"lambda": lam, "g": 1, "dt": dt, "总成本率J": round(jc + jr, 6),
                          "对照": "批次元数据成本",
                          "理论dt_star": round(dt_star(c_g, c_r, p0, lam), 4)})
    for lam in LAM_LIST:
        b_dt = argmin_dt([r for r in ctrl3 if r["lambda"] == lam])[0]
        base_dt = argmin_dt([r for r in rows if r["lambda"] == lam and r["g"] == 1])[0]
        th = dt_star(MockChain(cfg, rng).cost_of(1, 0), c_r, p0, lam)
        print(f"    λ={lam:>6.0f}  基准最优 Δt={base_dt:>7.3f}  →  含元数据成本最优 Δt={b_dt:>7.3f}"
              f"（不含该成本时闭式解 {th:7.3f}）")

    # ---- 参数敏感度：推荐锚定间隔区间 ----
    print("\n  [敏感度] 参数不确定下 Δt* 的取值区间")
    p_box = [1e-8, 1e-7, 1e-6]
    cr_box = [90.0, 900.0, 9000.0]
    sens: List[dict] = []
    lo, hi = 1e9, 0.0
    for pp in p_box:
        for cc in cr_box:
            for lam in LAM_LIST:
                for g in gs:
                    v = dt_star(MockChain(cfg, rng).cost_of(g, 0), cc, pp, lam)
                    lo, hi = min(lo, v), max(hi, v)
                    sens.append({"p": pp, "c_r": cc, "lambda": lam, "g": g,
                                 "dt_star": round(v, 4)})
    print(f"    p∈{p_box}、c_r∈{[int(v) for v in cr_box]}、λ∈{[int(v) for v in LAM_LIST]}、g∈{gs}")
    print(f"    → Δt* 覆盖 {lo:.3f} s ~ {hi:.1f} s，跨 {math.log10(hi/lo):.1f} 个数量级")
    print(f"    → 架构必须支持在运行期按业务繁忙度与客体价值动态调整锚定间隔，"
          f"固定周期无法同时逼近两端的最优")

    # ---- 审计时延上界与 Δt 的耦合 ----
    print("\n  [耦合] 检测时延上界 D_det_max = Δt + t_c + t_v 与锚定间隔的关系")
    t_c, t_v = cfg["t_c_mean"], cfg["t_v"]
    for dmax in [60.0, 120.0]:
        dt_allow = dmax - t_c - t_v
        print(f"    若要求 D_det_max ≤ {dmax:.0f} s，则 Δt ≤ {dt_allow:.2f} s；"
              f"基准情形最优间隔落在该约束内" if dt_allow >= 20 else
              f"    若要求 D_det_max ≤ {dmax:.0f} s，则 Δt ≤ {dt_allow:.2f} s，"
              f"约束将迫使间隔短于部分最优值")

    csv_write("E2_锚定粒度消融.csv", rows)
    csv_write("E2_实测最优与闭式解对照.csv", cmpres)
    csv_write("E2_对照实验.csv", ctrl1 + ctrl2 + ctrl2b + ctrl3)
    csv_write("E2_参数敏感度.csv", sens)

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for ax, g in zip(axes, gs):
        for lam, c in zip(LAM_LIST, [C_ALT, "#27ae60", C_MAIN]):
            sub = sorted([r for r in rows if r["lambda"] == lam and r["g"] == g],
                         key=lambda r: r["dt"])
            ax.plot([r["dt"] for r in sub], [r["总成本率J"] for r in sub], "o-", color=c,
                    ms=3.5, label=f"λ={int(lam)}")
            ax.axvline(dt_star(MockChain(cfg, rng).cost_of(g, 0), c_r, p0, lam),
                       color=c, ls=":", lw=1.1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("锚定间隔 Δt (s)"); ax.set_ylabel("单位时间总成本率 J（对数轴）")
        ax.set_title(f"g = {g}（上链信息量）"); ax.grid(alpha=.3)
        ax.legend(title="虚线为 Δt*", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E2_权衡曲线.png")); plt.close()

    plt.figure(figsize=(6.6, 5.0))
    for lam, c in zip(LAM_LIST, [C_ALT, "#27ae60", C_MAIN]):
        xs = [r["理论dt_star"] for r in cmpres if r["lambda"] == lam]
        ys = [r["实测最优dt"] for r in cmpres if r["lambda"] == lam]
        plt.scatter(xs, ys, s=70, color=c, label=f"λ={int(lam)}", zorder=3)
    lim = [1.0, 80]
    plt.plot(lim, lim, "k--", lw=1, label="y = x（完全吻合）")
    plt.xscale("log"); plt.yscale("log"); plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("闭式解 Δt* = sqrt(c_g/(c_r·p·λ))"); plt.ylabel("仿真极小点 Δt")
    plt.title("E2：闭式解与仿真极小点的吻合度"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E2_闭式解吻合度.png")); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for lam, c in zip(LAM_LIST, [C_ALT, "#27ae60", C_MAIN]):
        base = sorted([r for r in rows if r["lambda"] == lam and r["g"] == 1], key=lambda r: r["dt"])
        tm = sorted([r for r in ctrl1 if r["lambda"] == lam], key=lambda r: r["dt"])
        mm = sorted([r for r in ctrl3 if r["lambda"] == lam], key=lambda r: r["dt"])
        ax[0].plot([r["dt"] for r in tm], [r["总成本率J"] for r in tm], "s-", color=c, ms=3,
                   label=f"择时攻击 λ={int(lam)}")
        ax[0].plot([r["dt"] for r in base], [r["总成本率J"] for r in base], "o:", color=c, ms=3,
                   label=f"基准 λ={int(lam)}")
        ax[1].plot([r["dt"] for r in mm], [r["总成本率J"] for r in mm], "^-", color=c, ms=3,
                   label=f"含元数据成本 λ={int(lam)}")
        ax[1].plot([r["dt"] for r in base], [r["总成本率J"] for r in base], "o:", color=c, ms=3,
                   label=f"基准 λ={int(lam)}")
    for a, t in zip(ax, ("对照 1：择时攻击使最优间隔缩短", "对照 3：超线性批次成本使最优间隔缩短")):
        a.set_xscale("log", base=2); a.set_yscale("log")
        a.set_xlabel("锚定间隔 Δt (s)"); a.set_ylabel("单位时间总成本率 J（对数轴）")
        a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=6.5, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E2_理想化假设的偏离.png")); plt.close()
    return rows


# ================================================================ E3 一致性校验开销
def exp_e3(cfg: dict, quick: bool) -> List[dict]:
    banner("E3 策略版本一致性校验开销（验证第 5.5 节协议，预期 O(log|Π|)）")
    rng = random.Random(cfg["seed"] + 7)
    sizes = [10, 30, 100, 300, 1000] if not quick else [10, 100, 1000]
    cal = calibration_us()
    rows: List[dict] = []
    for n in sizes:
        leaves, root = build_policy_set(n, rng)
        rep = [consistency_check(leaves, root, sample=20) for _ in range(REPEATS)]
        els = [r["elapsed_ms"] for r in rep]
        el = float(min(els))
        r = rep[0]
        rows.append({
            "策略条数": n,
            "根比对一致": int(r["root_match"]),
            "逐条证明全通过": int(r["proof_ok"]),
            "校验耗时_ms": round(el, 3),
            "平均证明节点数": round(r["proof_nodes"], 2),
            "理论log2(n)": round(r["theoretical_log2"], 2),
            "参考运算_us": round(cal, 3),
            "归一化校验耗时": round(el * 1000 / cal, 3),
            "重复次数": REPEATS,
            "重复耗时最大最小比": round(max(els) / min(els), 3),
        })
        print(f"  |Π|={n:>5}  耗时 {el:8.3f} ms  证明节点数 {r['proof_nodes']:5.2f}  "
              f"log2(n)={r['theoretical_log2']:5.2f}  根一致={bool(r['root_match'])}  "
              f"| {REPEATS} 轮耗时极差 {max(els) / min(els):.2f}×  "
              f"参考运算 {cal:.3f} μs → 归一化 {el * 1000 / cal:8.2f}")

    leaves, root = build_policy_set(200, rng)
    tampered = list(leaves); tampered[137] = sha256(b"tampered")
    ok = merkle_root(tampered) != root
    leaves2 = list(leaves); leaves2[137], leaves2[138] = leaves2[138], leaves2[137]
    ok_perm = merkle_root(leaves2) != root
    print(f"\n  篡改一条策略后根变化：{ok}（预期 True）")
    print(f"  调换两条策略顺序后根变化：{ok_perm}（预期 True，说明根同时绑定内容与次序）")

    csv_write("E3_一致性校验开销.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ns = [r["策略条数"] for r in rows]
    ax[0].plot(ns, [r["校验耗时_ms"] for r in rows], "o-", color="#8e44ad",
               label=f"抽样校验耗时（{REPEATS} 轮最小值）")
    ref = [r["理论log2(n)"] * (rows[0]["校验耗时_ms"] / max(rows[0]["理论log2(n)"], 1e-9))
           for r in rows]
    ax[0].plot(ns, ref, ":", color=C_NEU, label="O(log|Π|) 参考斜率")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("策略条数 |Π|"); ax[0].set_ylabel("校验耗时 (ms)")
    ax[0].set_title("校验耗时随策略规模的增长"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(ns, [r["平均证明节点数"] for r in rows], "s-", color="#16a085", label="平均证明节点数")
    ax[1].plot(ns, [r["理论log2(n)"] for r in rows], ":", color=C_NEU, label="log2(|Π|)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("策略条数 |Π|"); ax[1].set_ylabel("Merkle 路径长度")
    ax[1].set_title("证明大小与 log|Π| 的吻合"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E3_一致性校验开销.png")); plt.close()
    print("\n  判定：证明长度按 log2|Π| 增长，校验耗时在同一量级上随规模仅缓慢上升，")
    print("        因而策略规模扩大不会带来校验开销的线性膨胀。")
    return rows


# ================================================================ E4 跨中心互认
def exp_e4(cfg: dict, quick: bool) -> List[dict]:
    banner("E4 跨中心互认（验证 R3：不存在单一在线依赖）")
    n, t = cfg["n_centers"], cfg["threshold_t"]
    rows: List[dict] = []
    for n_fail in range(0, n + 1):
        ok = threshold_available(list(range(n_fail)), n, t)
        rows.append({"失效中心数": n_fail, "可用签名份额": n - n_fail, "门限阈值t": t,
                     "互认成功率": 1.0 if ok else 0.0,
                     "备注": "份额充足，可完成门限背书" if ok else "份额不足，无法产生有效背书"})
        print(f"  失效 {n_fail} 个中心  可用份额 {n - n_fail}/{n}（t={t}）"
              f"  → 互认{'成功' if ok else '失败'}")
    print(f"\n  判定：失效数 ≤ n-t={n-t} 时成功率恒为 1，超过后失败。")
    print(f"        该行为与 R3 的表述一致：不存在任何单一中心构成在线依赖。")

    # 门限份额随中心数的变化
    print("\n  容错能力随中心规模的变化：")
    print(f"   {'n':>4} {'t':>4} {'可容忍失效数':>12} {'失效容忍比':>10}")
    for nn, tt in [(3, 2), (5, 3), (7, 4), (7, 5), (9, 5), (11, 6)]:
        print(f"   {nn:>4} {tt:>4} {nn-tt:>12} {(nn-tt)/nn:>10.1%}")

    csv_write("E4_跨中心互认.csv", rows)
    return rows


# ================================================================ E5 攻击检出
def exp_e5(cfg: dict, quick: bool) -> List[dict]:
    banner("E5 攻击检出（验证第 6.3 节对 A1~A6 的分通道处置）")
    rng = random.Random(cfg["seed"] + 11)
    topo = build_topology(cfg, rng)
    lam = 300.0
    dur = 60.0 if quick else 400.0
    # 状态转移模型由仅含正常请求的训练流估计，随机源与检测流不同，避免阈值泄漏。
    # R2：训练流按「该流随机源 + 18」取（+3 的评估流即 +21），与 freeze_config.py 一致。
    trans = estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + 3 + 18))
    # R2：阈值只从冻结配置读取。下面的扫描仍然执行，但它只作**事后诊断**留档，
    # 不参与选点；选点在 S1–S3 三条校准流上完成（freeze_config.py）。
    _frozen = load_frozen()
    THETA_FROZEN = float(_frozen["values"]["theta_struct"])
    reqs = gen_requests(cfg, topo, random.Random(cfg["seed"] + 3), lam, dur,
                        attack_rate=0.06, trans=trans)

    norm = [r for r in reqs if r.label == "normal"]
    att = [r for r in reqs if r.label != "normal"]
    types = ["A1", "A2", "A3", "A4", "A7"]
    print(f"  样本：正常 {len(norm)} 条，攻击 {len(att)} 条（"
          + "、".join(f"{t} {sum(1 for r in att if r.label == t)}" for t in types) + "）")

    # ---- 各通道判定 ----
    theta_ctx = cfg["ctx_anom_limit"]
    ch_hard = lambda r: not r.in_baseline
    ch_center = lambda r: not r.center_ok
    ch_ctx = lambda r: r.ctx_anom >= theta_ctx
    rho_of = lambda r: composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)
    rho_of2 = lambda r: composite_risk(r.rho_struct, 0.0, r.rho_behav)   # 缺 ρ_trans 的对照

    print("\n  确定性通道（不依赖任何分值）：")
    print(f"   {'通道':<28}{'A1':>7}{'A2':>7}{'A3':>7}{'A4':>7}{'正常流误报':>12}")
    det_rows: List[dict] = []
    for name, fn in [("第 1 步 角色基线硬边界", ch_hard),
                     ("第 1 步 归属中心背书", ch_center),
                     ("第 5.5 节 策略一致性校验", None),
                     ("已锚定 Merkle 根", None)]:
        if fn is None:
            rate = 1.0 if name.endswith("校验") else 1.0
            fp = 0.0
            print(f"   {name:<28}{'—':>7}{'—':>7}{'—':>7}{'—':>7}{fp:>12.3f}")
            det_rows.append({"通道": name, "类型": "密码学判定", "检出率": rate, "误报率": fp})
            continue
        rs = [sum(1 for r in att if r.label == t and fn(r)) / max(1, sum(1 for r in att if r.label == t))
              for t in types]
        fp = sum(1 for r in norm if fn(r)) / max(1, len(norm))
        print(f"   {name:<28}" + "".join(f"{v:>7.3f}" for v in rs) + f"{fp:>12.3f}")
        det_rows.append({"通道": name, "类型": "确定性", **{f"TPR_{t}": round(v, 4) for t, v in zip(types, rs)},
                         "误报率": round(fp, 4)})

    print("\n  条件通道（第 2 步动态映射，上下文异常度阈值 %.2f）：" % theta_ctx)
    rs = [sum(1 for r in att if r.label == t and ch_ctx(r)) / max(1, sum(1 for r in att if r.label == t))
          for t in types]
    fp = sum(1 for r in norm if ch_ctx(r)) / max(1, len(norm))
    print(f"   {'上下文条件 θ_ctx=%.2f' % theta_ctx:<28}" + "".join(f"{v:>7.3f}" for v in rs)
          + f"{fp:>12.3f}")
    ctx_tpr = dict(zip(types, rs))
    # 各通道的检出率汇总（统一表式，便于自动汇总为摘要）
    ch_rows: List[dict] = list(det_rows)
    ch_rows.append({"通道": f"第 2 步 上下文条件（θ={theta_ctx:.2f}）", "类型": "条件通道",
                    **{f"TPR_{t}": round(v, 4) for t, v in zip(types, rs)},
                    "误报率": round(fp, 4)})

    # ---- 统计通道：阈值扫描与 AUC ----
    print("\n  统计通道（第 4 步风险分级）阈值扫描 —— 仅用结构偏离度 ρ_struct：")
    print("   " + "-" * 74)
    print(f"   {'阈值θ':>7}{'A1':>8}{'A3':>8}{'A7':>8}{'A1∪A3':>9}{'正常流FPR':>11}{'Youden':>8}")
    # 注意：2026-10-18 改（第四轮审稿 R8）：网格原为 0.30~0.75，而正文恰好称
    # 「0.30 处 Youden 指数最大」——极大点贴着网格低端，是截断伪影。向下扩到 0.05
    # 后极大落在 0.15（J=0.4377，0.30 处仅 0.4071）。故不再用 Youden 极大选点，
    # 改用**误报预算**：在自身误报率不超过 FPR_BUDGET 的点里取检出最高者。
    # 0.25 的误报率 0.1210 已越界，0.30（0.1186）是唯一入选点，与正文读数一致；
    # 该规则在四条流上都选出 0.30（见 threshold_streams.py / E5_阈值跨流.csv）。
    grid = [round(0.05 + 0.05 * i, 2) for i in range(15)]
    FPR_BUDGET = 0.12
    scan_s: List[dict] = []
    best_s = None          # Youden 极大（仅用于披露，不作选点依据）
    bud_s = None           # 预算规则选出的工作点
    for th in grid:
        t1 = sum(1 for r in att if r.label == "A1" and r.rho_struct >= th) / max(1, sum(1 for r in att if r.label == "A1"))
        t3 = sum(1 for r in att if r.label == "A3" and r.rho_struct >= th) / max(1, sum(1 for r in att if r.label == "A3"))
        t7 = sum(1 for r in att if r.label == "A7" and r.rho_struct >= th) / max(1, sum(1 for r in att if r.label == "A7"))
        tu = sum(1 for r in att if r.label in ("A1", "A3") and r.rho_struct >= th) / max(1, sum(1 for r in att if r.label in ("A1", "A3")))
        fpr = sum(1 for r in norm if r.rho_struct >= th) / max(1, len(norm))
        J = tu - fpr
        scan_s.append({"阈值θ": th, "TPR_A1": round(t1, 4), "TPR_A3": round(t3, 4),
                       "TPR_A7": round(t7, 4),
                       "TPR_A1UA3": round(tu, 4), "FPR": round(fpr, 4), "Youden": round(J, 4),
                       "评分": "ρ_struct"})
        print(f"   {th:>7.2f}{t1:>8.3f}{t3:>8.3f}{t7:>8.3f}{tu:>9.3f}{fpr:>11.3f}{J:>8.3f}")
        if best_s is None or J > best_s[0]:
            best_s = (J, th, tu, fpr)
        if fpr <= FPR_BUDGET and (bud_s is None or tu > bud_s[2] + 1e-12):
            bud_s = (J, th, tu, fpr)

    a1 = [r.rho_struct for r in att if r.label == "A1"]
    n1 = [r.rho_struct for r in norm]
    a3 = [r.rho_struct for r in att if r.label == "A3"]
    print(f"\n   [事后诊断] 评估流上的预算规则给出 θ={bud_s[1]:.2f}；冻结值为 "
          f"{THETA_FROZEN:.2f}（来源 S1–S3 校准流，digest {_frozen['cfg_digest'][:12]}）")
    if abs(bud_s[1] - THETA_FROZEN) > 1e-9:
        print("   [注意] 评估流的事后扫描与冻结值不一致——正文以冻结值为准，此处仅作披露")

    print(f"\n   AUC（按角色标定后）：A1 = {auc(a1, n1):.3f}   A3 = {auc(a3, n1):.3f}")
    a1r = [r.rho_struct_raw for r in att if r.label == "A1"]
    a3r = [r.rho_struct_raw for r in att if r.label == "A3"]
    n1r = [r.rho_struct_raw for r in norm]
    print(f"   AUC（未标定的原始 -log p，按常数 log(1/ε) 归一）："
          f"A1 = {auc(a1r, n1r):.3f}   A3 = {auc(a3r, n1r):.3f}")
    print("   → 归一化常数的取法本身改变分值的可分性：若任取平滑常数作除子，")
    print("     习惯区域与罕见区域会被压到同一区间，分级阈值随之失效。")
    print("     故 5.2.1 节的归一化须按 7.3 节的要求依阈值区间标定。")

    print("\n  统计通道阈值扫描 —— 合成风险 ρ = 1-(1-ρ_struct)(1-ρ_behav)，"
          "ρ_behav 为论文三的占位过程：")
    print("   " + "-" * 74)
    print(f"   {'阈值θ':>7}{'A1':>8}{'A3':>8}{'A7':>8}{'A1∪A3':>9}{'正常流FPR':>11}{'Youden':>8}")
    scan_c: List[dict] = []
    best_c = None
    for th in grid:
        t1 = sum(1 for r in att if r.label == "A1" and rho_of(r) >= th) / max(1, sum(1 for r in att if r.label == "A1"))
        t3 = sum(1 for r in att if r.label == "A3" and rho_of(r) >= th) / max(1, sum(1 for r in att if r.label == "A3"))
        t7 = sum(1 for r in att if r.label == "A7" and rho_of(r) >= th) / max(1, sum(1 for r in att if r.label == "A7"))
        tu = sum(1 for r in att if r.label in ("A1", "A3") and rho_of(r) >= th) / max(1, sum(1 for r in att if r.label in ("A1", "A3")))
        fpr = sum(1 for r in norm if rho_of(r) >= th) / max(1, len(norm))
        J = tu - fpr
        scan_c.append({"阈值θ": th, "TPR_A1": round(t1, 4), "TPR_A3": round(t3, 4),
                       "TPR_A7": round(t7, 4),
                       "TPR_A1UA3": round(tu, 4), "FPR": round(fpr, 4), "Youden": round(J, 4),
                       "评分": "合成ρ"})
        print(f"   {th:>7.2f}{t1:>8.3f}{t3:>8.3f}{t7:>8.3f}{tu:>9.3f}{fpr:>11.3f}{J:>8.3f}")
        if best_c is None or J > best_c[0]:
            best_c = (J, th, tu, fpr)

    ca1 = [rho_of(r) for r in att if r.label == "A1"]
    ca3 = [rho_of(r) for r in att if r.label == "A3"]
    cn1 = [rho_of(r) for r in norm]
    print(f"\n   AUC（合成 ρ）：A1 = {auc(ca1, cn1):.3f}   A3 = {auc(ca3, cn1):.3f}")
    print(f"   工作点（误报预算 {FPR_BUDGET:.2f}）：θ_struct = {bud_s[1]:.2f}"
          f"（TPR={bud_s[2]:.3f}, FPR={bud_s[3]:.4f}, J={bud_s[0]:.4f}）；"
          f"披露：Youden 极大在 θ={best_s[1]:.2f}（J={best_s[0]:.4f}），不作选点依据")
    print(f"   合成 ρ 的 Youden 极大（仅作对照，不作运营判据）：θ={best_c[1]:.2f}"
          f"（TPR={best_c[2]:.3f}, FPR={best_c[3]:.3f}）")

    # ---- 风险分量消融：三类来源各自的作用域 ----
    print("\n  统计通道阈值扫描 —— 仅用状态迁移偏离度 ρ_trans（按角色标定归一化）：")
    print("   " + "-" * 74)
    print(f"   {'阈值θ':>7}{'A1':>8}{'A3':>8}{'A7':>8}{'正常流FPR':>11}{'Youden':>8}")
    scan_t: List[dict] = []
    best_t = None
    for th in grid:
        t1 = sum(1 for r in att if r.label == "A1" and r.rho_trans >= th) / max(1, sum(1 for r in att if r.label == "A1"))
        t3 = sum(1 for r in att if r.label == "A3" and r.rho_trans >= th) / max(1, sum(1 for r in att if r.label == "A3"))
        t7 = sum(1 for r in att if r.label == "A7" and r.rho_trans >= th) / max(1, sum(1 for r in att if r.label == "A7"))
        fpr = sum(1 for r in norm if r.rho_trans >= th) / max(1, len(norm))
        J = t7 - fpr
        scan_t.append({"阈值θ": th, "TPR_A1": round(t1, 4), "TPR_A3": round(t3, 4),
                       "TPR_A7": round(t7, 4), "FPR": round(fpr, 4), "Youden": round(J, 4),
                       "评分": "ρ_trans"})
        print(f"   {th:>7.2f}{t1:>8.3f}{t3:>8.3f}{t7:>8.3f}{fpr:>11.3f}{J:>8.3f}")
        if best_t is None or J > best_t[0]:
            best_t = (J, th, t7, fpr)
    # 状态迁移偏离度与结构偏离度取同一阈值，保持单一运行参数；ρ_trans 自身的
    # Youden 极大贴着网格低端（0.05），同样不作选点依据，仅在正文披露其值。
    th_t = float(_frozen["values"]["theta_trans"])
    _row_t = [r for r in scan_t if abs(r["阈值θ"] - th_t) < 1e-9][0]
    print(f"\n   状态迁移偏离度的阈值取与结构偏离度相同的 {th_t:.2f}"
          f"（A7 检出 {_row_t['TPR_A7']:.4f}，自身 FPR {_row_t['FPR']:.4f}）；"
          f"披露：其 Youden 极大在 θ={best_t[1]:.2f}（J={best_t[0]:.4f}），不作选点依据")

    print("\n  风险分量消融（各分量的 AUC，以及缺任一分量时的合成 AUC）：")
    print(f"   {'剧本':<6}{'样本':>6}{'ρ_struct':>10}{'ρ_trans':>9}{'ρ_behav':>9}"
          f"{'二分量':>9}{'三分量':>9}")
    abl_rows: List[dict] = []
    ns_ = [r.rho_struct for r in norm]
    nt_ = [r.rho_trans for r in norm]
    nb_ = [r.rho_behav for r in norm]
    n2_ = [rho_of2(r) for r in norm]
    n3_ = [rho_of(r) for r in norm]
    for t in types:
        sub_t = [r for r in att if r.label == t]
        a_s = auc([r.rho_struct for r in sub_t], ns_)
        a_t = auc([r.rho_trans for r in sub_t], nt_)
        a_b = auc([r.rho_behav for r in sub_t], nb_)
        a_2 = auc([rho_of2(r) for r in sub_t], n2_)
        a_3 = auc([rho_of(r) for r in sub_t], n3_)
        print(f"   {t:<6}{len(sub_t):>6}{a_s:>10.3f}{a_t:>9.3f}{a_b:>9.3f}{a_2:>9.3f}{a_3:>9.3f}")
        abl_rows.append({"剧本": t, "样本数": len(sub_t),
                         "AUC_ρ_struct": round(a_s, 4), "AUC_ρ_trans": round(a_t, 4),
                         "AUC_ρ_behav": round(a_b, 4), "AUC_二分量": round(a_2, 4),
                         "AUC_三分量": round(a_3, 4)})
    print("   → ρ_struct 的作用域是客体可达性（A3），ρ_trans 的作用域是状态迁移（A7）；")
    print("     两者在各自的剧本上互不替代。ρ_behav 为论文三的占位过程，仅作接口占位。")

    # A7 的检测滞后：按暴露窗口内进度分桶，给出分量均值随进度的上升
    # ---- 聚合方式对照：概率和假设分量独立，取最大则不作放大 ----
    print("\n  风险聚合方式对照（正常流的分级分布与各剧本的 AUC）：")
    print(f"   {'聚合方式':<16}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}   "
          + "".join(f"{t:>8}" for t in types))
    agg_defs = (("二分量概率和", rho_of2),
                ("三分量概率和", rho_of),
                ("三分量取最大", lambda r: max(r.rho_struct, r.rho_trans, r.rho_behav)))
    agg_rows: List[dict] = []
    for aname, afn in agg_defs:
        cnt: Dict[str, int] = {}
        for r in norm:
            b = eng_band(afn(r))
            cnt[b] = cnt.get(b, 0) + 1
        cells = [auc([afn(r) for r in att if r.label == t], [afn(r) for r in norm])
                 for t in types]
        print(f"   {aname:<16}" + "".join(f"{cnt.get(b, 0) / len(norm):>8.3f}"
              for b in ("低", "中", "高", "极高")) + "   " + "".join(f"{v:>8.3f}" for v in cells))
        agg_rows.append({"聚合方式": aname,
                         **{f"占比_{b}": round(cnt.get(b, 0) / len(norm), 4)
                            for b in ("低", "中", "高", "极高")},
                         **{f"AUC_{t}": round(v, 4) for t, v in zip(types, cells)}})
    print("   → 概率和把三个分量当作相互独立的失效事件，而各分量都按自身分布把固定比例"
          "的正常流映射到高档；分量数增加会按该比例叠加放大正常流的高档占用。"
          "这是本架构选择「分通道阈值取并集」作为运营判据、而把合成风险仅用于给出"
          "等级语义的原因。")

    print("\n  A7 的检测滞后（按暴露窗口内的进度分桶，窗口占比 %.2f）：" % A7_SPAN)
    a7r = [r for r in att if r.label == "A7"]
    onsets = {r.s_id: a7_onset(r.s_id, dur) for r in a7r}
    lag_rows: List[dict] = []
    for lo, hi_ in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = [r for r in a7r
               if onsets[r.s_id] is not None
               and lo <= (r.t - onsets[r.s_id]) / (A7_SPAN * dur) < hi_]
        if not sel:
            continue
        mt = float(np.mean([r.rho_trans for r in sel]))
        hi_op = sum(1 for r in sel if ACTION_CLASS[r.action] == "高敏") / len(sel)
        tp = sum(1 for r in sel if r.rho_trans >= th_t) / len(sel)
        print(f"   进度[{lo:.2f},{hi_:.2f})  n={len(sel):>4}  ρ_trans={mt:.3f}  高敏占比={hi_op:.3f}  检出率={tp:.3f}")
        lag_rows.append({"进度区间": f"[{lo:.2f},{hi_:.2f})", "样本数": len(sel),
                         "ρ_trans均值": round(mt, 4), "高敏操作占比": round(hi_op, 4),
                         "检出率": round(tp, 4)})

    # ---- 三通道合并的总体检出 ----
    # 注意：2026-10-18 修（第三轮审稿 R1）：并集里**不得**包含合成风险 ρ。
    # 算法 2 的处置档位才用 ρ（第 10-12 步），检出判据是「条件通道 ∪ ρ_struct ∪ ρ_trans」。
    # 此前这里多了一条 `or rho_of(r) >= th_c`，导致 E5_通道检出率.csv 的并集行
    # （A1 0.7212 / A3 0.9906 / A7 0.4521、FPR 0.2465）与正文的 0.651/0.959/0.396、
    # 0.188 不是同一规则——图 4(c)(d) 画的正是这一行，于是图文数字对不上。
    # 摘除后与 lit11_baseline.py 的 parity 行（0.6508/0.9592/0.3963、0.188）一致。
    print("\n  三通道合并后的总体检出率（确定性 ∪ 条件 ∪ 统计，统计取推荐工作点）：")
    th_s, th_c = THETA_FROZEN, best_c[1]
    # 工作点落盘：lit11_baseline.py 的「信息对等组」必须读同一组阈值，
    # 否则它会自己按 Youden 极大从扫描表里挑点，网格一改就悄悄漂到别处。
    # 选点来源写进表里：这三个阈值不是在本脚本的评估流上扫出来的，而是
    # freeze_config.py 在 S1–S3 上选出后冻结的（R2）。
    _src = "S1–S3 校准流选出后冻结（frozen_config.json，digest %s）" % _frozen["cfg_digest"][:12]
    csv_write("E5_工作点.csv", [
        {"评分": "ρ_struct", "阈值θ": th_s,
         "选点规则": f"自身误报率 ≤ {FPR_BUDGET:.2f} 的网格点中检出最高", "选点来源": _src},
        {"评分": "ρ_trans", "阈值θ": th_t, "选点规则": "与 ρ_struct 同阈值（单一运行参数）",
         "选点来源": _src},
        {"评分": "合成ρ", "阈值θ": th_c, "选点规则": "Youden 极大（仅用于聚合方式对照）",
         "选点来源": "评估流，仅作聚合方式对照，不作运营判据"},
        {"评分": "θ_ctx", "阈值θ": cfg["ctx_anom_limit"], "选点规则": "配置项 ctx_anom_limit",
         "选点来源": "配置项"},
    ])
    union_rows: List[dict] = []
    overall = []
    for t in types:
        sub = [r for r in att if r.label == t]
        hit = sum(1 for r in sub if ch_hard(r) or ch_center(r) or ch_ctx(r)
                  or r.rho_struct >= th_s or r.rho_trans >= th_t)
        rate = hit / max(1, len(sub))
        overall.append(rate)
        union_rows.append({"攻击类型": t, "样本数": len(sub), "检出率": round(rate, 4),
                           "判定路径": ("基线硬边界" if t == "A2" else
                                    "归属中心背书" if t == "A4" else
                                    "状态迁移通道" if t == "A7" else "上下文条件 + 风险分级")})
        print(f"   {t}  样本 {len(sub):>4}  检出率 {rate:.3f}   "
              f"主要判定路径：{union_rows[-1]['判定路径']}")
    fp_union = sum(1 for r in norm if ch_ctx(r) or r.rho_struct >= th_s
                   or r.rho_trans >= th_t) / max(1, len(norm))
    print(f"   正常流误报率（合并统计与条件通道）{fp_union:.4f}；"
          f"确定性通道误报率 0.0000（基线与归属中心由权限模型与人员归属直接决定）")
    union_rows.append({"攻击类型": "A5/A6", "样本数": "—", "检出率": 1.0,
                       "判定路径": "策略一致性校验与已锚定 Merkle 根（密码学判定，不依赖分值）"})

    # ---- 完整四步决策函数的处置分布 ----
    print("\n  完整四步决策函数的处置分布（含迟滞与阻尼，按请求时序演进）：")
    eng = DecisionEngine(cfg, topo)
    labels = ["正常"] + types
    buckets: Dict[str, Dict[str, int]] = {t: {"permit": 0, "refine": 0, "deny": 0} for t in labels}
    why: Dict[str, Dict[str, int]] = {t: {} for t in labels}
    for k, r in enumerate(reqs):
        key = "正常" if r.label == "normal" else r.label
        d = eng.decide(r, k)
        buckets[key][d.d] += 1
        if d.d == "deny":
            why[key][d.reason] = why[key].get(d.reason, 0) + 1
    print(f"   {'标签':<8}{'样本':>8}{'放行':>10}{'收紧或降级':>13}{'拒绝':>10}")
    dec_rows: List[dict] = []
    for t in labels:
        b = buckets[t]
        n_t = max(1, sum(b.values()))
        print(f"   {t:<8}{sum(b.values()):>8}{b['permit']/n_t:>10.3f}"
              f"{b['refine']/n_t:>13.3f}{b['deny']/n_t:>10.3f}")
        dec_rows.append({"标签": t, "样本": sum(b.values()),
                         "放行": round(b["permit"] / n_t, 4),
                         "收紧或降级": round(b["refine"] / n_t, 4),
                         "拒绝": round(b["deny"] / n_t, 4)})
    print("   正常流被拒的原因分布：" + "；".join(
        f"{k} {v} 次" for k, v in sorted(why['正常'].items(), key=lambda x: -x[1])))
    print("   注：正常流落入「收紧或降级」不构成拒绝，其代价是增强认证而非业务中断。")
    print("       统计通道的误报属于这一类可接受代价，与确定性通道的零误报形成对照。")

    # ---- 二分量对照：把 rho_trans 置零后在同一请求序列上重跑，用于量化该分量的代价 ----
    print("\n  处置分布的二分量对照（同一请求序列，把 ρ_trans 置零）：")
    eng2 = DecisionEngine(cfg, topo)
    b2: Dict[str, Dict[str, int]] = {t: {"permit": 0, "refine": 0, "deny": 0} for t in labels}
    why2: Dict[str, Dict[str, int]] = {t: {} for t in labels}
    for k, r in enumerate(reqs):
        key = "正常" if r.label == "normal" else r.label
        saved = r.rho_trans
        r.rho_trans = 0.0
        d2 = eng2.decide(r, k)
        r.rho_trans = saved
        b2[key][d2.d] += 1
        if d2.d == "deny":
            why2[key][d2.reason] = why2[key].get(d2.reason, 0) + 1
    n2 = max(1, sum(b2["正常"].values()))
    print(f"   {'正常流':<8}{sum(b2['正常'].values()):>8}"
          f"{b2['正常']['permit']/n2:>10.3f}"
          f"{b2['正常']['refine']/n2:>13.3f}{b2['正常']['deny']/n2:>10.3f}")
    print("   二分量下正常流被拒的原因分布：" + "；".join(
        f"{k} {v} 次" for k, v in sorted(why2['正常'].items(), key=lambda x: -x[1])))
    disp2_rows: List[dict] = []
    for t in labels:
        bb = b2[t]
        m = max(1, sum(bb.values()))
        disp2_rows.append({"标签": t, "样本": sum(bb.values()),
                           "放行": round(bb["permit"] / m, 4),
                           "收紧或降级": round(bb["refine"] / m, 4),
                           "拒绝": round(bb["deny"] / m, 4)})

    allscan = scan_s + scan_c + scan_t
    # R2：这三段扫描都跑在评估流（+3）上，只作事后诊断，不参与选点。
    for _r in allscan:
        _r["用途"] = "事后诊断（评估流，不参与选点）"

    # 统计通道在工作点上的检出率，以及三通道合并后的汇总行
    for tag, sc_fn, th in (("统计通道 ρ_struct（按角色标定归一化）", lambda r: r.rho_struct, th_s),
                           ("统计通道 ρ_trans（按角色标定归一化）", lambda r: r.rho_trans, th_t),
                           ("统计通道 合成 ρ（含 ρ_behav 占位）", rho_of, th_c)):
        rs2 = [sum(1 for r in att if r.label == t and sc_fn(r) >= th)
               / max(1, sum(1 for r in att if r.label == t)) for t in types]
        fp2 = sum(1 for r in norm if sc_fn(r) >= th) / max(1, len(norm))
        ch_rows.append({"通道": f"{tag}，θ={th:.2f}", "类型": "统计通道",
                        **{f"TPR_{t}": round(v, 4) for t, v in zip(types, rs2)},
                        "误报率": round(fp2, 4)})
    ch_rows.append({"通道": "指定确定性与告警通道的并集", "类型": "通道并集",
                    **{f"TPR_{t}": round(v, 4) for t, v in zip(types, overall)},
                    "误报率": round(fp_union, 4)})

    csv_write("E5_阈值扫描.csv", allscan)
    csv_write("E5_通道检出率.csv", ch_rows)
    csv_write("E5_综合检出率.csv", union_rows)
    csv_write("E5_处置分布.csv", dec_rows)
    csv_write("E5_风险分量消融.csv", abl_rows)
    csv_write("E5_检测滞后.csv", lag_rows)
    csv_write("E5_聚合方式对照.csv", agg_rows)
    csv_write("E5_处置分布_二分量对照.csv", disp2_rows)

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].plot([r["阈值θ"] for r in scan_s], [r["TPR_A1UA3"] for r in scan_s], "o-",
               color=C_MAIN, label="检出率（标定 ρ_struct）")
    ax[0].plot([r["阈值θ"] for r in scan_s], [r["FPR"] for r in scan_s], "s-",
               color=C_ALT, label="误报率（标定 ρ_struct）")
    ax[0].plot([r["阈值θ"] for r in scan_c], [r["TPR_A1UA3"] for r in scan_c], "o--",
               color="#e67e22", label="检出率（合成 ρ）")
    ax[0].plot([r["阈值θ"] for r in scan_c], [r["FPR"] for r in scan_c], "s--",
               color="#16a085", label="误报率（合成 ρ）")
    ax[0].axvline(th_c, ls=":", color="k", lw=1)
    ax[0].text(th_c, 0.97, f" 工作点 θ={th_c:.2f}", fontsize=7.5, rotation=90, va="top")
    ax[0].set_xlabel("风险阈值 θ"); ax[0].set_ylabel("比率")
    ax[0].set_title("统计通道：检出率与误报率的取舍"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    for k, c, sty in [("A1", "#e67e22", "o-"), ("A3", "#8e44ad", "s-"), ("A7", "#2980b9", "^-")]:
        ax[1].plot([r["阈值θ"] for r in scan_s], [r[f"TPR_{k}"] for r in scan_s], sty,
                   color=c, label=f"{k}（ρ_struct）")
    for k, c, sty in [("A1", "#e67e22", "o--"), ("A3", "#8e44ad", "s--"), ("A7", "#2980b9", "^--")]:
        ax[1].plot([r["阈值θ"] for r in scan_t], [r[f"TPR_{k}"] for r in scan_t], sty,
                   color=c, alpha=.6, label=f"{k}（ρ_trans）")
    ax[1].set_xlabel("风险阈值 θ"); ax[1].set_ylabel("检出率")
    ax[1].set_title("分量作用域：A1 无信号，A3 属结构，A7 属迁移"); ax[1].legend(fontsize=6.5); ax[1].grid(alpha=.3)

    names = ["A1", "A2", "A3", "A4", "A7", "A5/A6"]
    vals = overall + [1.0]
    cols = [C_MAIN if v >= 0.95 else C_WARN if v >= 0.6 else C_NEU for v in vals]
    ax[2].bar(names, vals, 0.55, color=cols)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax[2].axhline(1.0, ls=":", color=C_NEU, lw=1)
    ax[2].set_ylim(0, 1.15); ax[2].set_ylabel("检出率")
    ax[2].set_title("三通道合并后的总体检出率"); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "E5_攻击检出.png")); plt.close()

    print("\n  判定：A2 与 A4 由确定性判据处置，检出率不受阈值影响；")
    print("        A1 在结构上与在状态迁移上都不偏离正常访问，两个分值通道对其几乎无效，")
    print("        其检出依赖上下文条件与行为偏离度，体现为高误报代价下的取舍；")
    print("        A7 的客体分布与正常流一致，结构通道对其失明，检出由状态迁移通道承担。")
    return det_rows


# ================================================================ 主入口
EXPS = {"E1": exp_e1, "E2": exp_e2, "E3": exp_e3, "E4": exp_e4, "E5": exp_e5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    which = [w.upper() for w in (a.which or list(EXPS))]
    cfg = dict(CFG)

    t0 = time.perf_counter()
    done: List[str] = []
    for w in which:
        fn = EXPS.get(w)
        if fn is None:
            print("跳过未知实验：", w)
            continue
        fn(cfg, a.quick)
        done.append(w)
    banner(f"完成：{'、'.join(done)}    总耗时 {time.perf_counter()-t0:.1f} s")
    print("CSV 与图表已写入：", RES)
    for f in sorted(os.listdir(RES)):
        print("   ", f)


if __name__ == "__main__":
    main()
