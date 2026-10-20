# -*- coding: utf-8 -*-
"""E5b 概率和合成的档位放大：成因、无效改法与有效改法

问题（论文一第 8.2 节点名为后续工作）
------------------------------------
E5 实测：正常流落入「高+极高」两档的比例，二分量概率和 0.1411，三分量概率和 0.2028。
论文把原因归为「概率和把三类来源当作相互独立的失效事件，各分量按自身分布把固定比例的
正常流映射到高档，分量数增加后这些比例按独立事件叠加」，并给出改进方向：
「把各分量在合成前重标定到相同的正常流占用率」。

本脚本把这一步做实，并区分三个层级的重标定，逐层用实测判定。

三层改法与实测判定
------------------
A 分量级·档位重标定（朴素等值 p=q*）      -> 更差（放大不降反升）
B 分量级·档位重标定（Sidak 修正 p_n）     -> 更差
C 分量级·分布重标定（PIT 到同一参考分布） -> 大幅更差
D 合成级·分档阈值重标定（按该 n 的正常流分位数定阈值）-> 命中目标，AUC 不变

机理（由 [3][4] 的实测给出）
  概率和 R = 1-prod(1-rho_k) 恒不小于 max_k rho_k，且分量各取中档即可越过 0.8：
  三个分量都取 0.6 时 R=0.936。放大由**分量取值量级**驱动，与各分量自身的档位占用率
  无关，故 A、B 触及不到机制。而 C 把高度偏斜的近零值分量（rho_trans 在正常流上
  97.9% 落在低档）抬到参考分布的 0.5 量级，等于给每个分量注入虚假量级，故更差。

有效改法 D 的含义：分级阈值的语义从「绝对分值」改为「正常流分位」，即阈值随 n 重定。
该操作为合成分的单调变换，排序不变，故各剧本 AUC 与未校正口径逐值相同。

工况一致性（由 [6] 给出）：标定流的工况须与运行工况同分布——标定流若取纯正常流、
运行流含攻击，则同一主体在攻击请求前后的正常请求的 rho_trans 会被抬高，分位数失真。

用法：
    python calib_agg.py            # 正式口径（与 E5 相同：lam=300, dur=400）
    python calib_agg.py --quick    # 快速口径（dur=60，仅用于冒烟）
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pgbac_core import (CFG, build_topology, composite_risk,  # noqa: E402
                         estimate_trans_model, gen_requests)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

BANDS = ("低", "中", "高", "极高")
KEYS = ("rho_struct", "rho_trans", "rho_behav")


# ---------------------------------------------------------------- 基础工具
def eng_band(rho: float) -> str:
    """与 run_experiments.eng_band 同口径（论文 5.2.2 节映射表，固定阈值）。"""
    for hi, name in ((0.25, "低"), (0.5, "中"), (0.8, "高")):
        if rho < hi:
            return name
    return "极高"


def occ_fixed(rows, fn) -> Dict[str, float]:
    cnt = {b: 0 for b in BANDS}
    for r in rows:
        cnt[eng_band(fn(r))] += 1
    n = max(1, len(rows))
    return {b: cnt[b] / n for b in BANDS}


def occ_thresh(rows, fn, th: Sequence[float]) -> Dict[str, float]:
    cnt = {b: 0 for b in BANDS}
    for r in rows:
        x = fn(r)
        b = ("低" if x < th[0] else "中" if x < th[1] else "高" if x < th[2] else "极高")
        cnt[b] += 1
    n = max(1, len(rows))
    return {b: cnt[b] / n for b in BANDS}


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, float)
    ranks[order] = np.arange(1, allv.size + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def midrank_pit(calib_vals: np.ndarray):
    """mid-rank 经验分位映射（对并列取中点），保证单调且确定。

    对含大量并列的离散分值，mid-rank 会把整段并列质量映射到同一个值，
    这是该做法在偏斜分布上的固有性质，不是实现缺陷。
    """
    sv = np.sort(np.asarray(calib_vals, float))
    m = sv.size

    def f(x: float) -> float:
        i_less = int(np.searchsorted(sv, x, side="left"))
        i_le = int(np.searchsorted(sv, x, side="right"))
        c_eq = i_le - i_less
        return (i_less + 0.5 * max(c_eq, 1)) / m

    return f


def band_calib_map(calib_vals: np.ndarray, p_target: float, hi: float = 0.8):
    """按「正常流高档占用率 = p_target」做分量级分段线性重标定（保留原量级）。"""
    a = np.asarray(calib_vals, float)
    t = min(max(float(np.quantile(a, 1.0 - p_target)), 0.0), 1.0)
    lo_span = t if t > 1e-12 else 1e-12
    hi_span = (1.0 - t) if (1.0 - t) > 1e-12 else 1e-12

    def f(x: float) -> float:
        if x <= t:
            return hi * (x / lo_span)
        return hi + (1.0 - hi) * ((x - t) / hi_span)

    return f, t


def q_thresholds(calib_vals, props: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(calib_vals, float)
    return (float(np.quantile(a, props[0])),
            float(np.quantile(a, props[0] + props[1])),
            float(np.quantile(a, props[0] + props[1] + props[2])))


def synth_sum(maps: Dict[str, object]):
    def fn(r):
        out = 1.0
        for k, f in maps.items():
            out *= (1.0 - min(max(f(getattr(r, k)), 0.0), 1.0))
        return 1.0 - out
    return fn


def fmt_occ(o: Dict[str, float]) -> str:
    return "".join(f"{o[b]:>8.4f}" for b in BANDS) + f"{o['高'] + o['极高']:>11.4f}"


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--lam", type=float, default=300.0)
    args = ap.parse_args()

    cfg = CFG
    lam = args.lam
    dur = args.dur if args.dur is not None else (60.0 if args.quick else 400.0)

    topo = build_topology(cfg, random.Random(cfg["seed"] + 11))
    trans = estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + 21))

    def stream(seed_off: int, rate: float):
        return gen_requests(cfg, topo, random.Random(cfg["seed"] + seed_off), lam, dur,
                            attack_rate=rate, trans=trans)

    det = stream(3, 0.06)                       # 检测流（与 E5 一致）
    norm = [r for r in det if r.label == "normal"]
    att = [r for r in det if r.label != "normal"]
    types = ["A1", "A2", "A3", "A4", "A7"]

    pure = [r for r in stream(3, 0.0) if r.label == "normal"]          # 同种子纯正常流
    cal_mix = [r for r in stream(31, 0.06) if r.label == "normal"]     # 标定流·混合工况
    cal_pure = [r for r in stream(31, 0.0) if r.label == "normal"]     # 标定流·纯正常

    print("=" * 80)
    print("E5b 概率和合成的档位放大：成因、无效改法与有效改法")
    print("=" * 80)
    print(f"  口径：lam={lam:g} req/s, dur={dur:g} s")
    print(f"  检测流 seed+3/含攻击：正常 {len(norm)}，攻击 {len(att)}（"
          + "、".join(f"{t} {sum(1 for r in att if r.label == t)}" for t in types) + "）")
    print(f"  同种子纯正常流 seed+3/无攻击：{len(pure)}")
    print(f"  标定流 seed+31/含攻击：{len(cal_mix)}；seed+31/无攻击：{len(cal_pure)}")

    comp2 = lambda r: composite_risk(r.rho_struct, 0.0, r.rho_behav)
    comp3 = lambda r: composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)
    compmax = lambda r: max(r.rho_struct, r.rho_trans, r.rho_behav)

    cal_vals = {k: np.array([getattr(r, k) for r in cal_mix], float) for k in KEYS}

    # ---- [0] 基线复核 ----
    o2, o3, om = occ_fixed(norm, comp2), occ_fixed(norm, comp3), occ_fixed(norm, compmax)
    p2_hi = o2["高"] + o2["极高"]
    print("\n  [0] 基线复核（检测流正常流，应与 E5_聚合方式对照.csv 一致）")
    print(f"   {'聚合方式':<16}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}{'高+极高':>11}")
    for nm, o in (("二分量概率和", o2), ("三分量概率和", o3), ("三分量取最大", om)):
        print(f"   {nm:<16}{fmt_occ(o)}")

    props2 = occ_fixed(cal_mix, comp2)
    props = (props2["低"], props2["中"], props2["高"], props2["极高"])
    q_star = props[2] + props[3]
    print(f"   → 设计目标（标定流·混合工况 二分量口径）：低/中/高/极高 = "
          + "/".join(f"{p:.4f}" for p in props) + f"，高+极高 q*={q_star:.4f}")

    # ---- [1] 各分量在正常流上的档位与量级 ----
    print("\n  [1] 各分量在正常流（标定流·混合工况）上的档位占用率与取值量级")
    print(f"   {'分量':<12}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}{'高+极高':>10}"
          f"{'均值':>9}{'P50':>8}{'P90':>8}")
    for k in KEYS:
        o = occ_fixed(cal_mix, lambda r, kk=k: getattr(r, kk))
        v = cal_vals[k]
        print(f"   {k:<12}" + "".join(f"{o[b]:>8.4f}" for b in BANDS)
              + f"{o['高'] + o['极高']:>10.4f}{v.mean():>9.4f}"
              f"{float(np.quantile(v, 0.5)):>8.4f}{float(np.quantile(v, 0.9)):>8.4f}")

    # ---- [2] 分量级·档位重标定（论文承诺的那一步）----
    p_sidak = 1.0 - (1.0 - q_star) ** (1.0 / 3.0)
    mapsA = {k: band_calib_map(cal_vals[k], q_star)[0] for k in KEYS}
    mapsB = {k: band_calib_map(cal_vals[k], p_sidak)[0] for k in KEYS}
    fA, fB = synth_sum(mapsA), synth_sum(mapsB)
    print(f"\n  [2] 改法 A/B：分量级「档位占用率」重标定（q*={q_star:.4f}, "
          f"Sidak p_3={p_sidak:.4f}）")
    print(f"   {'取法':<30}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}{'高+极高':>11}")
    for nm, fn in (("A 朴素等值 p=q*", fA), ("B Sidak 修正 p_3", fB)):
        print(f"   {nm:<30}{fmt_occ(occ_fixed(norm, fn))}")
    print(f"   {'（未校正三分量作参照）':<30}{fmt_occ(o3)}")
    print("   → 6 组占用率无一下降，放大被推得更高：档位占用率不是成因。")

    # ---- [3] 放大机制核验 ----
    c3 = np.array([comp3(r) for r in cal_mix], float)
    cm = np.array([compmax(r) for r in cal_mix], float)
    c2 = np.array([comp2(r) for r in cal_mix], float)
    print("\n  [3] 放大机制核验（标定流正常流）")
    print(f"   合成分 >= 最大分量的比例            = "
          f"{float(np.mean(c3 >= cm - 1e-12)):.4f}")
    print(f"   三个分量皆 < 0.8 而合成分 >= 0.8 的比例 = "
          f"{float(np.mean((cm < 0.8) & (c3 >= 0.8))):.4f}")
    print(f"   合成分均值：二分量 {c2.mean():.4f}  三分量 {c3.mean():.4f}  "
          f"取最大 {cm.mean():.4f}")
    print("   → R >= max_k rho_k，且分量各取中档（各 0.6）即得 R=0.936>0.8；")
    print("     放大由分量取值量级驱动，故任何只约束「各分量档位占用率」的改法都够不着它。")

    # ---- [4] 改法 C：分量级分布重标定（PIT）----
    pit = {k: midrank_pit(cal_vals[k]) for k in KEYS}
    mapsP = {k: (lambda x, kk=k: pit[kk](x)) for k in KEYS}
    fP = synth_sum(mapsP)
    oP = occ_fixed(norm, fP)
    print("\n  [4] 改法 C：分量级「分布」重标定（mid-rank PIT 到同一参考分布）")
    print(f"   {'PIT 后三分量概率和':<30}{fmt_occ(oP)}")
    for k in KEYS:
        u = np.array([pit[k](x) for x in cal_vals[k]], float)
        ou = occ_fixed(cal_mix, lambda r, kk=k: pit[kk](getattr(r, kk)))
        print(f"     {k:<12} 归一化后均值={u.mean():.4f}  高+极高占用率="
              f"{ou['高'] + ou['极高']:.4f}")
    print("   → 分量分布高度偏斜：rho_trans 在正常流上 97.9% 落在低档、近零值占多数。")
    print("     PIT 把这段近零质量抬到参考分布的 0.5 量级，等于给每个分量注入虚假量级，")
    print("     故 C 比未校正更差。把分量分布强行对齐是错误的一步。")

    # ---- [5] 改法 D：合成级分档阈值重标定 ----
    th_uncal = (0.25, 0.5, 0.8)
    th3_raw = q_thresholds([comp3(r) for r in cal_mix], props)
    th2_raw = q_thresholds([comp2(r) for r in cal_mix], props)
    th3_pure = q_thresholds([comp3(r) for r in cal_pure], props)
    print("\n  [5] 改法 D：合成级「分档阈值」重标定（阈值改为该 n 的正常流分位数）")
    print(f"   {'口径':<34}{'中档下界':>10}{'高档下界':>10}{'极高下界':>10}")
    for nm, th in (("未校正·固定阈值", th_uncal),
                   ("未校正·固定阈值（三分量同）", th_uncal),
                   ("重标定·二分量（标定流含攻击）", th2_raw),
                   ("重标定·三分量（标定流含攻击）", th3_raw),
                   ("重标定·三分量（标定流纯正常）", th3_pure)):
        print(f"   {nm:<34}" + "".join(f"{v:>10.4f}" for v in th))

    oD_in = occ_thresh(cal_mix, comp3, th3_raw)
    oD_out = occ_thresh(norm, comp3, th3_raw)
    oD_pure_out = occ_thresh(norm, comp3, th3_pure)
    print(f"\n   {'口径':<34}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}{'高+极高':>11}")
    print(f"   {'目标（标定流二分量）':<34}" + "".join(f"{p:>8.4f}" for p in props)
          + f"{q_star:>11.4f}")
    print(f"   {'D 三分量·标定流内（机制核验）':<34}{fmt_occ(oD_in)}")
    print(f"   {'D 三分量·检测流（留出）':<34}{fmt_occ(oD_out)}")
    print(f"   {'D 三分量·纯正常流标定后（留出）':<34}{fmt_occ(oD_pure_out)}")
    print(f"   {'未校正三分量（固定阈值）参照':<34}{fmt_occ(o3)}")
    print("   → 阈值改由正常流分位数定义后，档位占用率与分量数解耦：标定流内命中目标；")
    print("     留出到检测流仍有残差，源于攻击请求抬高了同一主体后续正常请求的 rho_trans，")
    print("     使检测流的正常流构成与纯正常流不同。故标定流工况须与运行工况同分布。")

    # ---- [6] 工况一致性 ----
    o_pure3 = occ_fixed(pure, comp3)
    print("\n  [6] 工况一致性核验（同一随机源、仅切换是否注入攻击）")
    print(f"   {'正常流来源':<30}{'低':>8}{'中':>8}{'高':>8}{'极高':>8}{'高+极高':>11}")
    print(f"   {'seed+3 含攻击（检测流正常流）':<30}{fmt_occ(o3)}")
    print(f"   {'seed+3 纯正常流':<30}{fmt_occ(o_pure3)}")
    print(f"   → 注入攻击把同一随机源下正常流的高+极高占用率由 "
          f"{o_pure3['高'] + o_pure3['极高']:.4f} 抬到 {o3['高'] + o3['极高']:.4f}，")
    print("     说明攻击的时序污染会沿状态序列传导到正常请求，属必须交代的口径效应。")

    # ---- [7] 判别力对照 ----
    print("\n  [7] 判别力对照（AUC，检测流留出）")
    print(f"   {'口径':<32}" + "".join(f"{t:>8}" for t in types))
    rows_csv: List[dict] = []

    def add(nm: str, fn, o: Dict[str, float], note: str) -> None:
        cells = [auc([fn(r) for r in att if r.label == t], [fn(r) for r in norm])
                 for t in types]
        print(f"   {nm:<32}" + "".join(f"{v:>8.3f}" for v in cells))
        rows_csv.append({"重标定口径": nm,
                         **{f"占比_{b}": round(o[b], 4) for b in BANDS},
                         "高加极高": round(o["高"] + o["极高"], 4),
                         **{f"AUC_{t}": round(v, 4) for t, v in zip(types, cells)},
                         "说明": note})

    add("二分量概率和（现行基线）", comp2, o2, "论文现行口径")
    add("三分量概率和（未校正）", comp3, o3, "档位放大未处理")
    add("三分量取最大（未校正）", compmax, om, "改用 max 聚合")
    add("A 朴素等值 p=q*（无效）", fA, occ_fixed(norm, fA), "分量级档位校正")
    add("B Sidak 修正 p_3（无效）", fB, occ_fixed(norm, fB), "分量级档位校正")
    add("C PIT 分布重标定（无效）", fP, oP, "分量级分布校正")
    add("D 合成级阈值重标定（有效）", comp3, oD_out, "单调变换，AUC 与未校正逐值相同")

    # ---- [8] 结论 ----
    print("\n  [8] 结论")
    print(f"   未校正：三分量高+极高占用率 {o3['高'] + o3['极高']:.4f}，相对二分量 "
          f"{p2_hi:.4f} 放大 {(o3['高'] + o3['极高']) / p2_hi - 1.0:+.1%}")
    for i, lab in ((3, "A 朴素等值"), (4, "B Sidak 修正"), (5, "C PIT 分布重标定")):
        v = rows_csv[i]["高加极高"]
        print(f"   {lab}：{v:.4f}（相对基线 {v / p2_hi - 1.0:+.1%}）—— 无效或反效")
    v = rows_csv[6]["高加极高"]
    print(f"   D 合成级阈值重标定：{v:.4f}（相对设计目标 {v / q_star - 1.0:+.1%}）；"
          "AUC 与未校正逐值相同 —— 有效且零判别力代价")

    out = os.path.join(RES, "E5b_合成重标定对照.csv")
    cols: List[str] = []
    for r in rows_csv:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_csv:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n  已写入 {out}")


if __name__ == "__main__":
    main()
