# -*- coding: utf-8 -*-
"""从本轮提交的 CSV 里取出论文 7.4.5、8.2 与三张表所需的全部数值，并直接重写三张表的表体。

用法（仅报告）：
    python report_e5_numbers.py
"""
from __future__ import annotations

import csv
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    with open(os.path.join(RES, name), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def f3(v):
    return "%.3f" % v


ch = load("E5_通道检出率.csv")
uni = load("E5_综合检出率.csv")
scan = load("E5_阈值扫描.csv")
det = load("E5_处置分布.csv")
det2 = load("E5_处置分布_二分量对照.csv")
abl = load("E5_风险分量消融.csv")
lag = load("E5_检测滞后.csv")
agg = load("E5_聚合方式对照.csv")

print("=" * 78)
print("一、样本")
tot = 0
for r in det:
    n = int(num(r["样本"]))
    print("   %-6s n=%d" % (r["标签"], n))
    if r["标签"] != "正常":
        tot += n
print("   攻击合计 =", tot, " 正常 + 攻击 =", int(num(det[0]["样本"])) + tot)

print()
print("=" * 78)
print("二、表 26 检测通道表体（照抄 CSV，仅名称与精度整理）")
for r in ch:
    keys = ["TPR_A1", "TPR_A2", "TPR_A3", "TPR_A4", "TPR_A7"]
    cells = []
    for k in keys:
        v = num(r.get(k))
        cells.append("—" if v is None else f3(v))
    cells.append(f3(num(r["误报率"], 0.0)))
    print("| %s | %s | %s |" % (r["通道"], r["类型"], " | ".join(cells)))

print()
print("=" * 78)
print("三、表 27 风险分量作用域消融（AUC）")
for r in abl:
    print("| %s | %s | %s | %s | %s | %s | %s |" % (
        r["剧本"], r["样本数"], f3(num(r["AUC_ρ_struct"])), f3(num(r["AUC_ρ_trans"])),
        f3(num(r["AUC_ρ_behav"])), f3(num(r["AUC_二分量"])), f3(num(r["AUC_三分量"]))))

print()
print("=" * 78)
print("四、表 28 处置分布（三分量）")
for r in det:
    print("| %s | %s | %s | %s | %s |" % (
        r["标签"], r["样本"], f3(num(r["放行"])), f3(num(r["收紧或降级"])), f3(num(r["拒绝"]))))

print()
print("五、二分量对照")
for r in det2:
    print("   %s 放行=%s 收紧或降级=%s 拒绝=%s" % (
        r["标签"], f3(num(r["放行"])), f3(num(r["收紧或降级"])), f3(num(r["拒绝"]))))

print()
print("=" * 78)
print("六、阈值扫描工作点")
for score in sorted({r["评分"] for r in scan}):
    sub = [r for r in scan if r["评分"] == score]
    best = max(sub, key=lambda r: num(r["Youden"], -1))
    print("   %-8s θ=%.2f  Youden=%.4f  A1=%.4f A3=%.4f A7=%.4f A1∪A3=%s FPR=%.4f" % (
        score, num(best["阈值θ"]), num(best["Youden"]), num(best["TPR_A1"]),
        num(best["TPR_A3"]), num(best["TPR_A7"]),
        ("—" if num(best.get("TPR_A1UA3")) is None else "%.4f" % num(best["TPR_A1UA3"])),
        num(best["FPR"])))

print()
print("=" * 78)
print("七、A7 检测滞后分桶")
for r in lag:
    print("   %-16s n=%s  ρ_trans=%s  高敏占比=%s  检出率=%s" % (
        r["进度区间"], r["样本数"], r["ρ_trans均值"], r["高敏操作占比"], r["检出率"]))

print()
print("=" * 78)
print("八、聚合方式对照")
for r in agg:
    hi = num(r["占比_高"]) + num(r["占比_极高"])
    print("   %-12s 高+极高=%.4f  AUC A1=%.4f A2=%.4f A3=%.4f A4=%.4f A7=%.4f" % (
        r["聚合方式"], hi, num(r["AUC_A1"]), num(r["AUC_A2"]), num(r["AUC_A3"]),
        num(r["AUC_A4"]), num(r["AUC_A7"])))

print()
print("=" * 78)
print("九、合并检出率")
for r in uni:
    print("   %-8s %-8s %s" % (r["攻击类型"], r["样本数"], r["检出率"]))
