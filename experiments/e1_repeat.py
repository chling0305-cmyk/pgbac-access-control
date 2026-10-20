# -*- coding: utf-8 -*-
"""E1 的重复运行稳定性采集。

`run_experiments.py` 每次运行 E1 都会覆盖同名结果文件，因此单次运行无法回答
两个必要的问题：(1) 表 20 中「归一化 P99」的跨请求率差异，是否大于同一请求率在不同次
运行之间的差异？(2) 逐项绝对时延在多大程度上是机器状态（进程被调度到性能核还是
能效核）造成的，而不是被测对象造成的？

本机为混合架构 CPU，参考运算（固定 64 字节的 10 万次 SHA-256）实测在 0.49 与
1.21 μs 两档之间切换，且档位在一次运行内基本不变——即进程整体被降级调度。
因此绝对时延可以相差 2.4 倍，而「耗时 / 参考运算」是跨档可比的量。

本脚本把 E1 连跑 R 次：
  * 每次记录各请求率的 P50/P90/P99/P99.9、判定容量、单核利用率、参考运算与归一化值，
    汇总成 `results/E1_重复运行稳定性.csv`；
  * `results/E1_决策时延与吞吐.csv` 的每一格取**五次运行的中位数**，并把 P50、P99、
    容量与参考运算的跨运行最小／最大值一并列出；单轮读数只作明细，写进
    `results/E1_代表运行选择.csv`，不再充当正文读数；
  * 写出 `results/E1_跨运行分位汇总.csv`，逐请求率给出各指标在重复运行间的最小值、
    中位数与最大值。

用法：
    python e1_repeat.py          # 默认重复 5 次
    python e1_repeat.py 8        # 指定重复次数
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_experiments as R  # noqa: E402

OUT = os.path.join(R.RES, "E1_重复运行稳定性.csv")
OUT_SUM = os.path.join(R.RES, "E1_跨运行分位汇总.csv")
OUT_PICK = os.path.join(R.RES, "E1_代表运行选择.csv")

KEEP = ["P50_us", "P90_us", "P99_us", "P99.9_us", "最大_us",
        "判定容量_req_s", "单核利用率", "参考运算_us", "归一化P50", "归一化P99"]


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    R.banner(f"E1 重复运行稳定性（连跑 {reps} 次）")

    runs = []          # [(序号, rows, 墙钟秒)]
    for i in range(1, reps + 1):
        t0 = time.perf_counter()
        res = R.exp_e1(dict(R.CFG), quick=False)
        wall = time.perf_counter() - t0
        asyncs = [r for r in res if "异步" in r["模式"]]
        runs.append((i, res, round(wall, 2), asyncs))
        print(f"  [第 {i}/{reps} 次] " + "  ".join(
            f"λ={int(r['lambda']):>4}→P50 {r['P50_us']:.1f}/P99 {r['P99_us']:.1f} μs"
            f"(参考 {r['参考运算_us']:.2f})" for r in asyncs))

    # ---- 逐轮长表 ----
    long_rows = []
    for i, _res, wall, asyncs in runs:
        rec = {"运行序号": i, "耗时_s": wall}
        for r in asyncs:
            k = int(r["lambda"])
            for c in KEEP:
                rec[f"{c}_λ{k}"] = r[c]
        long_rows.append(rec)
    cols = list(long_rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(long_rows)

    # ---- 主表：五次运行逐格取中位数（第六轮审稿 R4.2） ----
    # 原先主表写的是「代表运行」（参考运算中位数最低的那一轮）。评审指出两件事：
    # 单轮读数容易被误读成总体统计，而且「快／慢」分组缺少可核查的边界。现改为
    # 主表逐格取五次运行的中位数，并把跨运行的最小／最大值一并列出；代表运行只
    # 作为明细保留在 E1_代表运行选择.csv，不再充当正文读数。
    MED_COLS = ["P50_us", "P90_us", "P99_us", "P99.9_us", "最大_us",
                "判定容量_req_s", "单核利用率", "参考运算_us", "归一化P50", "归一化P99"]

    def ref_med(asyncs) -> float:
        return statistics.median([r["参考运算_us"] for r in asyncs])

    pick = min(runs, key=lambda x: (ref_med(x[3]), x[2]))
    pick_tbl = []
    for i, _res, wall, asyncs in runs:
        pick_tbl.append({"运行序号": i, "耗时_s": wall,
                         "参考运算中位数_us": round(ref_med(asyncs), 3),
                         "是否其中一轮": "是" if i == pick[0] else ""})
    R.csv_write("E1_代表运行选择.csv", pick_tbl)

    keys = [(r["模式"], float(r["lambda"])) for r in runs[0][1]]
    med_rows = []
    for key in keys:
        cells = []
        for _i, res, _w, _a in runs:
            hit = [r for r in res if (r["模式"], float(r["lambda"])) == key]
            assert len(hit) == 1, (key, len(hit))
            cells.append(hit[0])
        base = dict(cells[0])
        for c in MED_COLS:
            if c not in cells[0]:        # 同步对照行没有归一化列
                continue
            v = [float(r[c]) for r in cells]
            base[c] = round(statistics.median(v), 6 if c == "单核利用率" else 3)
            if c in ("P50_us", "P99_us", "判定容量_req_s", "参考运算_us"):
                base[c + "_跨运行最小"] = round(min(v), 3)
                base[c + "_跨运行最大"] = round(max(v), 3)
        base["统计口径"] = "五次运行的中位数"
        med_rows.append(base)
    R.csv_write("E1_决策时延与吞吐.csv", med_rows)

    # ---- 跨运行汇总 ----
    lams = sorted({int(r["lambda"]) for r in runs[0][3]})
    sum_rows = []
    for k in lams:
        d = {"lambda": k}
        for c in ("P50_us", "P99_us", "P99.9_us", "参考运算_us", "归一化P99", "判定容量_req_s"):
            v = [next(r[c] for r in asyncs if int(r["lambda"]) == k)
                 for _i, _res, _w, asyncs in runs]
            d[f"{c}_最小"] = round(min(v), 3)
            d[f"{c}_中位"] = round(statistics.median(v), 3)
            d[f"{c}_最大"] = round(max(v), 3)
        sum_rows.append(d)
    R.csv_write("E1_跨运行分位汇总.csv", sum_rows)

    print("\n  各请求率在重复运行间的分布（参考运算与 P99）：")
    print(f"   {'λ':>6}{'参考min':>9}{'参考max':>9}{'P50min':>9}{'P50max':>9}"
          f"{'P99min':>9}{'P99max':>9}{'归一P99min':>12}{'归一P99max':>12}")
    for d in sum_rows:
        k = d["lambda"]
        print(f"   {k:>6}{d['参考运算_us_最小']:>9.3f}{d['参考运算_us_最大']:>9.3f}"
              f"{d['P50_us_最小']:>9.2f}{d['P50_us_最大']:>9.2f}"
              f"{d['P99_us_最小']:>9.2f}{d['P99_us_最大']:>9.2f}"
              f"{d['归一化P99_最小']:>12.3f}{d['归一化P99_最大']:>12.3f}")
    print(f"\n   主表＝五次运行中位数；参考运算中位数最低的一轮是第 {pick[0]} 次"
          f"（{ref_med(pick[3]):.3f} μs，墙钟 {pick[2]:.2f} s），仅作明细")
    print(f"   写入：{OUT}\n         {OUT_SUM}\n         {OUT_PICK}\n         E1_决策时延与吞吐.csv")


if __name__ == "__main__":
    main()
