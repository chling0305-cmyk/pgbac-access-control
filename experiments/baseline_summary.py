# -*- coding: utf-8 -*-
"""论文一「方案级基线对比」汇总脚本。

把三条架构变体放在同一组指标下并排对比，数值全部现场读 results/*.csv，
不在脚本里写死任何实验结果。三条变体（架构路线，非外部论文方案）：

- B1 集中式判定：纯链下策略引擎，无链上锚定（去锚定后判定时延 = PGB-AC 决策路径）。
- B2 逐条记录上链：每个请求写一条链上交易，等共识确认后放行（同步上链）。
- PGB-AC 批次锚定：链下判定 + 异步批次锚定（本文）。

四维指标：决策时延、判定容量、单记录链上成本、可审计性（定性）。

用法：
    python baseline_summary.py
产出：
    results/E10_方案级基线对比.csv
"""
from __future__ import annotations

import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def read(name: str):
    p = os.path.join(RES, name)
    with io.open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    # ---- 时延 / 容量：来自 E1（异步锚定 = PGB-AC；同步锚定 = 逐条上链的确认等待） ----
    e1 = read("E1_决策时延与吞吐.csv")
    lam = 200.0  # 取 λ=200 作代表点
    async_row = None
    sync_row = None
    for r in e1:
        if float(r["lambda"]) != lam:
            continue
        if "异步" in r["模式"]:
            async_row = r
        elif "同步" in r["模式"]:
            sync_row = r

    # ---- 单记录链上成本：来自 E8 路线成本对照 ----
    e8 = read("E8_路线成本对照.csv")
    per_record = None      # B3 逐条记录上链
    batch = None           # 本文批次锚定
    for r in e8:
        if str(r["路线"]).startswith("B3"):
            per_record = r
        elif "本文" in str(r["路线"]):
            batch = r

    # ---- 吞吐上界：来自 E8 ----
    tp = read("E8_吞吐上界.csv")
    b3_tps = None
    ours_tps = None
    for r in tp:
        if str(r["路线"]).startswith("B3"):
            b3_tps = r
        elif "本文" in str(r["路线"]):
            ours_tps = r

    # 集中式判定（B1）：决策时延 = 异步锚定（PGB-AC）的决策路径时延，因为锚定不进入关键路径；
    # 容量同 PGB-AC；链上成本 = 0（无链上）；可审计性 = 无。
    rows = [
        {
            "方案": "B1 集中式判定（无链上锚定）",
            "决策时延P50_us": "%.2f" % float(async_row["P50_us"]),
            "决策时延P99_us": "%.2f" % float(async_row["P99_us"]),
            "判定容量_req_s": "%.0f" % float(async_row["判定容量_req_s"]),
            "单记录链上成本_gas": "0",
            "链吞吐_TPS": "—",
            "可审计性": "无（篡改可静默）",
        },
        {
            "方案": "B2 逐条记录上链（同步上链）",
            "决策时延P50_us": "%.2f" % float(sync_row["P50_us"]),
            "决策时延P99_us": "%.2f" % float(sync_row["P99_us"]),
            "判定容量_req_s": "%.1f" % float(sync_row["判定容量_req_s"]),
            "单记录链上成本_gas": "%.2f" % float(per_record["单条记录平均gas"]),
            "链吞吐_TPS": "%.0f" % float(b3_tps["吞吐_TPS"]),
            "可审计性": "有（全量记录）",
        },
        {
            "方案": "PGB-AC 批次锚定（本文）",
            "决策时延P50_us": "%.2f" % float(async_row["P50_us"]),
            "决策时延P99_us": "%.2f" % float(async_row["P99_us"]),
            "判定容量_req_s": "%.0f" % float(async_row["判定容量_req_s"]),
            "单记录链上成本_gas": "%.2f" % float(batch["单条记录平均gas"]),
            "链吞吐_TPS": "%.0f" % float(ours_tps["吞吐_TPS"]),
            "可审计性": "有（批次锚点 + 逐条日志）",
        },
    ]

    out = os.path.join(RES, "E10_方案级基线对比.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print("  %-28s 时延P50=%9s μs 容量=%10s 单记录gas=%10s 吞吐=%6s 可审计=%s"
              % (r["方案"], r["决策时延P50_us"], r["判定容量_req_s"],
                 r["单记录链上成本_gas"], r["链吞吐_TPS"], r["可审计性"]))
    print("写出: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
