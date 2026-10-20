# -*- coding: utf-8 -*-
"""E13：跨机部署下的往返时延标定。

测试床：四台 KVM 虚机（2 vCPU / 4 GiB / Ubuntu 22.04.5，同一物理宿主机），
各运行一个 FISCO BCOS 2.11 PBFT 共识节点（build_chain.sh -l "<ip>:1"×4，
group.1.genesis 含 4 个 node id，epoch_sealer_num=4）。压测客户端与 node0
同机（与 E9 口径一致），走本机回环访问 channel，共识与状态同步在四台之间
的真实链路上完成。

自变量：节点间单程延时 d，由 netem 在四台网卡 ens3 出口同时施加，
合成往返 RTT = 2d + 基线（实测 0.30 ms）。每档以 ICMP 实测值标注。

读数口径与 E9 完全一致（AnchorLoop，λ=200，Δt 九档阶梯，确认时延为客户端
提交到收到回执的墙钟毫秒）。阴性对照：每批 gas 只应由批内记录数 k 决定，
与 RTT 无关；若随 RTT 变化即说明测量串线。

数据来源：chain/bench/results_e13/loop_<tag>.csv（逐笔明细）
输出：experiments/results/E13_跨机时延.csv     每档每 Δt 的聚合读数
      experiments/results/E13_RTT敏感性.csv    每档在稳态 Δt 档上的代表性读数
      experiments/results/E13_RTT实测.csv      各档 netem 设定与 ICMP 实测
"""
import collections
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "chain", "bench", "results_e13")
OUT = os.path.join(HERE, "results")

# tag -> (设定单程 ms, 扰动描述, ICMP 实测 (min, avg, max) ms, 备注)
# 实测值取自每档开跑前 `ping -c 30 -i 0.2`（.82 -> .81）；
# rtt0 为无注入档，其 RTT 在同一洁净状态下另测复核。
TAGS = [
    # rtt0 为无注入档，其 ICMP 实测取撤除 netem 后的复核值：rtt200 档收尾后
    # 测得 0.310 ms（0.239–0.405），loss1 档收尾后再测 0.329 ms（0.260–0.374）。
    ("rtt0",   0.0,   "",                 (0.239, 0.310, 0.405), "无注入基线；两次撤净复核 0.310 / 0.329 ms"),
    ("rtt1",   0.5,   "delay 0.5ms",      (1.301, 1.473, 1.672), ""),
    ("rtt5",   2.5,   "delay 2.5ms",      (5.292, 5.474, 5.682), ""),
    ("rtt20",  10.0,  "delay 10ms",       (20.306, 20.521, 20.662), ""),
    ("rtt50",  25.0,  "delay 25ms",       (50.286, 50.495, 50.690), ""),
    ("rtt200", 100.0, "delay 100ms",      (200.244, 200.458, 200.670), ""),
    ("jit5",   10.0,  "delay 10ms 5ms",   (14.468, 21.242, 27.276), "固定 20 ms 往返上叠加 ±5 ms 抖动（mdev 3.112 ms）"),
    ("loss01", 10.0,  "delay 10ms loss 0.1%", (20.293, 20.477, 20.674), "固定 20 ms 往返上叠加 0.1% 丢包"),
    ("loss1",  10.0,  "delay 10ms loss 1%",   (20.139, 20.493, 20.670), "固定 20 ms 往返上叠加 1% 丢包"),
]

# E9（单机四节点）对照值，用于回归检查
E9 = {
    100: 275, 200: 252, 500: 274, 1000: 11,
    2000: 9, 5000: 8, 10000: 8, 20000: 8, 60000: 9,
}
STEADY = 1000   # Δt >= 1000 ms 为稳态档（交易到达即被随后出块收录）


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pct(xs, q):
    """最近秩分位数，与 chain/bench/analyze.py 同法。"""
    if not xs:
        return None
    xs = sorted(xs)
    k = int(-(-len(xs) * q // 1)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def agg(rows, dt_ms):
    ok = [r for r in rows if r["状态"].strip().lower().startswith("0x0")]
    lat = [int(r["确认时延ms"]) for r in ok if int(r["确认时延ms"]) >= 0]
    build = [int(r["Merkle建根ms"]) for r in rows]
    gas = [int(r["gasUsed"], 16) for r in ok if r["gasUsed"]]
    blocks = sorted({r["区块号"] for r in ok if r["区块号"]})
    cyc = sorted({int(r["周期序号"]) for r in rows})
    ks = [int(r["批内记录数k"]) for r in rows]
    return dict(
        dt=dt_ms, cyc=len(cyc), n=len(rows), succ=len(ok), k=max(ks) if ks else 0,
        build_p50=pct(build, 0.50), build_max=max(build) if build else 0,
        p50=pct(lat, 0.50), p95=pct(lat, 0.95), mx=max(lat) if lat else None,
        blocks=len(blocks),
        gas=(sum(gas) / len(gas)) if gas else 0,
        rate=((sum(gas) / len(gas)) / (dt_ms / 1000.0)) if gas else 0,
    )


def main():
    if not os.path.isdir(SRC):
        sys.exit("缺少数据目录：%s" % SRC)
    detail, steady, meta = [], [], []
    for tag, d, spec, ping, note in TAGS:
        fn = os.path.join(SRC, "loop_%s.csv" % tag)
        if not os.path.exists(fn):
            print("[跳过] %s 尚无数据" % tag)
            continue
        rows = load(fn)
        g = collections.defaultdict(list)
        for r in rows:
            g[int(r["Δt毫秒"])].append(r)
        for dt in sorted(g):
            a = agg(g[dt], dt)
            a_tag = dict(tag=tag, d=d, spec=spec, rtt_avg=(ping[1] if ping else None))
            detail.append(dict(a_tag, **a))
            if dt == STEADY:
                steady.append(dict(a_tag, **a))
        ok = [r for r in rows if r["状态"].strip().lower().startswith("0x0")]
        rtt_ok = ping[1] if ping else None
        meta.append(dict(tag=tag, d=d, spec=spec or "(无注入)",
                         rtt_min=(ping[0] if ping else None),
                         rtt_avg=rtt_ok, rtt_max=(ping[2] if ping else None),
                         epochs=len({int(r["周期序号"]) for r in rows}),
                         n=len(rows), succ=len(ok), note=note))

    os.makedirs(OUT, exist_ok=True)

    def dump(name, rows, cols):
        p = os.path.join(OUT, name)
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        print("[写入] %s (%d 行)" % (p, len(rows)))

    dump("E13_RTT实测.csv", meta,
         ["tag", "d", "spec", "rtt_min", "rtt_avg", "rtt_max",
          "epochs", "n", "succ", "note"])
    dump("E13_跨机时延.csv", detail,
         ["tag", "d", "rtt_avg", "dt", "cyc", "n", "succ", "k", "build_p50",
          "build_max", "p50", "p95", "mx", "blocks", "gas", "rate"])
    dump("E13_RTT敏感性.csv", steady,
         ["tag", "d", "rtt_avg", "dt", "cyc", "succ", "k", "build_p50",
          "build_max", "p50", "p95", "mx", "blocks", "gas", "rate"])

    # 终端摘要
    print("\n== 各档在 Δt=%d ms（稳态）上的读数 ==" % STEADY)
    print("tag      RTT(ms)  P50  P95  max   gas      占用区块  对照E9 P50=%d"
          % E9[STEADY])
    for r in steady:
        print("%-8s %-8s %-4s %-4s %-5s %-9.0f %-9d %s"
              % (r["tag"], r["rtt_avg"] if r["rtt_avg"] else "-", r["p50"],
                 r["p95"], r["mx"], r["gas"], r["blocks"],
                 "OK" if abs((r["p50"] or 0) - E9[STEADY]) <= 8 else "偏离"))

    # 「确认时延 — 往返时延」的线性拟合。
    #
    # 分两段报：rtt1 至 rtt50 落在最小出块间隔 500 ms 仍为约束的区间内，
    # 拟合斜率即正文引用的「共识两轮广播加转发回执」的实测倍率；rtt200 档
    # 的共识轮次本身已超过最小出块间隔，曲线在此处折向上方，故单列对照
    # （合并拟合会把这一拐点压成一条斜率失真的直线）。
    # 抖动/丢包档共用同一名义往返时延，不进拟合，单独列出。
    def linfit(tags):
        ps = [(r["rtt_avg"], r["p50"]) for r in steady
              if r["tag"] in tags and r["rtt_avg"] and r["p50"] is not None]
        n = len(ps)
        mx = sum(x for x, _ in ps) / n
        my = sum(y for _, y in ps) / n
        sxy = sum((x - mx) * (y - my) for x, y in ps)
        sxx = sum((x - mx) ** 2 for x, _ in ps)
        syy = sum((y - my) ** 2 for _, y in ps)
        kk = sxy / sxx
        bb = my - kk * mx
        rr = (sxy ** 2) / (sxx * syy) if syy else float("nan")
        return kk, bb, rr, n, ps

    for label, tg in (("最小出块间隔为约束段", ["rtt1", "rtt5", "rtt20", "rtt50"]),
                      ("全档（含越过最小出块间隔的 rtt200）",
                       ["rtt1", "rtt5", "rtt20", "rtt50", "rtt200"])):
        kk, bb, rr, nn, ps = linfit(tg)
        print("\n线性拟合 [%s]（%s）：P50 = %.2f + %.3f × RTT , R² = %.5f , n = %d"
              % (label, ", ".join(tg), bb, kk, rr, nn))
        print("  逐点残差：", " ".join("%s:%+.1f" % (t, y - (bb + kk * x))
                                   for (x, y), t in zip(ps, tg)))

    print("\n各档 P50/RTT 倍率（含无注入基线）：")
    for r in steady:
        if r["rtt_avg"] and r["p50"]:
            print("  %-8s RTT=%-8s P50=%-5s 倍率=%.2f"
                  % (r["tag"], r["rtt_avg"], r["p50"], r["p50"] / r["rtt_avg"]))

    # 抖动与丢包三档（与 rtt20 同名往返时延，不参与拟合）
    print("\n== 抖动/丢包档（名义往返同为 20 ms）==")
    for r in steady:
        if r["tag"] in ("jit5", "loss01", "loss1"):
            print("  %-8s RTT=%-8s P50=%-4s P95=%-4s max=%-5s gas=%.0f"
                  % (r["tag"], r["rtt_avg"], r["p50"], r["p95"], r["mx"],
                     r["gas"]))

    # 阴性对照：每批 gas 只应由批内记录数 k 决定
    print("\n== 阴性对照：稳态档每批平均 gas（应只随 k 变）==")
    for r in steady:
        print("  %-8s k=%-6d gas=%.1f" % (r["tag"], r["k"], r["gas"]))
    gs = [r["gas"] for r in steady]
    if gs:
        print("  极差 = %.0f gas（%.3f%%）" % (max(gs) - min(gs),
                                            100.0 * (max(gs) - min(gs)) / max(gs)))


if __name__ == "__main__":
    main()
