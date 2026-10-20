# -*- coding: utf-8 -*-
"""E15：完备性缺口与锚定失败实验的离线汇总与校验。

数据来源：chain/bench/results_e15/<场景>.csv，由 AnchorFault.java 写出。
  场景串：base 对照 | drop:<周期> | fail:<周期,...> | tamper:<周期>:<删几条>

三张表：
  E15_锚定失败窗口.csv   锚定失败后的重排队开销与「已执行、未确认」窗口
  E15_完备性缺口.csv     上链前删改 / 整批丢弃 能否被链上根发现
  E15_批级成本.csv       正常批、并入重排队记录的批、失败交易各自的 gas

叶集合与 Merkle 构造必须与 AnchorFault.java 逐字一致：
  叶 = SHA256("rec|<Δt>|<周期>|<序号>")；近似二叉树，奇数节点与自身配对。
"""
import csv
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "chain", "bench", "results_e15")
OUT = os.path.join(HERE, "results")

DT_MS = 1000     # 与 run_e15.sh 一致
K = 200          # 批内记录数 k = λ·Δt


def h(b):
    return hashlib.sha256(b).digest()


def leaf(dt, p, i):
    return h(("rec|%d|%d|%d" % (dt, p, i)).encode("utf-8"))


def leaf_alt(dt, p, i):
    """被改写的叶：与 AnchorFault.java 的 alter 分支逐字一致。"""
    return h(("rec|%d|%d|%d|altered" % (dt, p, i)).encode("utf-8"))


def merkle(leaves):
    cur = list(leaves)
    if not cur:
        return bytes(32)
    while len(cur) > 1:
        m = (len(cur) + 1) // 2
        nxt = []
        for i in range(m):
            j = 2 * i + 1 if 2 * i + 1 < len(cur) else 2 * i
            nxt.append(h(cur[2 * i] + cur[j]))
        cur = nxt
    return cur[0]


def hexs(b):
    return "0x" + b.hex()


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(-(-len(xs) * q // 1)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def parse_scenario(sc):
    """返回 (mode, targets, arg)。arg 对 tamper/miss 是条数 m，对 alter 是叶序号 i。"""
    if ":" not in sc:
        return sc, set(), 0
    mode, rest = sc.split(":", 1)
    if mode in ("tamper", "miss", "alter"):
        p, m = rest.split(":")
        return mode, {int(p)}, int(m)
    return mode, {int(x) for x in re.split(r"[;,]", rest) if x.strip()}, 0


def edit_desc(mode, arg):
    """把改动场景翻译成人话，用于表内结论列。"""
    if mode == "tamper":
        return "自尾部删除 %d 条记录" % arg
    if mode == "miss":
        return "自中部漏记 %d 条记录（叶序号空间出现空洞）" % arg
    if mode == "alter":
        return "改写第 %d 条记录内容（条数不变）" % arg
    return mode


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    if not os.path.isdir(SRC):
        sys.exit("缺少数据目录：%s" % SRC)
    files = sorted(n for n in os.listdir(SRC) if n.endswith(".csv"))
    if not files:
        sys.exit("results_e15 下没有 CSV")

    win_rows, gap_rows, cost_rows = [], [], []

    for fn in files:
        rows = load(os.path.join(SRC, fn))
        if not rows:
            continue
        sc = rows[0]["场景"]
        mode, targets, edit_arg = parse_scenario(sc)

        # ── 逐周期重建叶集合，与链上根对账 ────────────────────────────
        # 两套并行重建：
        #   *_full  「应有」口径——按规程，本周期应产生 k 条记录，失败批次应被并入
        #   *_act   「实际」口径——本次真实提交的叶集合（改动场景下与应有集合不同）
        # 链上根一律取 AnchorFault.java 通过合约 view 函数 getBatch 读回的字段；
        # 「提交根」只作为驱动自证，不再充当判定依据。
        pend_full, pend_act = [], []
        for r in rows:
            p = int(r["周期"])
            own_full = [leaf(DT_MS, p, i) for i in range(K)]
            edited = (mode in ("tamper", "miss", "alter") and p in targets)
            own_act = list(own_full)
            if edited and mode == "tamper":
                # 自尾部删掉 m 条（与 Java 的 own.remove(own.size()-1) 一致）
                own_act = own_full[:len(own_full) - edit_arg]
            elif edited and mode == "miss":
                # 自中部连续漏掉 m 条，其余叶保留原序号（与 Java 的反复 remove(from) 一致）
                frm = max(0, len(own_full) // 2 - edit_arg // 2)
                own_act = own_full[:frm] + own_full[frm + edit_arg:]
            elif edited and mode == "alter":
                # 改写第 i 条内容，条数与序号均不变
                if 0 <= edit_arg < len(own_act):
                    own_act[edit_arg] = leaf_alt(DT_MS, p, edit_arg)

            lv_full = pend_full + own_full
            lv_act = pend_act + own_act
            pend_full, pend_act = [], []
            status = r["回执状态"]
            submitted = (r.get("merkleRoot") or "").strip()
            onchain = (r.get("链上根") or "").strip()
            live = bool(onchain)

            if status == "DROPPED":
                # 整批丢弃：本批叶集合（含此前待重排队的）直接消失，链上无任何记录
                gap_rows.append(dict(
                    场景=sc, 周期=p, 应有叶数=len(lv_full), 本批叶数=int(r["本批叶数"]),
                    根是否在链上="否", 链上根="(未上链)",
                    按应有叶集合重算根=hexs(merkle(lv_full)),
                    与应有集合一致="—", 与提交集合一致="—",
                    结论="整批丢弃：链上无根也无该窗口的批记录，改动无法由根本身发现；"
                         "相邻批元数据留下 tStart 空洞，被丢弃的记录不可恢复，且重排队也不会补回",
                ))
                continue

            ref = onchain if live else submitted
            eq_full = (ref.lower() == hexs(merkle(lv_full)).lower())
            eq_act = (ref.lower() == hexs(merkle(lv_act)).lower())

            if edited:
                concl = ("上链前%s：链上根与「实际提交的叶集合」%s、"
                         "与「应有的完整叶集合」%s ⇒ 按应有集合重算可发现不一致；"
                         "但只持有提交方自己那份叶集合的验证方无法发现该改动"
                         % (edit_desc(mode, edit_arg), "一致" if eq_act else "不一致",
                            "一致" if eq_full else "不一致"))
            elif mode == "fail" and status != "0x0":
                concl = ("锚定失败（回执 %s）：本批根覆盖「此前积压 %s 条 ＋ 本周期 %d 条」共 %d 叶，"
                         "未上链，已并入下一批重排队"
                         % (status, r["并入记录数"], K, len(lv_full)))
            elif mode == "fail" and int(r["并入记录数"]) > 0:
                concl = ("重排队后的成功批：本批根覆盖「此前积压 %s 条 ＋ 本周期 %d 条」共 %d 叶，"
                         "与链上根一致" % (r["并入记录数"], K, len(lv_full)))
            else:
                concl = "正常批：链上根与叶集合自洽"

            gap_rows.append(dict(
                场景=sc, 周期=p, 应有叶数=len(lv_full), 本批叶数=int(r["本批叶数"]),
                根是否在链上=("是" if live else "否"),
                链上根=(onchain if live else "(未上链)"),
                按应有叶集合重算根=hexs(merkle(lv_full)),
                与应有集合一致=("是" if eq_full else "否"),
                与提交集合一致=("是" if eq_act else "否"),
                结论=concl,
            ))

            # 重排队状态推进：与 AnchorFault.java 逐字一致
            if status == "0x0":
                pend_full, pend_act = [], []
            else:
                pend_full, pend_act = lv_full, lv_act

            # ── 失败窗口与成本 ────────────────────────────────────────
            if status == "0x0":
                w = int(r["未确认窗口ms"])
                win_rows.append(dict(场景=sc, 周期=p, 结果="成功", 并入记录数=int(r["并入记录数"]),
                                     本批叶数=int(r["本批叶数"]),
                                     确认时延ms=int(r["确认时延ms"]), 未确认窗口ms=w,
                                     gasUsed=int(r["gasUsed"], 16) if r["gasUsed"] else 0,
                                     区块号=r["区块号"], 链上批号=r["链上批号"]))
                cost_rows.append(dict(场景=sc, 周期=p, 结果="成功", 改动=edited,
                                      并入记录数=int(r["并入记录数"]), 本批叶数=int(r["本批叶数"]),
                                      gasUsed=int(r["gasUsed"], 16) if r["gasUsed"] else 0,
                                      确认时延ms=int(r["确认时延ms"])))
            else:
                win_rows.append(dict(场景=sc, 周期=p, 结果="失败(" + status + ")",
                                     并入记录数=int(r["并入记录数"]), 确认时延ms=int(r["确认时延ms"]),
                                     未确认窗口ms=-1,
                                     gasUsed=int(r["gasUsed"], 16) if r["gasUsed"] else 0,
                                     区块号=r["区块号"], 链上批号=r["链上批号"]))
                cost_rows.append(dict(场景=sc, 周期=p, 结果="失败", 并入记录数=0, 本批叶数=int(r["本批叶数"]),
                                      gasUsed=int(r["gasUsed"], 16) if r["gasUsed"] else 0,
                                      确认时延ms=int(r["确认时延ms"])))

    os.makedirs(OUT, exist_ok=True)

    def dump(name, rows, cols):
        fp = os.path.join(OUT, name)
        with open(fp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        print("[写入] %s (%d 行)" % (fp, len(rows)))

    # ── 表 1：锚定失败窗口（按场景聚合） ────────────────────────────
    #
    # 「未确认窗口」的定义要收紧，否则非失败场景的最大值会被两处测量毛刺污染
    # （每个场景的第 0 个周期 ~470–490 ms 属冷启动；alter 场景 p=27 有一次
    # 2523 ms 的链抖动），而它们与锚定失败毫无关系。
    # 这里只统计**带重排队积压**的成功批（并入记录数 > 0）：
    #   窗口 = 该批 sched + 确认时延 − 最早失败批的 sched = n_f·Δt + t_c
    # 正常批另用 P50 报告，量级在十几毫秒，远低于 Δt + t_c。
    agg = []
    for sc in sorted({r["场景"] for r in win_rows}):
        rs = sorted([r for r in win_rows if r["场景"] == sc], key=lambda x: x["周期"])
        fails = [r for r in rs if r["结果"].startswith("失败")]
        okc = [r for r in rs if r["结果"] == "成功"]
        norm = [r for r in okc if r["并入记录数"] == 0]
        carried = [r for r in okc if r["并入记录数"] > 0]
        tc = pct([r["确认时延ms"] for r in norm], 0.50) or 0
        nc = max([r["并入记录数"] for r in carried] or [0])
        nf = nc // K                      # 连续失败笔数 = 积压记录数 / 批内记录数
        if carried:
            c = max(carried, key=lambda x: x["未确认窗口ms"])
            win = c["未确认窗口ms"]
            pred = nf * DT_MS + c["确认时延ms"]      # 与窗口定义同源的闭式预测
            carr_lat = c["确认时延ms"]
            carr_leaves = c["本批叶数"]
        else:
            win = pred = carr_lat = None
            carr_leaves = None
        agg.append(dict(
            场景=sc, 周期数=len(rs), 失败笔数=len(fails),
            失败交易平均gas=(sum(r["gasUsed"] for r in fails) / len(fails)) if fails else 0,
            正常批确认时延P50_ms=tc, 正常批窗口P50_ms=pct([r["未确认窗口ms"] for r in norm], 0.50),
            重排队记录数=nc, 重排队批叶数=(carr_leaves if carried else ""),
            重排队批确认时延_ms=(carr_lat if carried else ""),
            最早失败批未确认窗口_ms=(win if carried else ""),
            预测窗口_nfΔt加t_c_ms=(pred if carried else ""),
            Δt_and_tc上界_ms=DT_MS + tc,
            超出上界_ms=((win - (DT_MS + tc)) if carried else ""),
        ))
        if carried and pred != win:
            print("  [注意] %s 窗口自洽校验失败：实测 %s ≠ 预测 %s" % (sc, win, pred))
    dump("E15_锚定失败窗口.csv", agg,
         ["场景", "周期数", "失败笔数", "失败交易平均gas", "正常批确认时延P50_ms", "正常批窗口P50_ms",
          "重排队记录数", "重排队批叶数", "重排队批确认时延_ms", "最早失败批未确认窗口_ms",
          "预测窗口_nfΔt加t_c_ms", "Δt_and_tc上界_ms", "超出上界_ms"])

    # ── 表 2：完备性缺口（只留异常与对照代表行） ────────────────────
    # 正常批不必逐场景罗列，只留 base 的两行做对照；其余场景只留被改动/被丢弃
    # 的那一周期，以及锚定失败场景里失败与重排队成功的那几行。
    EDIT_MODES = ("drop", "tamper", "miss", "alter")
    keep = []
    for r in gap_rows:
        sc_mode = r["场景"].split(":")[0]
        if r["场景"] == "base":
            if r["周期"] in (0, 1):
                keep.append(r)
        elif sc_mode in EDIT_MODES:
            if not r["结论"].startswith("正常批"):
                keep.append(r)
        elif r["结论"].startswith("重排队后的成功批") or r["结论"].startswith("锚定失败（"):
            keep.append(r)
    dump("E15_完备性缺口.csv", keep,
         ["场景", "周期", "应有叶数", "本批叶数", "根是否在链上", "链上根",
          "按应有叶集合重算根", "与应有集合一致", "与提交集合一致", "结论"])

    # ── 表 3：批级成本 ──────────────────────────────────────────────
    # 每场景第 0 个周期（新合约上的首笔锚定）gas 明显低于稳态，单独列一类，
    # 不混进稳态均值；它与同场景的时延冷启动是同一件事。
    cagg = []
    for sc in sorted({r["场景"] for r in cost_rows}):
        rs = [r for r in cost_rows if r["场景"] == sc]
        steady = [r for r in rs if r["周期"] != 0]
        for label, sel in (("首批", [r for r in rs if r["周期"] == 0]),
                           ("正常批", [r for r in steady if r["结果"] == "成功"
                                       and r["并入记录数"] == 0 and not r.get("改动")]),
                           ("改动批", [r for r in steady if r["结果"] == "成功" and r.get("改动")]),
                           ("重排队并入批", [r for r in steady if r["结果"] == "成功"
                                             and r["并入记录数"] > 0]),
                           ("失败交易", [r for r in steady if r["结果"] == "失败"])):
            if not sel:
                continue
            cagg.append(dict(
                场景=sc, 类别=label, 笔数=len(sel),
                平均gas=sum(r["gasUsed"] for r in sel) / len(sel),
                平均叶数=sum(r["本批叶数"] for r in sel) / len(sel),
                单叶gas=(sum(r["gasUsed"] for r in sel) / sum(r["本批叶数"] for r in sel))
                       if sum(r["本批叶数"] for r in sel) else 0,
                确认时延P50_ms=pct([r["确认时延ms"] for r in sel], 0.50),
            ))
    dump("E15_批级成本.csv", cagg,
         ["场景", "类别", "笔数", "平均gas", "平均叶数", "单叶gas", "确认时延P50_ms"])

    # ── 终端摘要 ────────────────────────────────────────────────────
    print("\n== 完备性缺口 ==")
    for r in keep:
        print("  %-16s 周期=%-3s 本批叶=%-4s 在链上=%s 与应有/提交集合一致=%s/%s  %s"
              % (r["场景"], r["周期"], r["本批叶数"], r["根是否在链上"],
                 r["与应有集合一致"], r["与提交集合一致"], r["结论"][:64]))
    print("\n== 锚定失败窗口 ==")
    for r in agg:
        print("  %-16s 失败=%-2s 重排队记录=%-5s 窗口=%-6s 预测=%-6s Δt+t_c=%-6s 超出=%s"
              % (r["场景"], r["失败笔数"], r["重排队记录数"],
                 r["最早失败批未确认窗口_ms"] if r["最早失败批未确认窗口_ms"] != "" else "—",
                 r["预测窗口_nfΔt加t_c_ms"] if r["预测窗口_nfΔt加t_c_ms"] != "" else "—",
                 r["Δt_and_tc上界_ms"],
                 r["超出上界_ms"] if r["超出上界_ms"] != "" else "—"))


if __name__ == "__main__":
    main()
