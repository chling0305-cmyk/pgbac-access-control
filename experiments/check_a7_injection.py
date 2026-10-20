# -*- coding: utf-8 -*-
"""A7 剧本注入的样本不变性检查。

A7 的注入判据必须是主体身份与时间的纯函数，且只改写原本会标为正常请求的样本，
否则会消耗随机数、扰动 A1 至 A4 与正常流的抽样结果，使 E5 的历史结论失效。
本脚本把 A7 关闭与开启两种情形跑在同一随机种子上，逐类核对样本数，
并按暴露窗口内的进度分桶核对 ρ_trans 的渐进性。

用法：
    python check_a7_injection.py

全部检查通过时退出码为 0；任一项不符即断言失败并给出差异。
"""
from __future__ import annotations

import random

import numpy as np

import pgbac_core as C

SEED_OFFSET_REQ = 3
LAM, DUR, RATE = 300.0, 400.0, 0.06
TYPES = ("A1", "A2", "A3", "A4")


def auc(pos, neg) -> float:
    """平均秩口径的 AUC，正确处理并列值（与 run_experiments 中同一实现）。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size, float)
    ranks[order] = np.arange(1, allv.size + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def main() -> int:
    cfg = dict(C.CFG)
    topo = C.build_topology(cfg, random.Random(cfg["seed"] + 11))

    # ---- 1) 关闭 A7：应在同一随机种子上复现既有的 A1 至 A4 样本 ----
    saved = C.A7_VICTIM_PCT
    C.A7_VICTIM_PCT = 0
    base = C.gen_requests(cfg, topo, random.Random(cfg["seed"] + SEED_OFFSET_REQ),
                          LAM, DUR, attack_rate=RATE)
    C.A7_VICTIM_PCT = saved
    n_base = {t: sum(1 for r in base if r.label == t) for t in TYPES}

    trans = C.estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + 21))
    reqs = C.gen_requests(cfg, topo, random.Random(cfg["seed"] + SEED_OFFSET_REQ),
                          LAM, DUR, attack_rate=RATE, trans=trans)
    n_now = {t: sum(1 for r in reqs if r.label == t) for t in TYPES + ("A7",)}

    print("A7 关闭：", n_base)
    print("A7 开启：", n_now)

    # 开启 A7 后，被判为 A7 的样本原本都是正常请求，故 A1 至 A4 的计数相同。
    moved = sum(1 for r in reqs if r.label == "A7")
    norm_base = sum(1 for r in base if r.label == "normal")
    norm_now = sum(1 for r in reqs if r.label == "normal")
    print("正常流：关闭 %d → 开启 %d（差 %d，A7 %d 条）"
          % (norm_base, norm_now, norm_base - norm_now, moved))
    assert all(n_now[t] == n_base[t] for t in TYPES), \
        "A7 的注入扰动了既有剧本的样本数：%s vs %s" % (n_base, n_now)
    assert moved > 0, "A7 未注入任何样本，检查 A7_VICTIM_PCT 与注入判据"
    assert norm_base - norm_now == moved, \
        "正常流减少的条数（%d）与 A7 的条数（%d）不一致，注入判据不是纯函数" % (norm_base - norm_now, moved)

    # ---- 2) 两个分量的作用域 ----
    norm = [r for r in reqs if r.label == "normal"]
    ns = [r.rho_struct for r in norm]
    nt = [r.rho_trans for r in norm]
    hi_norm = sum(1 for r in norm if C.ACTION_CLASS[r.action] == "高敏") / len(norm)
    print()
    print("%-6s%8s%12s%12s%12s" % ("剧本", "n", "ρ_struct", "ρ_trans", "高敏占比"))
    res = {}
    for lab in TYPES + ("A7",):
        s = [r for r in reqs if r.label == lab]
        a_s = auc([r.rho_struct for r in s], ns)
        a_t = auc([r.rho_trans for r in s], nt)
        hi = sum(1 for r in s if C.ACTION_CLASS[r.action] == "高敏") / len(s)
        res[lab] = (a_s, a_t)
        print("%-6s%8d%12.3f%12.3f%12.3f" % (lab, len(s), a_s, a_t, hi))
    print("%-6s%8d%12s%12s%12.3f" % ("正常", len(norm), "—", "—", hi_norm))

    # A7 的客体分布与正常流一致，结构通道应退化为随机判别；判别力来自状态迁移。
    assert abs(res["A7"][0] - 0.5) < 0.06, \
        "A7 在结构通道上并非随机判别（AUC=%.3f），与论文表 27 的口径不符" % res["A7"][0]
    assert res["A7"][1] > res["A7"][0] + 0.1, \
        "A7 的状态迁移判别力不足：ρ_struct=%.3f ρ_trans=%.3f" % res["A7"]
    # A3 的偏离在客体粒度上，两个分量的强弱应互换。
    assert res["A3"][0] > res["A3"][1] + 0.2, \
        "A3 的两个分量强弱未互换：ρ_struct=%.3f ρ_trans=%.3f" % res["A3"]

    # ---- 3) A7 的渐进性 ----
    print()
    print("A7 的渐进性（按暴露窗口内的进度分桶）：")
    a7 = [r for r in reqs if r.label == "A7"]
    onsets = {r.s_id: C.a7_onset(r.s_id, DUR) for r in a7}
    means, his = [], []
    for lo, hi_ in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        sel = [r for r in a7
               if lo <= (r.t - onsets[r.s_id]) / (C.A7_SPAN * DUR) < hi_]
        if not sel:
            continue
        m = float(np.mean([r.rho_trans for r in sel]))
        h = sum(1 for r in sel if C.ACTION_CLASS[r.action] == "高敏") / len(sel)
        means.append(m)
        his.append(h)
        print("   进度 [%.2f, %.2f)  n=%4d  ρ_trans 均值=%.3f  高敏占比=%.3f"
              % (lo, hi_, len(sel), m, h))
    # 每桶只有两百余个样本，均值的单步抖动可达 0.3；因此判据取趋势而非逐桶单调：
    # 末桶须显著高于首桶，且高敏操作占比应逐桶上升（后者由注入方式直接决定，不受噪声影响）。
    assert len(means) == 4, "分桶数不足：%s" % means
    assert means[-1] > means[0] + 0.15, "ρ_trans 未随暴露进度上升：%s" % means
    assert all(b > a for a, b in zip(his, his[1:])), "高敏操作占比未随进度上升：%s" % his

    print()
    print("全部检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
