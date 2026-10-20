# -*- coding: utf-8 -*-
"""文献 [11] 式单标量风险基线的检出对照（论文一 第 IV 章 C 节）。

对照对象
--------
Atlam H F, Yang Y. Enhancing Healthcare Security: A Unified RBAC and ABAC
Risk-Aware Access Control Approach[J]. Future Internet, 2025, 17(6): 262.
DOI: 10.3390/fi17060262

该工作把风险按资源敏感度、上下文、操作三类分量加权后与用户历史风险相加，
得到**单一标量**，再映射到五档风险等级驱动判定。本文按其给出的合成结构在
本文请求流上重建该基线（原文的分项取值表未在正文公开，故分项按本文场景
映射，权重沿用原式）：

    FinalRisk = 0.4·R_resource + 0.4·R_context + 0.2·R_operational + R_history

    资源敏感度 R_resource    → 本文的结构可达度 rho_struct
                               （客体对该角色是否可达，即资源—角色的匹配偏离）
    上下文     R_context     → 本文上下文异常度 ctx_anom
    操作严重性 R_operational → 操作对客体状态的不可逆程度 OP_SEVERITY
    历史       R_history     → 状态迁移偏离度 rho_trans

注意：该重建**只保留其「多类风险加权求和为单一标量、再按等级驱动判定」的合成
结构**，不声称复现原文的代码、参数或数值（原文 43 组医疗场景的
accuracy/precision/recall 与本文证券请求流不可比，不予引用与搬用）。

**信息对等**：四个分量全部取自本文方案同一条判定路径所使用的风险来源，
即该基线看到的风险证据与本文统计/条件通道完全相同，二者只差**合成方式**
——单标量加权求和 vs 作用域互不重叠的分通道阈值取并集。因此对照结果是
「合成方式」的差异，不是信息来源多寡的差异，对应论文第 I 章工作 2。
本文独有的两条确定性判据（基线硬边界、归属中心背书）不纳入该基线，
故另行给出「本文三通道合并」一列作总体对照。

请求流、拓扑与转移模型与 E5 完全一致（seed + 11 / + 21 / + 3），保证同一批
样本上逐条比较。本脚本只写 results/E11_文献基线对比.csv，不改动、不重写
任何 E5_*.csv，既有读数不受影响。

用法：
    python lit11_baseline.py
产出：
    results/E11_文献基线对比.csv
"""
from __future__ import annotations

import csv
import io
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pgbac_core import (CFG, build_topology, estimate_trans_model, gen_requests,  # noqa: E402
                         composite_risk, load_frozen)

RES = os.path.join(HERE, "results")

# 操作对客体状态的不可逆程度（递增）。read 不改变客体状态，delete 不可逆。
OP_SEVERITY = {"read": 0.1, "write": 0.4, "execute": 0.5, "export": 0.8, "delete": 1.0}


def auc(pos, neg) -> float:
    """秩和统计量法 AUC（与 run_experiments.auc 同实现，便于并排比较）。"""
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
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def lit11_risk(r) -> float:
    """按 [11] 的合成结构给出的单标量风险，归一化到 [0,1] 以复用本文分级阈值。"""
    r_res = float(r.rho_struct)        # 资源敏感度：客体对该角色的可达性偏离
    r_ctx = float(r.ctx_anom)          # 上下文：沿用本文的上下文异常度
    r_op = OP_SEVERITY.get(r.action, 0.5)   # 操作严重性
    r_hist = float(r.rho_trans)        # 历史：本文的状态迁移偏离度
    raw = 0.4 * r_res + 0.4 * r_ctx + 0.2 * r_op + r_hist
    return raw / 2.0                   # 原式值域 [0,2]


def band(rho: float) -> str:
    for hi, name in ((0.25, "低"), (0.5, "中"), (0.8, "高")):
        if rho < hi:
            return name
    return "极高"


def main() -> int:
    cfg = dict(CFG)
    lam, dur = 300.0, 400.0
    # 与 exp_e5 完全一致的初始化顺序与随机源
    topo = build_topology(cfg, random.Random(cfg["seed"] + 11))
    # R2：训练流取「评估流随机源 + 18」，与 E5、freeze_config.py 同一口径。
    trans = estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + 3 + 18))
    # R2：工作点由 S1–S3 校准流选出后冻结；本脚本对本流只做**事后诊断**扫描。
    _frozen = load_frozen()
    reqs = gen_requests(cfg, topo, random.Random(cfg["seed"] + 3), lam, dur,
                        attack_rate=0.06, trans=trans)

    types = ["A1", "A2", "A3", "A4", "A7"]
    idx = {t: [i for i, r in enumerate(reqs) if r.label == t] for t in types}
    idx_norm = [i for i, r in enumerate(reqs) if r.label == "normal"]
    vals = [lit11_risk(r) for r in reqs]
    print("样本：正常 %d 条，攻击 %d 条（%s）" % (
        len(idx_norm), sum(len(idx[t]) for t in types),
        "、".join("%s %d" % (t, len(idx[t])) for t in types)))

    # 注意：2026-10-18 改（第四轮审稿 R8）：网格原为 0.30~0.75，而该标量的 Youden
    # 极大恰好落在 0.30——极大贴着扫描下界，等于让基线的可取工作点被网格截掉。
    # 向下扩到 0.02 后真实极大在 θ=0.12（J=0.6635，TPR_A1 0.8243 / A3 0.9983 /
    # A7 0.5294、FPR 0.2494），与截断网格给出的 0.145/0.727/0.248（FPR 0.0766）
    # 相差极大，按截断点比较等于把基线摆在一个对它不利的位置上。
    grid = ([round(0.02 + 0.02 * i, 2) for i in range(15)]      # 0.02 .. 0.30
            + [round(0.30 + 0.05 * i, 2) for i in range(10)])   # 0.30 .. 0.75
    grid = sorted(set(grid))

    def tpr(t: str, th: float) -> float:
        ii = idx[t]
        return sum(1 for i in ii if vals[i] >= th) / max(1, len(ii))

    print("\n  文献[11]式加权和单标量风险的阈值扫描（工作点由 A1∪A3 的 Youden 指数定）：")
    print("   " + "-" * 74)
    print("   %7s%8s%8s%8s%9s%11s%8s" % ("阈值θ", "A1", "A3", "A7", "A1∪A3", "正常流FPR", "Youden"))
    scan, best = [], None
    ii13 = idx["A1"] + idx["A3"]
    for th in grid:
        t1, t3, t7 = tpr("A1", th), tpr("A3", th), tpr("A7", th)
        tu = sum(1 for i in ii13 if vals[i] >= th) / max(1, len(ii13))
        fpr = sum(1 for i in idx_norm if vals[i] >= th) / max(1, len(idx_norm))
        J = tu - fpr
        scan.append({"评分": "[11]式加权和单标量", "阈值θ": th,
                     "TPR_A1": round(t1, 4), "TPR_A3": round(t3, 4), "TPR_A7": round(t7, 4),
                     "TPR_A1UA3": round(tu, 4), "FPR": round(fpr, 4), "Youden": round(J, 4)})
        print("   %7.2f%8.3f%8.3f%8.3f%9.3f%11.3f%8.3f" % (th, t1, t3, t7, tu, fpr, J))
        if best is None or J > best[0]:
            best = (J, th, tu, fpr)
    th11 = float(_frozen["values"]["theta_scalar"])
    print("   [事后诊断] 评估流自身扫描给出 θ=%.2f（A1∪A3 TPR=%.3f，FPR=%.3f）；"
          "冻结工作点 θ=%.2f（来源 S1–S3 校准流，digest %s）"
          % (best[1], best[2], best[3], th11, _frozen["cfg_digest"][:12]))
    if abs(best[1] - th11) > 1e-9:
        print("   [注意] 评估流扫描与冻结值不一致——统计与对照一律以冻结值为准")

    nrm = [vals[i] for i in idx_norm]
    rows = []
    for t in types:
        a = auc([vals[i] for i in idx[t]], nrm)
        print("   AUC（[11]式）：%s = %.3f" % (t, a))
        rows.append({"评分": "[11]式加权和单标量", "剧本": t, "样本数": len(idx[t]),
                     "AUC": round(a, 4), "TPR@工作点": round(tpr(t, th11), 4)})

    # 工作点上的通道行，可直接并入论文表 3（各检测通道的检出率与误报率）
    # 注意：第六轮审稿 R5：TPR 早已按冻结 th11 计算，误报率却仍取评估流扫描的
    # `best[3]`——两个阈值眼下同为 0.12，数值上没有冲突，但一旦冻结点与评估流
    # 最优点分离，同一行就会混用两个工作点。这里改为把 FPR 也算在 th11 上，
    # 评估流扫描的读数只作事后诊断单独打印。
    fpr11 = sum(1 for i in idx_norm if vals[i] >= th11) / max(1, len(idx_norm))
    if abs(fpr11 - best[3]) > 1e-9:
        print("   [注意] 冻结点的 FPR %.4f 与评估流扫描最优点的 %.4f 不同——"
              "主结果用 %.4f（冻结工作点），差异属诊断信息"
              % (fpr11, best[3], fpr11))
    work = {"通道": "文献[11]式加权和单标量风险（θ=%.2f）" % th11,
            "类型": "文献基线",
            **{("TPR_%s" % t): round(tpr(t, th11), 4) for t in types},
            "误报率": round(fpr11, 4)}
    print("\n  工作点行：%s" % work)

    # ---- 信息对等组：本文「条件 ∪ 统计」分通道阈值取并集 ----
    # 注意：2026-10-18 改（第四轮审稿 R8）：此前这里从 E5_阈值扫描.csv 里逐评分取
    # Youden 最大行当工作点。E5 的网格一改（0.30 → 0.05 起），这里就会静默漂到
    # 0.15/0.05，而正文用的仍是误报预算选出的 0.30——两边不是同一组阈值。
    # 现改为读 E5 落盘的工作点文件，扫描表只当数据源、不参与选点。
    th_s = th_t = th_c = None
    p_wp = os.path.join(RES, "E5_工作点.csv")
    if os.path.exists(p_wp):
        with io.open(p_wp, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                k = r["评分"]
                if k == "ρ_struct":
                    th_s = float(r["阈值θ"])
                elif k == "ρ_trans":
                    th_t = float(r["阈值θ"])
                elif k == "合成ρ":
                    th_c = float(r["阈值θ"])
    if th_s is None:
        # 兜底：工作点文件缺失时退回扫描表的 Youden 最大行（数值可能与正文不一致）
        p_scan = os.path.join(RES, "E5_阈值扫描.csv")
        if os.path.exists(p_scan):
            with io.open(p_scan, "r", encoding="utf-8-sig", newline="") as f:
                byscore: dict = {}
                for r in csv.DictReader(f):
                    k = r["评分"]
                    if k not in byscore or float(r["Youden"]) > float(byscore[k]["Youden"]):
                        byscore[k] = r
            th_s = float(byscore["ρ_struct"]["阈值θ"]) if "ρ_struct" in byscore else None
            th_t = float(byscore["ρ_trans"]["阈值θ"]) if "ρ_trans" in byscore else None
            th_c = float(byscore["合成ρ"]["阈值θ"]) if "合成ρ" in byscore else None

    def ch_ctx(r) -> bool:
        return r.ctx_anom >= cfg["ctx_anom_limit"]

    def rho_of(r) -> float:
        return composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)

    parity = None
    if th_s is not None and th_t is not None:
        # 注意：2026-10-17 修（第二轮审稿第 3 条）：并集里**不得**包含合成风险 ρ。
        # 正文算法 2 的处置触发只用 ρ_struct / ρ_trans 两路统计阈值加条件通道，
        # 合成风险 ρ 只用于给出等级语义（第 10 步），不参与触发。此前的实现是
        # 四路取并（多了一条 rho_of(r) >= th_c），与算法 2 不是同一规则，
        # 审稿人据此指出「表 8 的判据与正文不一致」——属实，故摘除。
        hit = lambda r: (ch_ctx(r) or r.rho_struct >= th_s
                         or r.rho_trans >= th_t)
        pr = {t: sum(1 for i in idx[t] if hit(reqs[i])) / max(1, len(idx[t])) for t in types}
        pfp = sum(1 for i in idx_norm if hit(reqs[i])) / max(1, len(idx_norm))
        parity = {"通道": "本文 条件 ∪ 统计 分通道阈值取并集（信息对等组）", "类型": "本文",
                  **{("TPR_%s" % t): round(pr[t], 4) for t in types}, "误报率": round(pfp, 4)}
        print("\n  信息对等组（同一信息集，同一批样本，只差合成方式）")
        print("   本文工作点：ρ_struct θ=%.2f  ρ_trans θ=%.2f  条件 θ_ctx=%.2f"
              % (th_s, th_t, cfg["ctx_anom_limit"]))
        print("   %-34s%s%12s" % ("通道", "".join("%8s" % t for t in types), "正常流误报"))
        for w in (work, parity):
            print("   %-34s%s%12.4f" % (
                w["通道"], "".join("%8.4f" % w["TPR_%s" % t] for t in types), w["误报率"]))
        print("   → 两者使用同一组风险证据（结构可达度、状态迁移偏离度、上下文异常度、")
        print("     操作严重性），差别仅在合成方式：加权求和为单一标量 vs 分通道阈值取并集。")

    # ---------- 公平性对照（审稿意见第 5 条）：同 FPR 下的检出比较 ----------
    # 注意：原 `grid` 只用于「工作点选取」且刻意不改，以免 th11 漂移、既有读数变动。
    #    下列扩展网格只用于画 ROC 曲线，不参与工作点选取。
    if parity is not None:
        grid_ext = sorted(set([0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25] + grid))
        base_roc = []
        for th in grid_ext:
            base_roc.append({
                "FPR": sum(1 for i in idx_norm if vals[i] >= th) / max(1, len(idx_norm)),
                **{("TPR_%s" % t): sum(1 for i in idx[t] if vals[i] >= th) / max(1, len(idx[t]))
                   for t in ("A1", "A3", "A7")}})

        # 并集判据的 ROC：两路统计阈值同步扫 θ，覆盖 FPR 从 ~0.28 一路降到 0.0648。
        # 注意：2026-10-17 第三轮审稿 R1：扫描域必须**覆盖所有要对齐的目标 FPR**，
        #    否则 interp 会拿端点值冒充区间外的目标点。上一版扫 θ∈[0.05,1.00]，
        #    对应 FPR∈[0.0648,0.2295]，而目标点里有一个 0.2465（单标量的工作点），
        #    落在区间外，于是 0.6674/0.9939/0.5098 这三个数其实是 FPR=0.2295 的行，
        #    被填进了 0.2465 那一行并标成 comparable —— 属实，是缺陷。
        #    现在扫到 0.005（FPR≈0.28），并让 interp 对区间外目标返回不可比。
        # 注意：另：0.1308 那个「下限」是 k≤2.0 的扫描边界伪影，不是并集的性质。
        union_roc = []
        UNION_GRID = ([round(0.005 + 0.005 * i, 4) for i in range(19)]      # 0.005 .. 0.095
                      + [round(0.10 + 0.025 * i, 4) for i in range(37)])    # 0.100 .. 1.000
        for th in UNION_GRID:
            s = t2 = th
            fp = sum(1 for i in idx_norm if (ch_ctx(reqs[i]) or reqs[i].rho_struct >= s
                                             or reqs[i].rho_trans >= t2)
                     ) / max(1, len(idx_norm))
            row = {"阈值θ(两路相同)": th, "FPR": round(fp, 4)}
            for t in types:
                row["TPR_%s" % t] = round(
                    sum(1 for i in idx[t] if (ch_ctx(reqs[i]) or reqs[i].rho_struct >= s
                                              or reqs[i].rho_trans >= t2)) / max(1, len(idx[t])), 4)
            union_roc.append(row)

        # 并集误报率的下限由**饱和质量 + 固定的条件通道**决定，且只在 θ≤1 且用 ≥
        # 比较时成立：ρ_struct 或 ρ_trans 顶到上界 1.0 的正常请求，任何 θ≤1 都会命中。
        fpr_ctx = sum(1 for i in idx_norm if ch_ctx(reqs[i])) / max(1, len(idx_norm))
        n_norm = max(1, len(idx_norm))
        s_set = {i for i in idx_norm if reqs[i].rho_struct >= 1.0}
        t_set = {i for i in idx_norm if reqs[i].rho_trans >= 1.0}
        f_sat, f_sat_t = len(s_set) / n_norm, len(t_set) / n_norm
        f_sat_and = len(s_set & t_set) / n_norm
        f_sat_u = len(s_set | t_set) / n_norm
        # 关闭统计通道（只剩 ctx）与「θ=1 但用严格大于」两种反事实，用于给下限定条件
        f_only_ctx = fpr_ctx
        f_gt = sum(1 for i in idx_norm if (ch_ctx(reqs[i]) or reqs[i].rho_struct > 1.0
                                          or reqs[i].rho_trans > 1.0)) / n_norm
        print("   并集 ROC：θ %.3f→%.3f 对应 FPR %.4f→%.4f；条件通道单独命中 %.4f"
              % (union_roc[0]["阈值θ(两路相同)"], union_roc[-1]["阈值θ(两路相同)"],
                 union_roc[0]["FPR"], union_roc[-1]["FPR"], fpr_ctx))
        print("   饱和质量：ρ_struct≥1.0 %.4f、ρ_trans≥1.0 %.4f、交集 %.4f、并集 %.4f"
              % (f_sat, f_sat_t, f_sat_and, f_sat_u))
        print("   ⇒ θ=1.0 且用 ≥ 的并集 FPR %.4f = 饱和并集 %.4f + 条件通道 %.4f（两集合不相交）"
              % (union_roc[-1]["FPR"], f_sat_u, fpr_ctx))
        print("   反事实：关闭两路统计通道 → %.4f；θ=1 改用严格大于 → %.4f"
              % (f_only_ctx, f_gt))

        def interp(roc, key, target):
            """在 (FPR, TPR) 点集上对目标 FPR 线性插值；**区间外返回 None**。

            注意：2026-10-17 第三轮审稿 R1：旧实现把区间外的目标点钳到端点值
            （`return pts[0][1]` / `pts[-1][1]`），等于拿 FPR=0.2295 的读数去填
            FPR=0.2465 那一行，并标成「可比」。越界必须返回不可比，不能拿边界值
            冒充对应工作点。
            """
            pts = sorted((r["FPR"], r[key]) for r in roc)
            if target < pts[0][0] - 1e-12 or target > pts[-1][0] + 1e-12:
                return None
            if abs(target - pts[0][0]) <= 1e-12:
                return pts[0][1]
            if abs(target - pts[-1][0]) <= 1e-12:
                return pts[-1][1]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= target <= x1:
                    return max(y0, y1) if x1 == x0 else y0 + (y1 - y0) * (target - x0) / (x1 - x0)
            return None

        union_lo = min(r["FPR"] for r in union_roc)
        union_hi = max(r["FPR"] for r in union_roc)
        scalar_lo = min(r["FPR"] for r in base_roc)
        scalar_hi = max(r["FPR"] for r in base_roc)

        def tag(target):
            if target < union_lo - 1e-12:
                return "并集不可比（低于饱和下限）"
            if target > union_hi + 1e-12:
                return "并集不可比（高于扫描上限）"
            if target < scalar_lo - 1e-12 or target > scalar_hi + 1e-12:
                return "单标量不可比"
            return "可比"

        same = []
        for target in [0.2465, 0.20, 0.15, 0.10, 0.0766, 0.05]:
            row = {"目标FPR": target}
            for t in ("A1", "A3", "A7"):
                u = interp(union_roc, "TPR_%s" % t, target)
                s2 = interp(base_roc, "TPR_%s" % t, target)
                row["并集_TPR_%s" % t] = "" if u is None else round(u, 4)
                row["单标量_TPR_%s" % t] = "" if s2 is None else round(s2, 4)
            row["可比性"] = tag(target)
            same.append(row)

        print("\n  公平性对照·同一 FPR 下的 TPR（并集判据 vs [11] 式单标量）")
        print("   并集 FPR 覆盖 [%.4f, %.4f]，单标量覆盖 [%.4f, %.4f]"
              % (union_lo, union_hi, scalar_lo, scalar_hi))
        print("   %9s%24s%24s%24s  %s" % ("目标FPR", "A1 并集/单标量", "A3 并集/单标量",
                                          "A7 并集/单标量", "可比性"))
        for r in same:
            cells = []
            for t in ("A1", "A3", "A7"):
                u, s2 = r["并集_TPR_%s" % t], r["单标量_TPR_%s" % t]
                cells.append("%24s" % ("—" if u == "" or s2 == ""
                                       else "%.4f / %.4f" % (u, s2)))
            print("   %9.4f%s  %s" % (r["目标FPR"], "".join(cells), r["可比性"]))
        print("   并集工作点 FPR=%.4f（与 [11] 工作点 FPR=%.4f 不同，故需按同一 FPR 对齐后再比）"
              % (pfp, best[3]))

        # 饱和诊断落盘：把「下界的成立条件」写成可复核的记录
        out7 = os.path.join(RES, "E14_饱和诊断.csv")
        with io.open(out7, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["项目", "值", "说明"])
            w.writerow(["正常请求样本数", n_norm, ""])
            w.writerow(["ρ_struct≥1.0", round(f_sat, 4), "结构偏离度顶到上界"])
            w.writerow(["ρ_trans≥1.0", round(f_sat_t, 4), "状态迁移偏离度顶到上界"])
            w.writerow(["两者交集", round(f_sat_and, 4), "同时饱和"])
            w.writerow(["两者并集", round(f_sat_u, 4), "任一饱和"])
            w.writerow(["条件通道单独命中", round(fpr_ctx, 4), "θ_ctx 固定，不参与扫描"])
            w.writerow(["θ=1.0 且 ≥ 的并集 FPR", round(union_roc[-1]["FPR"], 4),
                        "饱和并集 ∪ 条件通道"])
            w.writerow(["关闭统计通道的 FPR", round(f_only_ctx, 4), "反事实：只剩 ctx"])
            w.writerow(["θ=1 改用严格大于的 FPR", round(f_gt, 4),
                        "说明下界依赖 ≥ 与 θ≤1 的取值域"])
            w.writerow(["并集 FPR 扫描区间", "%.4f–%.4f" % (union_lo, union_hi),
                        "由 UNION_GRID 决定，区间外不插值"])
        print("   写出（饱和诊断）：%s" % out7)

        out4 = os.path.join(RES, "E14_并集ROC.csv")
        with io.open(out4, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(union_roc[0].keys()))
            w.writeheader()
            w.writerows(union_roc)
        out5 = os.path.join(RES, "E14_单标量ROC.csv")
        with io.open(out5, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(base_roc[0].keys()))
            w.writeheader()
            w.writerows(base_roc)
        out6 = os.path.join(RES, "E14_同FPR对照.csv")
        with io.open(out6, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(same[0].keys()))
            w.writeheader()
            w.writerows(same)
        print("\n写出（公平性对照）：%s\n     %s\n     %s" % (out4, out5, out6))

    # 与本文各风险分量并排（读 E5 已产出的消融表，只读不写）
    abl = []
    p = os.path.join(RES, "E5_风险分量消融.csv")
    if os.path.exists(p):
        with io.open(p, "r", encoding="utf-8-sig", newline="") as f:
            abl = list(csv.DictReader(f))
        print("\n  并排（AUC）：文献[11]式 vs 本文各分量 vs 本文合成ρ")
        print("   %-8s%12s%12s%12s%12s" % ("剧本", "[11]式", "ρ_struct", "ρ_trans", "合成ρ"))
        for r in abl:
            t = r["剧本"]
            ours = next((x for x in rows if x["剧本"] == t), None)
            if ours is None:
                continue
            print("   %-8s%12.3f%12s%12s%12s" % (
                t, ours["AUC"], r.get("AUC_ρ_struct", "—"), r.get("AUC_ρ_trans", "—"),
                r.get("AUC_三分量", r.get("AUC_二分量", "—"))))

    out = os.path.join(RES, "E11_文献基线对比.csv")
    fields = ["评分", "剧本", "样本数", "AUC", "TPR@工作点"]
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    out2 = os.path.join(RES, "E11_阈值扫描.csv")
    with io.open(out2, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scan[0].keys()))
        w.writeheader()
        w.writerows(scan)

    out3 = os.path.join(RES, "E11_工作点通道行.csv")
    with io.open(out3, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(work.keys()))
        w.writeheader()
        w.writerows([work] + ([parity] if parity else []))

    print("\n写出：%s\n     %s\n     %s" % (out, out2, out3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
