# -*- coding: utf-8 -*-
"""统计通道工作点的跨流复算与选点协议（第四轮审稿 R8）。

原稿只说「统计通道的工作点随后按同一流程在另外三条流上重新导出，三次均落在 0.30」，
但没有留下可复算的脚本、流标识与样本量；审稿人据此要求给出数据划分、流的生成方式
与跨运行的检出率/误报率。本脚本把这件事做成可复算的。

协议（正文方法 D 节的数据分工按此写）
--------------------------------------
1. 训练流：仅含正常请求，随机源 seed+off+118，与评估流的请求源不相交；
   只用于估计状态迁移模型与逐角色可达子图。
2. 校准流：随机源 seed+off，off∈{101,202,303}。阈值**只在这三条流上**确定
   （freeze_config.py），三条流必须选出同一值才写入冻结配置。
3. 评估流：off=3 即 E5 所用流，正文所有比率都由它给出，且**不参与选点**；
   其余三条流在正文里只作稳定性对照。

选点规则（不是 Youden 极大）
---------------------------
θ_struct 取「自身误报率不超过预算 b=0.12 的网格点里检出最高的那个」。
把网格向下扩到 0.05 之后，结构偏离度的 Youden 极大落在 0.15——
贴着网格低端，属准则本身的退化，故不作选点依据；0.30 是预算规则唯一选出的点
（0.25 的误报率 0.1210 已越界）。θ_trans 与 θ_struct 取同一值，以保持单一运行参数。

输出 results/E5_阈值跨流.csv。
"""
import csv
import io
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pgbac_core as C            # noqa: E402

RES = os.path.join(HERE, "results")
GRID = [round(0.05 + 0.05 * i, 2) for i in range(15)]   # 0.05 .. 0.75
BUDGET = 0.12                                            # 结构偏离度的误报预算
TRANS_OFF = 18                                           # 迁移模型的随机源偏移（off=3 时即 E5 的 +21）
# R2：阈值不再写死在这个脚本里，而是读冻结配置。本脚本的角色是**报告**四条流上
# 的选点结果与并集读数，不承担选点；选点在 freeze_config.py 里完成。
_FROZEN = C.load_frozen()
THETA = float(_FROZEN["values"]["theta_struct"])

OFFSETS = [(3, "S0 评估流"), (101, "S1 校准流"), (202, "S2 校准流"), (303, "S3 校准流")]


def curves(reqs):
    """返回 ρ_struct 与 ρ_trans 的整条扫描曲线，以及与 E5 一致的 Youden 定义。"""
    norm = [r for r in reqs if r.label == "normal"]
    a1 = [r for r in reqs if r.label == "A1"]
    a3 = [r for r in reqs if r.label == "A3"]
    a7 = [r for r in reqs if r.label == "A7"]
    a13 = a1 + a3
    out = {}
    for key in ("rho_struct", "rho_trans"):
        rows = []
        for th in GRID:
            fpr = sum(1 for r in norm if getattr(r, key) >= th) / max(1, len(norm))
            t3 = sum(1 for r in a3 if getattr(r, key) >= th) / max(1, len(a3))
            t7 = sum(1 for r in a7 if getattr(r, key) >= th) / max(1, len(a7))
            tu = sum(1 for r in a13 if getattr(r, key) >= th) / max(1, len(a13))
            j = (tu if key == "rho_struct" else t7) - fpr
            rows.append((th, t3, t7, tu, fpr, j))
        out[key] = rows
    return out, norm


def budget_point(rows):
    """预算规则：误报率不超预算的点里检出最高者；同检出取更严的阈值。"""
    ok = [r for r in rows if r[4] <= BUDGET]
    if not ok:
        return None
    return max(ok, key=lambda r: (r[3], r[0]))


def free_max(rows, idx):
    return max(rows, key=lambda r: r[idx])


def union_at(reqs, th):
    """固定阈值下的并集检出率与并集误报率（与 E5 同一规则，不含合成风险）。"""
    ch_hard = lambda r: not r.in_baseline
    ch_center = lambda r: not r.center_ok
    ch_ctx = lambda r: r.ctx_anom >= C.CFG["ctx_anom_limit"]
    hit = lambda r: (ch_hard(r) or ch_center(r) or ch_ctx(r)
                     or r.rho_struct >= th or r.rho_trans >= th)
    norm = [r for r in reqs if r.label == "normal"]
    res = {}
    for t in ("A1", "A3", "A7"):
        sub = [r for r in reqs if r.label == t]
        res[t] = round(sum(1 for r in sub if hit(r)) / max(1, len(sub)), 4)
    res["FPR"] = round(sum(1 for r in norm if hit(r)) / max(1, len(norm)), 4)
    return res


def main():
    cfg = dict(C.CFG)
    topo = C.build_topology(cfg, random.Random(cfg["seed"] + 11))
    out = []
    for off, tag in OFFSETS:
        trans = C.estimate_trans_model(cfg, topo,
                                       random.Random(cfg["seed"] + off + TRANS_OFF))
        reqs = C.gen_requests(cfg, topo, random.Random(cfg["seed"] + off), 300.0, 400.0,
                              attack_rate=0.06, trans=trans)
        n_norm = sum(1 for r in reqs if r.label == "normal")
        n_att = len(reqs) - n_norm
        cvs, _ = curves(reqs)
        bs = budget_point(cvs["rho_struct"])
        fs = free_max(cvs["rho_struct"], 5)
        ft = free_max(cvs["rho_trans"], 5)
        at30 = [r for r in cvs["rho_struct"] if abs(r[0] - THETA) < 1e-9][0]
        tt30 = [r for r in cvs["rho_trans"] if abs(r[0] - THETA) < 1e-9][0]
        un = union_at(reqs, THETA)
        print("%-12s 正常 %6d 攻击 %5d | 预算选点 θ=%.2f（FPR=%.4f）| "
              "自由极大 θ_s=%.2f J=%.4f / θ_t=%.2f J=%.4f"
              % (tag, n_norm, n_att, bs[0] if bs else -1, bs[4] if bs else -1,
                 fs[0], fs[5], ft[0], ft[5]))
        print("             @0.30  A3=%.4f FPR_s=%.4f | A7=%.4f FPR_t=%.4f | "
              "并集 A1=%.4f A3=%.4f A7=%.4f FPR=%.4f"
              % (at30[1], at30[4], tt30[2], tt30[4],
                 un["A1"], un["A3"], un["A7"], un["FPR"]))
        out.append({
            "流标识": tag, "随机源偏移": off,
            "正常样本": n_norm, "攻击样本": n_att,
            "预算选点θ": bs[0] if bs else "", "预算选点FPR": round(bs[4], 4) if bs else "",
            "自由极大θ_s": fs[0], "自由极大J_s": round(fs[5], 4),
            "自由极大θ_t": ft[0], "自由极大J_t": round(ft[5], 4),
            "θ=0.30_TPR_A3": round(at30[1], 4), "θ=0.30_FPR_struct": round(at30[4], 4),
            "θ=0.30_J_struct": round(at30[5], 4),
            "θ=0.30_TPR_A7": round(tt30[2], 4), "θ=0.30_FPR_trans": round(tt30[4], 4),
            "并集_A1": un["A1"], "并集_A3": un["A3"], "并集_A7": un["A7"],
            "并集_FPR": un["FPR"],
        })

    p = os.path.join(RES, "E5_阈值跨流.csv")
    with io.open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\n写出 %s" % p)
    print("冻结阈值 θ=%.2f（frozen_config.json digest %s）"
          % (THETA, _FROZEN["cfg_digest"][:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
