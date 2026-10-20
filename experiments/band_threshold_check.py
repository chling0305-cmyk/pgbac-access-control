# -*- coding: utf-8 -*-
"""状态机门限对照：统一门限 vs 按目标档位下界。

背景（第二轮意见 R3 的「新增数值还需说明」）
-------------------------------------------------
正文在说明第 15 步的门限 h_up 时报告：把门限改为按目标档位下界定义后，
正常流被收紧的比例由 0.187 升至 0.436，拒绝比例不变（0.105）。而同一篇
在结果 B 报告的正常流处置分布是「收紧或降级 0.204、拒绝 0.109」。

两组数不是同一口径，必须写明：
  * 主结果（E5_处置分布.csv）：完整 E5 正常流 111807 条，报告口径把
    「收紧（ℓ=1）」与「降级（ℓ=2）」合并为一类；
  * 本对照：60 s 短流（quick 口径，见脚本输出），用于比较两种门限的**方向**，
    样本量小一个数量级，且对照时四档分列。

本脚本只读现有实现，不改判定代码：现行门限直接调用 pgbac_core.DecisionEngine，
替代门限用子类覆写升级条件（其余分支逐行照搬，保证只改一处变量）。

用法：python experiments/band_threshold_check.py
产物：experiments/results/E16_门限对照.csv
"""
from __future__ import annotations

import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pgbac_core as core
from pgbac_core import CFG, build_topology, composite_risk, estimate_trans_model, gen_requests

OUT = os.path.join(HERE, "results", "E16_门限对照.csv")


class BandEngine(core.DecisionEngine):
    """升级门限按目标档位下界定义；降级门限按当前档位下界减 margin。"""

    def __init__(self, cfg, topo, margin_up=0.0, margin_down=0.08):
        super().__init__(cfg, topo)
        self.lo = [0.0] + [hi for (hi, _) in cfg["risk_bands"][:-1]]
        self.mu = margin_up
        self.md = margin_down

    def decide(self, r, step_idx):
        # 前置分支与 DecisionEngine 逐行一致，唯一差别在下面的升级条件
        if not r.in_baseline:
            return core.Decision("deny", "基线外客体", composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 1)
        if not r.center_ok:
            return core.Decision("deny", "非归属中心", composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 1)
        cons = set()
        via_cond = False
        if r.sec == 4 and r.action in ("delete", "export") and r.conf < 0.6:
            cons.add("增强认证")
            via_cond = True
        if r.ctx_anom >= self.cfg["ctx_anom_limit"]:
            cons.add("增强认证")
            via_cond = True
        trust_side = False
        need = self.cfg["security_min_trust"][r.sec]
        if r.trust_L < need:
            if r.conf >= 0.7:
                cons.add("增强认证")
                trust_side = True
            else:
                return core.Decision("deny", "信任不足",
                                     composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 3)

        rho = composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)
        name, band = self._band(rho)
        key = (r.s_id, r.obj)
        st = self._state.setdefault(key, {"band": 0, "streak": 0})
        if band > st["band"] and (rho >= self.lo[band] + self.mu or band - st["band"] > 1):
            st["streak"] += 1
            if st["streak"] >= self.cfg["damping_windows"] or band - st["band"] > 1:
                st["band"] = min(st["band"] + 1, len(self.cfg["risk_bands"]) - 1)
                st["streak"] = 0
        elif band < st["band"] and rho <= self.lo[st["band"]] - self.md:
            st["band"] = band
            st["streak"] = 0
        else:
            st["streak"] = 0
        band = st["band"]          # 与 DecisionEngine 相同：动作取自稳定档位 ℓ
        name = self.cfg["risk_bands"][band][1]
        if band == 3:
            return core.Decision("deny", "高风险拒绝", rho, name, 4)
        if band == 1:
            cons.add("增强认证")
        elif band == 2:
            cons.add("权限降级")
        C = frozenset(cons)
        if not C:
            return core.Decision("permit", "正常放行", rho, name, 4, C)
        head = "权限降级" if band == 2 else ("收紧约束" if band == 1 else "需增强认证")
        tail = "＋信任不足" if trust_side else ("＋条件通道" if via_cond else "")
        return core.Decision("refine", head + tail, rho, name, 4, C)


def run(engine, reqs):
    """按附加约束集合分列：含「权限降级」记降级，否则记收紧。"""
    tally = {}
    for k, r in enumerate(reqs):
        d = engine.decide(r, k)
        key = "正常" if r.label == "normal" else r.label
        slot = tally.setdefault(key, {"n": 0, "放行": 0, "收紧": 0, "降级": 0, "拒绝": 0})
        slot["n"] += 1
        if d.d == "permit":
            slot["放行"] += 1
        elif d.d == "deny":
            slot["拒绝"] += 1
        elif "权限降级" in d.cons:
            slot["降级"] += 1
        else:
            slot["收紧"] += 1
    return tally


def main() -> None:
    cfg = dict(CFG)
    rng = random.Random(cfg["seed"] + 11)
    topo = build_topology(cfg, rng)
    trans = estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + 21))
    reqs = gen_requests(cfg, topo, random.Random(cfg["seed"] + 3), 300.0, 60.0,
                        attack_rate=0.06, trans=trans)
    n_norm = sum(1 for r in reqs if r.label == "normal")
    print("短流样本：%d 条请求，其中正常 %d 条（dur=60 s，quick 口径）" % (len(reqs), n_norm))
    print("主结果口径：E5_处置分布.csv 的正常流 111807 条，收紧与降级合并报告\n")

    runs = [
        ("统一门限 h_up=0.50 / h_down=0.40（现行）", core.DecisionEngine(cfg, topo)),
        ("按目标档位下界，升级余量 0（替代）", BandEngine(cfg, topo, margin_up=0.0, margin_down=0.08)),
    ]
    rows = []
    for tag, eng in runs:
        t = run(eng, reqs)
        norm = t["正常"]
        n = norm["n"]
        print("  %-34s 放行 %.3f  收紧 %.3f  降级 %.3f  拒绝 %.3f" % (
            tag, norm["放行"] / n, norm["收紧"] / n, norm["降级"] / n, norm["拒绝"] / n))
        rows.append({
            "门限配置": tag, "样本": n,
            "放行": round(norm["放行"] / n, 4),
            "收紧（ℓ=1）": round(norm["收紧"] / n, 4),
            "降级（ℓ=2）": round(norm["降级"] / n, 4),
            "拒绝": round(norm["拒绝"] / n, 4),
            "收紧或降级合计": round((norm["收紧"] + norm["降级"]) / n, 4),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n写入", OUT)
    print("注：短流样本量与主结果不同，两列数值不可直接比较，只用于判断方向。")


if __name__ == "__main__":
    main()
