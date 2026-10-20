# -*- coding: utf-8 -*-
"""算法 2 伪码与原型实现的逐求值步一致性核对（第五轮意见 R3.2）。

背景
----
v3.9.22 把算法 2 的附加约束由单值 con 改成集合 C，但原型仍在「信任等级不足
且置信度高」处**提前返回** refine，不进入风险档位；第 2 步的两处条件约束同样
提前返回。于是一部分请求绕过了第 15 步的状态累积，遇到 ℓ = 3 时也不会被风险
侧否决——伪码与实现不是「约束粒度不同」，而是决策可能不同。

本脚本做三件事：

1. 按论文算法 2 逐步实现参考判定（ReferenceMerge），动作取自稳定档位 ℓ；
2. 逐求值步调用原型 DecisionEngine.decide；
3. 把两者在每个求值步上的输入、求值前后的 ℓ 与 streak、输出决策与约束集合
   逐项列出并比对。覆盖审稿人点名的 ℓ = 3 组合，以及越级、条件通道与
   信任侧同时给约束的组合。

用法：python experiments/decision_merge_check.py
产物：experiments/results/E16_约束合并核对.csv
"""
from __future__ import annotations

import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pgbac_core import (CFG, DecisionEngine, Request, build_topology,  # noqa: E402
                        composite_risk)

OUT = os.path.join(HERE, "results", "E16_约束合并核对.csv")

HI = CFG["hysteresis_up"]
HD = CFG["hysteresis_down"]
M = CFG["damping_windows"]
BANDS = CFG["risk_bands"]
CTX_LIMIT = CFG["ctx_anom_limit"]


def band_of(rho: float) -> int:
    for i, (hi, _name) in enumerate(BANDS):
        if rho < hi:
            return i
    return len(BANDS) - 1


class ReferenceMerge:
    """论文算法 2 的逐行参考实现：约束只做并集，动作取自 ℓ。"""

    def __init__(self) -> None:
        self.state = {}

    def decide(self, key, *, in_baseline, has_policy, L, Lmin, conf,
               sec, action, ctx_anom, rs, rt, rb, e_effect="permit"):
        C = set()                                                  # 步 1
        if not in_baseline:                                        # 步 2–4
            return "deny", C
        if not has_policy:                                         # 步 6
            return "deny", C
        if sec == 4 and action in ("delete", "export") and conf < 0.6:   # 步 5 的 ABAC 条件
            C.add("增强认证")
        if ctx_anom >= CTX_LIMIT:
            C.add("增强认证")
        if L < Lmin:                                               # 步 7–11
            if conf >= 0.7:
                C.add("增强认证")                                   # 步 8：只记录，不放行
            else:
                return "deny", C                                   # 步 9
        rho = composite_risk(rs, rt, rb)                           # 步 13
        b = band_of(rho)                                           # 步 14
        st = self.state.setdefault(key, {"l": 0, "streak": 0})      # 步 15
        if b > st["l"] and (rho >= HI or b - st["l"] > 1):
            st["streak"] += 1
            if st["streak"] >= M or b - st["l"] > 1:
                st["l"] = min(st["l"] + 1, len(BANDS) - 1)
                st["streak"] = 0
        elif b < st["l"] and rho <= HD:
            st["l"] = b
            st["streak"] = 0
        else:
            st["streak"] = 0
        ell = st["l"]                                              # 步 17：动作取自 ℓ
        if e_effect == "deny" or ell == 3:                          # 步 18
            return "deny", set()
        C = C | {0: set(), 1: {"增强认证"}, 2: {"权限降级"}, 3: set()}[ell]
        return ("permit" if not C else "refine"), C                 # 步 19


def make_req(topo, role, obj, trust_L, conf, rs, rt, rb, sid,
             action="read", ctx_anom=0.0, t=0.0, center=0, in_baseline=None):
    if in_baseline is None:
        in_baseline = obj in topo.role_perm[role]
    return Request(
        t=t, s_id=sid, role=role, center=center, home=0, obj=obj,
        obj_i=topo.objects.index(obj), sec=topo.sec_of[obj], action=action,
        in_baseline=in_baseline, ctx_anom=ctx_anom,
        rho_struct=rs, rho_trans=rt, rho_behav=rb, trust_L=trust_L, conf=conf,
    )


def pick(topo, domain, level):
    for o in topo.objects:
        if topo.domain_of[o] == domain and topo.sec_of[o] == level:
            return o
    raise SystemExit("找不到满足条件的客体：%s / sec=%d" % (domain, level))


def main() -> None:
    rng = random.Random(CFG["seed"])
    topo = build_topology(CFG, rng)
    eng = DecisionEngine(CFG, topo)
    ref = ReferenceMerge()

    high_obj = pick(topo, "清算", 4)      # sec=4 ⇒ L_min=4
    low_obj = pick(topo, "交易", 2)

    # 名称, 角色, 客体, L, conf, (ρ_s, ρ_t, ρ_b), 求值步数, 动作, ctx, 说明
    # 分量值按合成式 ρ = 1 − (1−a)³ 反解，使合成风险落在目标档位：
    #   a = 0.13 ⇒ 0.341（中档）、0.28 ⇒ 0.627（高档）、0.55 ⇒ 0.909（极高档）
    cases = [
        ("信任不足 + conf 高 + ρ 低（ℓ=0）", "合规审计", high_obj, 3, 0.90,
         (0.05, 0.05, 0.05), 1, "read", 0.0, "意见点名的组合：认证要求不得被档位映射清空"),
        ("信任不足 + conf 低 + ρ 低", "合规审计", high_obj, 3, 0.30,
         (0.05, 0.05, 0.05), 1, "read", 0.0, "信任与置信度均不足，两侧都应拒绝"),
        ("信任充足 + ρ 低", "柜台经纪", low_obj, 4, 0.90,
         (0.05, 0.05, 0.05), 1, "read", 0.0, "无任何约束才是 permit 的唯一情形"),
        ("信任充足 + ρ 中 0.341（三步，不越过 h_up）", "柜台经纪", low_obj, 4, 0.90,
         (0.13, 0.13, 0.13), 3, "read", 0.0, "中风险区间不升级，三轮都停在同一档位"),
        ("信任不足 + ρ 中 0.341（三步）", "合规审计", high_obj, 3, 0.90,
         (0.13, 0.13, 0.13), 3, "read", 0.0, "档位不动但 C 非空：不得放行"),
        ("信任不足 + conf 高 + ρ 高 0.627（四步）", "合规审计", high_obj, 3, 0.90,
         (0.28, 0.28, 0.28), 4, "read", 0.0, "两处要求并存：增强认证 + 权限降级"),
        ("信任不足 + conf 高 + ρ 极高 0.909（五步，到 ℓ=3）", "合规审计", high_obj, 3, 0.90,
         (0.55, 0.55, 0.55), 5, "read", 0.0, "审稿人点名：ℓ=3 必须由风险侧否决，不能被信任分支跳过"),
        ("信任充足 + ρ 极高 0.909（单步）", "柜台经纪", low_obj, 4, 0.90,
         (0.55, 0.55, 0.55), 1, "read", 0.0, "越级只升一档：单窗只收紧，不直接拒绝"),
        ("条件通道命中（高敏导出 + conf 低）+ ρ 低", "合规审计", high_obj, 4, 0.50,
         (0.05, 0.05, 0.05), 1, "export", 0.0, "第 2 步的条件约束与信任侧同为约束来源"),
        ("条件通道命中（上下文异常）+ ρ 高 0.627（三步）", "柜台经纪", low_obj, 4, 0.90,
         (0.28, 0.28, 0.28), 3, "read", 0.9, "条件约束与档位约束并存"),
    ]

    rows = []
    for i, (name, role, obj, L, conf, rho, n_win, action, ctx, note) in enumerate(cases, 1):
        sid = "s-%02d" % i
        key = (sid, obj)
        Lmin = CFG["security_min_trust"][topo.sec_of[obj]]
        for w in range(n_win):
            # 注意：2026-10-18 改（第五轮审稿 R3.1）：原来每一步都发同一条请求（t 恒为 0）。
            # 状态推进的单位是**请求**，故每一步必须是一条**新请求**；否则测的是
            # 「同一条请求被重复求值」，而那正是评审指出不该计入升级的情形。
            req = make_req(topo, role, obj, L, conf, *rho, sid=sid,
                           action=action, ctx_anom=ctx, t=float(w))
            st_i = eng._state.get(key, {"band": 0, "streak": 0})
            l_before, s_before = st_i["band"], st_i["streak"]
            impl = eng.decide(req, w)
            d_ref, c_ref = ref.decide(
                key, in_baseline=True, has_policy=True, L=L, Lmin=Lmin, conf=conf,
                sec=topo.sec_of[obj], action=action, ctx_anom=ctx, **dict(zip(("rs", "rt", "rb"), rho)))
            st_ref = ref.state.get(key, {"l": 0, "streak": 0})
            rows.append({
                "组合": name,
                "求值步": w + 1,
                "角色": role,
                "策略效果": "permit",
                "L": L,
                "L_min": Lmin,
                "conf": conf,
                "rho_struct": rho[0],
                "rho_trans": rho[1],
                "rho_behav": rho[2],
                "合成rho": round(composite_risk(*rho), 4),
                "求值前ℓ": l_before,
                "求值前streak": s_before,
                "求值后ℓ": st_ref["l"],
                "求值后streak": st_ref["streak"],
                "伪码决策": d_ref,
                "伪码约束集合": "+".join(sorted(c_ref)) or "∅",
                "实现决策": impl.d,
                "实现约束集合": "+".join(sorted(impl.cons)) or "∅",
                "决策一致": "是" if d_ref == impl.d else "否",
                "约束一致": "是" if set(c_ref) == set(impl.cons) else "否",
                "说明": note if w == 0 else "",
            })
        # 两个引擎共用同一组状态，逐求值步推进后必须继续对齐
        eng._state.pop(key, None)
        ref.state.pop(key, None)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("算法 2 约束合并核对：%d 个组合、%d 个求值步" % (len(cases), len(rows)))
    for r in rows:
        print("  %-46s 步%d ℓ%d→%d  伪码=%-6s %-16s 实现=%-6s %-16s 一致=%s" % (
            r["组合"][:46], r["求值步"], r["求值前ℓ"], r["求值后ℓ"],
            r["伪码决策"], r["伪码约束集合"], r["实现决策"], r["实现约束集合"],
            r["决策一致"]))
    n_leak = sum(1 for r in rows if r["伪码决策"] == "permit" and r["伪码约束集合"] != "∅")
    n_leak_impl = sum(1 for r in rows if r["实现决策"] == "permit" and r["实现约束集合"] != "∅")
    print("C 非空却放行：伪码 %d 处、实现 %d 处（均要求为 0）" % (n_leak, n_leak_impl))
    print("决策一致 %d/%d，约束集合一致 %d/%d" % (
        sum(1 for r in rows if r["决策一致"] == "是"), len(rows),
        sum(1 for r in rows if r["约束一致"] == "是"), len(rows)))
    print("写入", OUT)

    # ---- 同一请求重复求值：不得计入升级（R3.1 的第二半） ----
    # 评审的做法是对同一 (subject, object)、相同输入连续调用四次，看 ℓ 是否被推高。
    # 状态推进的单位既然是「请求」，同一条请求被问第二次就不是新的风险观察。
    eng2 = DecisionEngine(CFG, topo)
    req = make_req(topo, "合规审计", high_obj, 3, 0.90, 0.28, 0.28, 0.28,
                   sid="s-rep", action="read", ctx_anom=0.0, t=7.5)
    key = (req.s_id, req.obj)
    rep = []
    print("\n同一请求重复求值（同一 (s,o)、同一输入、同一请求时刻）：")
    for k in range(4):
        before = dict(eng2._state.get(key, {"band": 0, "streak": 0}))
        d = eng2.decide(req, k)
        after = dict(eng2._state.get(key, {"band": 0, "streak": 0}))
        rep.append({"调用次数": k + 1, "求值前ℓ": before.get("band", 0),
                    "求值后ℓ": after.get("band", 0),
                    "求值前streak": before.get("streak", 0),
                    "求值后streak": after.get("streak", 0),
                    "决策": d.d, "约束集合": "+".join(sorted(d.cons)) or "∅"})
        print("  第%d次  ℓ %d→%d  streak %d→%d  %s %s"
              % (k + 1, rep[-1]["求值前ℓ"], rep[-1]["求值后ℓ"],
                 rep[-1]["求值前streak"], rep[-1]["求值后streak"],
                 d.d, rep[-1]["约束集合"]))
    p2 = os.path.join(HERE, "results", "E16_重复求值.csv")
    with open(p2, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
        wr.writeheader()
        wr.writerows(rep)
    print("写入", p2)

    # ---- 同一标识、输入变化：不得复用旧 permit（第六轮审稿 R1） ----
    # 评审的做法是：先对一个合法低风险请求取 permit，保持 (主体, 客体, 动作,
    # 请求时刻) 四项不变，只改决定授权的输入，看缓存是否把旧 permit 直接返回。
    # 修复后的语义是「快照不一致即重新求值」，四条变化都应与新引擎的判定一致。
    rho_hi = (1.0, 1.0, 1.0)
    variants = [
        ("基线外客体", dict(in_baseline=False), "deny"),
        ("非归属中心发起", dict(center=1), "deny"),
        ("信任等级不足且置信度低", dict(trust_L=1, conf=0.3), "deny"),
        ("三风险分量均为 1.0", dict(rho_struct=rho_hi[0], rho_trans=rho_hi[1],
                                rho_behav=rho_hi[2]), "refine"),
    ]
    ch_rows = []
    for name, chg, expect in variants:
        e1 = DecisionEngine(CFG, topo)
        first = make_req(topo, "柜台经纪", low_obj, 4, 0.90, 0.05, 0.05, 0.05,
                         sid="s-cache", action="read", ctx_anom=0.0, t=11.0)
        d0 = e1.decide(first, 0)
        changed = make_req(topo, "柜台经纪", low_obj,
                           chg.get("trust_L", 4), chg.get("conf", 0.90),
                           chg.get("rho_struct", 0.05), chg.get("rho_trans", 0.05),
                           chg.get("rho_behav", 0.05), sid="s-cache", action="read",
                           ctx_anom=0.0, t=11.0, in_baseline=chg.get("in_baseline", True),
                           center=chg.get("center", 0))
        d1 = e1.decide(changed, 1)
        ch_rows.append({"输入变化": name, "首次求值": d0.d, "变化后求值": d1.d,
                        "期望": expect, "一致": "是" if d1.d == expect else "否",
                        "约束集合": "+".join(sorted(d1.cons)) or "∅",
                        "判定依据": d1.reason})
        print("  %-24s 首次=%-7s 变化后=%-7s 期望=%-7s 一致=%s"
              % (name, d0.d, d1.d, expect, ch_rows[-1]["一致"]))
    p3 = os.path.join(HERE, "results", "E16_缓存输入变化.csv")
    # 求值上限：同一标识、四次不同输入。论文的终止规则是「任一条请求至多求值
    # 三次」，第 4 次必须按终止规则拒绝，且不得再推进档位。
    e3 = DecisionEngine(CFG, topo)
    outs = []
    for k, rs in enumerate((0.05, 0.10, 0.15, 0.20)):
        rr = make_req(topo, "柜台经纪", low_obj, 4, 0.90, rs, rs, rs,
                      sid="s-cap", action="read", ctx_anom=0.0, t=12.0)
        d = e3.decide(rr, k)
        outs.append((d.d, d.reason))
    ch_rows.append({"输入变化": "同一标识第 4 次不同输入（求值上限 3）",
                    "首次求值": outs[0][0], "变化后求值": outs[-1][0],
                    "期望": "deny", "一致": "是" if outs[-1][0] == "deny" else "否",
                    "约束集合": "∅", "判定依据": outs[-1][1]})
    print("  求值上限：四次求值 → %s；第 4 次依据＝%s"
          % ("/".join(o[0] for o in outs), outs[-1][1]))

    with open(p3, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ch_rows[0].keys()))
        wr.writeheader()
        wr.writerows(ch_rows)
    print("写入", p3)

    # ---- 同一标识、状态被其他请求推进：不得复用旧判定（第七轮审稿 R2） ----
    # 评审构造的过程：先让一条低风险请求拿到判定并进缓存；再用**四条新请求**
    # 把同一 (s, o) 的 ℓ 推到 3；最后把第一条请求原样再问一次。快照只覆盖请求
    # 自带字段时，这一问会命中旧条目、原样返回；但此时风险侧已否决，
    # 正确行为是重新求值并给出 deny。对照是把缓存清空后问同一条，两者必须一致。
    def _build_state_engine():
        e = DecisionEngine(CFG, topo)
        r0 = make_req(topo, "柜台经纪", low_obj, 4, 0.90, 0.18, 0.18, 0.18,
                      sid="s-state", action="read", ctx_anom=0.0, t=20.0)
        d0 = e.decide(r0, 0)
        trail = []
        for k in range(1, 5):
            rk = make_req(topo, "柜台经纪", low_obj, 4, 0.90, 0.55, 0.55, 0.55,
                          sid="s-state", action="read", ctx_anom=0.0, t=20.0 + k)
            dk = e.decide(rk, k)
            trail.append((k, e._state[("s-state", low_obj)]["band"], dk.d))
        return e, r0, d0, trail

    e4, r_first, d_first, trail = _build_state_engine()
    hist = e4.recorded(e4._ident(r_first))
    d_again = e4.decide(r_first, 5)
    e5, r_first2, d_first2, _ = _build_state_engine()
    e5._memo.clear()                       # 只取消旧判定缓存，历史状态保留
    d_ctrl = e5.decide(r_first2, 5)

    st_rows = [{
        "步骤": "1 首次求值（三个分量 0.18，合成 %.4f）"
                % composite_risk(0.18, 0.18, 0.18),
        "求值前ℓ": 0, "求值后ℓ": 0, "决策": d_first.d,
        "约束集合": "+".join(sorted(d_first.cons)) or "∅",
        "说明": "低风险请求，判定入缓存",
    }]
    for k, band, dk in trail:
        st_rows.append({
            "步骤": "%d 新请求（三个分量 0.55，合成 %.4f）"
                    % (k + 1, composite_risk(0.55, 0.55, 0.55)),
            "求值前ℓ": band - 1 if dk == "deny" else band,
            "求值后ℓ": band, "决策": dk,
            "约束集合": "∅" if dk == "deny" else "增强认证/权限降级",
            "说明": "推进同一 (s, o) 的档位",
        })
    st_rows.append({
        "步骤": "6 回到第 1 条请求，字段完全不变",
        "求值前ℓ": e4._state[("s-state", low_obj)]["band"],
        "求值后ℓ": e4._state[("s-state", low_obj)]["band"],
        "决策": d_again.d, "约束集合": "+".join(sorted(d_again.cons)) or "∅",
        "说明": "状态快照已变（别的请求推进了同一 (s, o)），按重新求值给出 %s（依据：%s）"
                % (d_again.d, d_again.reason),
    })
    st_rows.append({
        "步骤": "对照 相同历史状态、清空旧判定缓存",
        "求值前ℓ": 3, "求值后ℓ": 3,
        "决策": d_ctrl.d, "约束集合": "+".join(sorted(d_ctrl.cons)) or "∅",
        "说明": "与第 6 步一致则为 %s" % ("一致" if d_ctrl.d == d_again.d else "不一致"),
    })
    st_rows.append({
        "步骤": "历史查询 recorded()（不作为再次执行的授权）",
        "求值前ℓ": "—", "求值后ℓ": "—",
        "决策": hist.d if hist else "无记录",
        "约束集合": ("+".join(sorted(hist.cons)) or "∅") if hist else "∅",
        "说明": "只读当时判定，不校验当前状态",
    })
    p4 = os.path.join(HERE, "results", "E16_缓存状态变化.csv")
    with open(p4, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(st_rows[0].keys()))
        wr.writeheader()
        wr.writerows(st_rows)
    print("\n同一 (s, o) 状态被其他请求推进后，旧请求再求值：")
    for r_ in st_rows:
        print("  %-46s ℓ%s→%s  %-7s %s" % (r_["步骤"][:46], r_["求值前ℓ"],
                                          r_["求值后ℓ"], r_["决策"], r_["说明"][:30]))
    print("  第 6 步 %s / 对照 %s / 一致=%s；历史查询返回 %s"
          % (d_again.d, d_ctrl.d, "是" if d_again.d == d_ctrl.d else "否",
             hist.d if hist else "无"))
    print("写入", p4)

    # ---- 交错重试：A → B → 重试 A（第八轮审稿 R1） ----
    # 审稿构造：同一 (s, o)，角色允许、信任充足、无上下文异常，三个风险分量均为
    # 0.28（合成 0.626752，落「高」档），m=2。A、B 是不同时刻的两条新请求，最后
    # 一次是 A 的原样重试。修复前 A 重试会再走一次第 15 步，把 streak 从 1 加到 2
    # 并提前升档（ℓ 1→2，处置由「收紧约束」变成「权限降级」）。修复后 A 重试不再
    # 推进状态：ℓ 保持 1、streak 保持 1，处置按当前状态重算（仍是收紧约束）；
    # 随后一条真正的新请求 C 才把 streak 推到触发升档。
    e6 = DecisionEngine(CFG, topo)
    key = ("s-ab", low_obj)

    def _ab_req(tag, t):
        return make_req(topo, "柜台经纪", low_obj, 4, 0.90, 0.28, 0.28, 0.28,
                        sid="s-ab", action="read", ctx_anom=0.0, t=t)

    req_a = _ab_req("A", 30.0)
    req_b = _ab_req("B", 31.0)
    req_c = _ab_req("C", 32.0)
    ab_rows = []

    def _step(label, req, is_new, idx):
        before = dict(e6._state.get(key, {"band": 0, "streak": 0}))
        d = e6.decide(req, idx)
        after = e6._state[key]
        ab_rows.append({
            "调用": label, "观察标识": "s-ab/%s@%.1f" % (low_obj, req.t),
            "是否新观察": "是" if is_new else "否",
            "求值前ℓ": before["band"], "求值后ℓ": after["band"],
            "求值前streak": before["streak"], "求值后streak": after["streak"],
            "决策": d.d, "约束集合": "+".join(sorted(d.cons)) or "∅",
            "说明": d.reason,
        })
        print("  %-14s 新观察=%s  ℓ %d→%d  streak %d→%d  %-7s %s"
              % (label, ab_rows[-1]["是否新观察"], before["band"], after["band"],
                 before["streak"], after["streak"], d.d, ab_rows[-1]["约束集合"]))

    _step("A 第一次", req_a, True, 0)
    _step("B 第一次", req_b, True, 1)
    _step("A 重试（原样）", req_a, False, 2)
    _step("C 第一次", req_c, True, 3)
    p5 = os.path.join(HERE, "results", "E16_交错重试.csv")
    with open(p5, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(ab_rows[0].keys()))
        wr.writeheader()
        wr.writerows(ab_rows)
    hold = ab_rows[2]
    print("  A 重试后 ℓ=%s、streak=%s（与 B 之后相同），处置=%s；新请求 C 才是下一次观察，ℓ 升到 %s"
          % (hold["求值后ℓ"], hold["求值后streak"], hold["决策"],
             ab_rows[3]["求值后ℓ"]))
    print("写入", p5)

    # ------------------------------------------------------------------
    # 第九轮审稿 R1：同一标识下的历史快照回放（u → v → u）。
    # 缓存对每个标识只保留最近一份输入摘要，u 被 v 覆盖之后再回放 u 会被当成新
    # 观察、再推进一次状态。这是原型**已知的边界**，正文按此收窄声明；本表把
    # 它记成可复核的读数，避免文字与实现再次脱节。
    # 三条调用共用同一请求标识（主体、客体、动作、请求时刻全同），即原报文回放。
    e7 = DecisionEngine(CFG, topo)
    key7 = ("s-uvu", low_obj)

    def _uvu(rho_s):
        return make_req(topo, "柜台经纪", low_obj, 4, 0.90, rho_s, 0.28, 0.28,
                        sid="s-uvu", action="read", ctx_anom=0.0, t=30.0)

    uvu_rows = []
    for label, rho_s, idx, note in [
        ("u 第一次", 0.28, 0, "标识与快照首次出现：新观察，推进档位"),
        ("v 第一次（结构分量 0.29）", 0.29, 1, "与保留快照不同：按新观察推进"),
        ("回放原 u", 0.28, 2, "历史快照回放：最近快照已被 v 覆盖，故再计一次观察（已知边界）"),
    ]:
        before = dict(e7._state.get(key7, {"band": 0, "streak": 0}))
        d = e7.decide(_uvu(rho_s), idx)
        after = e7._state[key7]
        uvu_rows.append({
            "调用": label,
            "三分量": "0.28/0.28/0.28" if abs(rho_s - 0.28) < 1e-9 else "0.29/0.28/0.28",
            "求值前ℓ": before["band"], "求值后ℓ": after["band"],
            "求值前streak": before["streak"], "求值后streak": after["streak"],
            "决策": d.d, "约束集合": "-".join(sorted(d.cons)) or "∅",
            "说明": note,
        })
        print("  %-24s ℓ %d→%d  streak %d→%d  %-7s %s"
              % (label, before["band"], after["band"], before["streak"],
                 after["streak"], d.d, uvu_rows[-1]["约束集合"]))
    p6 = os.path.join(HERE, "results", "E16_历史快照回放.csv")
    with open(p6, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(uvu_rows[0].keys()))
        wr.writeheader()
        wr.writerows(uvu_rows)
    print("  回放原 u 后 ℓ=%s：与保留快照匹配的重试不推进，回放被覆盖的旧快照仍推进——"
          "正文按此收窄去重范围" % uvu_rows[-1]["求值后ℓ"])
    print("写入", p6)


if __name__ == "__main__":
    main()
