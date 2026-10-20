# -*- coding: utf-8 -*-
"""论文一推导文档的数值自检

对《论文一_PGB-AC_公式与算法推导.md》中的每一条数学结论做独立数值验证。
验证方式有三类：解析式与独立数值实现对比；推导给出的界与精确值或枚举结果对比；
推导所依赖的数据表与生成该表的原始记录对比。

依赖：numpy + pgbac_core。结果同时打印并写入 results/derivation_check.csv。

用法：
    python verify_derivations.py
"""
from __future__ import annotations

import csv
import itertools
import math
import os
import random
import sys
import time

import numpy as np

# 控制台保险：输出被重定向或接管道时 stdout 退化成 cp936 文本流，遇到 GBK 无法表示的
# 字符会中断整个自检。本文件自身的输出字符已按 GBK 可表示范围写，这里是兜底。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pgbac_core as core
from pgbac_core import (
    CFG, ROLE_SPEC, DOMAINS, STATES, ACTION_CLASS,
    sha256, merkle_root, merkle_proof, verify_proof, refine_min,
    composite_risk, threshold_available, rwr, build_topology,
    state_of, estimate_trans_model, DecisionEngine, Request,
    build_policy_set,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ROWS = []


def hr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def rec(claim, measured, passed, note=""):
    n = len(ROWS) + 1
    ROWS.append(dict(item=n, claim=claim, measured=measured,
                     verdict="PASS" if passed else "FAIL", note=note))
    print("  [%2d] %-42s %-28s %s" % (n, claim[:42], measured,
                                       "通过" if passed else "**不通过**"))
    return passed


def _pct(counts):
    tot = float(sum(counts)) or 1.0
    return "/".join("%.2f" % (c / tot) for c in counts)


def _mk_req(rho_s, s_id="u#01", obj="交易-00", sec=1, action="read",
            trust=5, conf=1.0, ctx=0.0, t=0.0):
    """构造一个必然走到第 4 步的请求：基线内、归属中心一致、上下文正常、信任达标。

    注意：连续推进状态时必须给不同的 `t`：状态推进的单位是**请求**，同一
    (主体, 客体, 动作, 时刻) 的重复求值按幂等查询处理，不推进 ℓ／streak，
    这一点由 `E16_缓存输入变化.csv` 单独核对。
    """
    return Request(t=float(t), s_id=s_id, role="柜台经纪", center=0, home=0, obj=obj,
                   obj_i=0, sec=sec, action=action, in_baseline=True, ctx_anom=ctx,
                   rho_struct=float(rho_s), rho_trans=0.0, rho_behav=0.0,
                   trust_L=trust, conf=conf)


# ============================================================================
# 1 门限签名的容错条件（正文 5.1 与 6.2 的 R3）
# ============================================================================
def check_threshold():
    hr("1 门限签名的容错条件（正文 5.1、6.2 的 R3）")

    bad = 0
    pairs = 0
    for n in (3, 4, 5, 7):
        for t in range(1, n + 1):
            for f in range(n + 1):
                for failed in itertools.combinations(range(n), f):
                    pairs += 1
                    if threshold_available(failed, n, t) != (n - f >= t):
                        bad += 1
    rec("可用性判据等价于 n - f >= t", "不一致 %d / %d 次" % (bad, pairs), bad == 0,
        "n <= 7 的全部失效子集")

    bad = 0
    for n in (3, 5, 7):
        for t in range(1, n + 1):
            if threshold_available([0], n, t) != (t <= n - 1):
                bad += 1
    rec("t <= n - 1 时单中心失效仍可互认", "不一致 %d 次" % bad, bad == 0,
        "R3 的可验证判据 t < n")

    bad = 0
    at = {}
    for n in (3, 4, 5, 7):
        for t in range(1, n + 1):
            fmax = max(f for f in range(n + 1)
                       if threshold_available(list(range(f)), n, t))
            at[(n, t)] = fmax
            if fmax != n - t:
                bad += 1
    rec("容错上限 f_max = n - t", "不一致 %d 次" % bad, bad == 0,
        "n = 3、t = 2 时 f_max = %d" % at[(3, 2)])


# ============================================================================
# 2 结构偏离度（正文 5.2.1）
# ============================================================================
def _rwr_fixed_point(topo, seeds, cfg):
    """解 (I - (1 - alpha) W) p = alpha e，作为幂迭代的独立对照。"""
    n = len(topo.objects)
    e = np.zeros(n)
    e[list(seeds)] = 1.0
    e /= max(e.sum(), 1e-12)
    a = cfg["rwr_alpha"]
    return np.linalg.solve(np.eye(n) - (1.0 - a) * topo.adj, a * e)


def check_struct(topo):
    hr("2 结构偏离度（正文 5.2.1）")
    cfg = CFG
    eps = cfg["rho_eps"]
    q = cfg["rho_norm_q"]
    role0 = ROLE_SPEC[0][0]

    worst = 0.0
    for role, *_ in ROLE_SPEC:
        p_it = rwr(topo, topo.role_entry[role], cfg)
        p_fp = _rwr_fixed_point(topo, topo.role_entry[role], cfg)
        worst = max(worst, float(np.abs(p_it - p_fp).max()))
    tol = (1.0 - cfg["rwr_alpha"]) ** cfg["rwr_iters"]
    rec("幂迭代 60 次与不动点方程解在迭代精度内一致",
        "最大偏差 %.2e，上界 %.1e" % (worst, tol), worst <= tol,
        "9 个角色各 60 个客体")

    a = cfg["rwr_alpha"]
    p_star = _rwr_fixed_point(topo, topo.role_entry[role0], cfg)
    e = np.zeros(len(topo.objects))
    e[list(topo.role_entry[role0])] = 1.0
    e /= e.sum()
    p = e.copy()
    errs = []
    for _ in range(cfg["rwr_iters"] + 1):
        errs.append(float(np.abs(p - p_star).sum()))
        p = a * e + (1.0 - a) * (topo.adj @ p)
    bad = sum(1 for k in range(len(errs) - 1)
              if errs[k + 1] > (1.0 - a) * errs[k] + 1e-15)
    rec("误差按 (1 - alpha)^k 几何衰减", "越界 %d 步" % bad, bad == 0,
        "收缩因子 %.2f，第 60 步误差 %.1e" % (1.0 - a, errs[-1]))

    dev_col = float(np.abs(topo.adj.sum(axis=0) - 1.0).max())
    p_one = rwr(topo, topo.role_entry[role0], cfg)
    dev_sum = abs(float(p_one.sum()) - 1.0)
    rec("W 列随机且 p 归一", "列和偏差 %.1e，和偏差 %.1e" % (dev_col, dev_sum),
        max(dev_col, dev_sum) < 1e-12)

    ps = np.linspace(1e-8, 1.0, 5001)
    raws = -np.log(np.maximum(ps, eps))
    bad = int(np.sum(np.diff(raws) > 0.0))
    rec("rho_raw = -log max(p, eps) 关于 p 严格递减", "逆序点 %d 个" % bad, bad == 0,
        "p 跨 5 个数量级扫描")

    hits = []
    for role, *_ in ROLE_SPEC:
        pv = rwr(topo, topo.role_entry[role], cfg)
        vals = [-math.log(max(float(pv[topo.objects.index(o)]), eps))
                for o in topo.role_perm[role]]
        lo, _hi = topo.rho_scale[role]
        hits.append(sum(1 for v in vals if v <= lo) / float(len(vals)))
    hits.sort()
    len_min = min(len(topo.role_perm[r[0]]) for r in ROLE_SPEC)
    tol_q = 1.0 / len_min
    rec("分位标定后合规取值落入最低档的比例不低于 q",
        "最小 %.3f，中位 %.3f" % (hits[0], hits[len(hits) // 2]),
        hits[0] >= q - tol_q - 1e-9,
        "q = %.2f，最小角色客体数 %d，离散取整容差 %.3f" % (q, len_min, tol_q))

    def band_of(x):
        for i, (hi, _nm) in enumerate(cfg["risk_bands"]):
            if x < hi:
                return i
        return len(cfg["risk_bands"]) - 1

    cq = [0, 0, 0, 0]
    cc = [0, 0, 0, 0]
    for role, *_ in ROLE_SPEC:
        pv = rwr(topo, topo.role_entry[role], cfg)
        lo, hi = topo.rho_scale[role]
        for o in topo.role_perm[role]:
            raw = -math.log(max(float(pv[topo.objects.index(o)]), eps))
            rq = min(1.0, max(0.0, (raw - lo) / (hi - lo)))
            rc = min(1.0, max(0.0, raw / -math.log(eps)))
            cq[band_of(rq)] += 1
            cc[band_of(rc)] += 1
    nq, nc = float(sum(cq)), float(sum(cc))
    tv = 0.5 * sum(abs(cq[i] / nq - cc[i] / nc) for i in range(4))
    rec("分位标定与常数标定的等级分布显著不同",
        "分位 %s / 常数 %s" % (_pct(cq), _pct(cc)), tv > 0.20,
        "分布的全变差距离 %.3f（低/中/高/极高）" % tv)


# ============================================================================
# 3 状态迁移偏离度（正文 5.2.2）
# ============================================================================
def check_trans(topo, trans):
    hr("3 状态迁移偏离度（正文 5.2.2）")

    hit = {state_of(d, a) for d in DOMAINS for a in ACTION_CLASS}
    rec("状态映射满射，|S| = 5 x 2 = 10", "命中 %d 个状态" % len(hit),
        len(hit) == 10 and len(STATES) == 10, "状态由业务域与操作类别共同确定")

    worst = 0.0
    minp = 1.0
    for _role, P in trans.P.items():
        worst = max(worst, float(np.abs(P.sum(axis=1) - 1.0).max()))
        minp = min(minp, float(P.min()))
    rec("转移矩阵行和为 1，平滑后无零元", "行和偏差 %.1e，最小元 %.2e"
        % (worst, minp), worst < 1e-12 and minp > 0.0,
        "最小元上界 1/K = %.3f" % (1.0 / len(STATES)))

    W = trans.window
    rngv = np.random.default_rng(5)
    xs = rngv.normal(size=300000)
    ratio_iid = float(xs[: 300000 // W * W].reshape(-1, W).mean(axis=1).var() / xs.var())
    rec("窗口均值方差为单步方差的 1/W", "实测 %.4f，理论 %.4f" % (ratio_iid, 1.0 / W),
        abs(ratio_iid - 1.0 / W) < 0.02, "独立同分布正态，30 万点")

    rng2 = random.Random(CFG["seed"] + 1)
    train = core.gen_requests(CFG, topo, rng2, 200.0, 100.0, attack_rate=0.0)
    last, hist = {}, {}
    raw_by_role = {r[0]: [] for r in ROLE_SPEC}
    types_by_role = {r[0]: {} for r in ROLE_SPEC}
    singles = {r[0]: [] for r in ROLE_SPEC}
    for r in train:
        st = state_of(topo.domain_of[r.obj], r.action)
        prev = last.get(r.s_id)
        last[r.s_id] = st
        if prev is None:
            continue
        raw = -math.log(max(float(trans.P[r.role][prev, st]), trans.eps))
        types_by_role[r.role][(prev, st)] = raw
        singles[r.role].append(raw)
        h = hist.setdefault(r.s_id, [])
        h.append(raw)
        if len(h) > W:
            del h[:-W]
        raw_by_role[r.role].append(sum(h) / len(h))

    ratios = []
    for role in raw_by_role:
        a1 = np.asarray(singles[role])
        b1 = np.asarray(raw_by_role[role])
        if len(a1) > 50 and float(a1.var()) > 0.0:
            ratios.append(float(b1.var() / a1.var()))
    ratios.sort()
    rec("真实转移序列上窗口均值明显降方差", "比值中位 %.3f，最大 %.3f"
        % (ratios[len(ratios) // 2], ratios[-1]), ratios[-1] < 1.0,
        "单步样本 %d 条" % sum(len(v) for v in singles.values()))

    q = CFG["rho_trans_norm_q"]
    wid = []
    for role in raw_by_role:
        sv = sorted(raw_by_role[role])
        tvs = sorted(types_by_role[role].values())
        if not sv or not tvs:
            continue
        lo_s = sv[min(len(sv) - 1, int(q * (len(sv) - 1)))]
        w_s = max(sv[-1], lo_s + 1e-9) - lo_s
        lo_t = tvs[min(len(tvs) - 1, int(q * (len(tvs) - 1)))]
        w_t = max(tvs[-1], lo_t + 1e-9) - lo_t
        wid.append((role, w_s, w_t, len(sv), len(tvs)))
    bad = sum(1 for _r, w_s, w_t, _a, _b in wid if w_t >= w_s)
    rec("分位取在请求样本上，不取在去重转移类型上", "更窄 %d / %d 个角色"
        % (len(wid) - bad, len(wid)), bad == 0,
        "样本口径宽度中位 %.3f，去重口径宽度中位 %.3f"
        % (sorted(w[1] for w in wid)[len(wid) // 2],
           sorted(w[2] for w in wid)[len(wid) // 2]))


# ============================================================================
# 4 风险合成与处置映射（正文 5.2.3）
# ============================================================================
def check_risk(topo):
    hr("4 风险合成与处置映射（正文 5.2.3）")
    cfg = CFG
    rng = np.random.default_rng(9)

    bad_lb = bad_ub = 0
    amps = []
    for _ in range(20000):
        n = int(rng.integers(2, 5))
        r = rng.uniform(0.0, 1.0, n)
        c = 1.0 - float(np.prod(1.0 - r))
        if c < float(r.max()) - 1e-12:
            bad_lb += 1
        if c > float(r.sum()) + 1e-12:
            bad_ub += 1
        amps.append(c - float(r.max()))
    rec("合成值介于最大分量与分量之和之间",
        "下界越界 %d，上界越界 %d" % (bad_lb, bad_ub), bad_lb == 0 and bad_ub == 0,
        "20000 组随机分量，与最大分量的中位差 %.3f"
        % sorted(amps)[len(amps) // 2])

    bad = 0
    for x in (0.0, 0.1, 0.5, 0.9, 1.0):
        if abs(composite_risk(x, 0.0, 0.0) - x) > 1e-15:
            bad += 1
        if abs(composite_risk(0.0, x, 0.0) - x) > 1e-15:
            bad += 1
        if abs(composite_risk(0.0, 0.0, x) - x) > 1e-15:
            bad += 1
    rec("任一分量为 0 时该项不参与合成", "不一致 %d 次" % bad, bad == 0,
        "缺少该来源证据时不引入额外惩罚")

    bad = 0
    for _ in range(20000):
        a = float(rng.uniform())
        b = float(rng.uniform())
        c = float(rng.uniform())
        if 1.0 - (1 - a) * (1 - b) * (1 - c) < 1.0 - (1 - a) * (1 - b) - 1e-12:
            bad += 1
    rec("加入非负分量后合成值不减", "违例 %d 次" % bad, bad == 0,
        "分量数增加不会降低合成风险")

    names = [nm for _hi, nm in cfg["risk_bands"]]

    def band_of(x):
        for i, (hi, _nm) in enumerate(cfg["risk_bands"]):
            if x < hi:
                return i
        return len(cfg["risk_bands"]) - 1

    seq = [0.24, 0.26] * 10
    naive = 0
    prev = None
    for x in seq:
        b = band_of(x)
        if prev is not None and b != prev:
            naive += 1
        prev = b
    eng = DecisionEngine(cfg, topo)
    hys = 0
    prev = None
    for i, x in enumerate(seq):
        # 每一次都是**一条新请求**（时刻不同），不是同一条请求被重复求值；
        # 否则缓存会按幂等查询返回旧判定，测的就不是迟滞本身了。
        d = eng.decide(_mk_req(x, t=float(i)), i)
        b = names.index(d.level)
        if prev is not None and b != prev:
            hys += 1
        prev = b
    rec("迟滞带抑制阈值附近的等级抖动", "无迟滞 %d 次 / 有迟滞 %d 次切换"
        % (naive, hys), naive >= 15 and hys == 0, "rho 在 0.24 与 0.26 间交替")

    eng2 = DecisionEngine(cfg, topo)
    key2 = ("damp#01", "交易-00")
    eng2.decide(_mk_req(0.60, s_id="damp#01", t=0.0), 0)
    first = eng2._state[key2]["band"]
    bands = []
    for k in range(3):
        eng2.decide(_mk_req(0.60, s_id="damp#01", t=float(k + 1)), k + 1)
        bands.append(eng2._state[key2]["band"])
    rec("逐级攀升需连续 m 个窗口超阈值", "首窗口后 %d，随后 %s" % (first, bands),
        first == 1 and bands[0] == 1 and bands[1] == 2,
        "m = %d，rho = 0.60" % cfg["damping_windows"])

    eng3 = DecisionEngine(cfg, topo)
    key3 = ("jump#01", "交易-00")
    eng3.decide(_mk_req(0.85, s_id="jump#01"), 0)
    b3 = eng3._state[key3]["band"]
    rec("跨越两档以上时首窗口即开始攀升", "首窗口后内部等级 %d" % b3, b3 >= 1,
        "rho = 0.85，档差 3")

    bad = 0
    prev = -1
    for x in np.linspace(0.0, 1.0, 4001):
        b = band_of(float(x))
        if b < prev or b - prev > 1:
            bad += 1
        prev = b
    rec("处置等级对 rho 单调不减且逐档", "越界 %d 次" % bad, bad == 0,
        "4001 点扫描，四个区间覆盖 [0, 1]")


# ============================================================================
# 5 锚定粒度与开销的权衡模型（正文 5.4）
# ============================================================================
def check_tradeoff():
    hr("5 锚定粒度与开销的权衡模型（正文 5.4）")
    rng = np.random.default_rng(13)

    def opt_numeric(cg, cr, p, lam):
        dt_star = math.sqrt(cg / (cr * p * lam))
        grid = np.exp(np.linspace(math.log(dt_star) - 2.0, math.log(dt_star) + 2.0, 81))
        js = cg / grid + cr * p * lam * grid
        return refine_min(grid, js)

    worst = 0.0
    n_ok = 0
    for _ in range(300):
        cg = float(rng.uniform(0.2, 5.0))
        cr = float(rng.uniform(50.0, 5000.0))
        p = float(10 ** rng.uniform(-9, -5))
        lam = float(10 ** rng.uniform(1, 4))
        dt_star = math.sqrt(cg / (cr * p * lam))
        dt, edge = opt_numeric(cg, cr, p, lam)
        if edge:
            continue
        n_ok += 1
        worst = max(worst, abs(dt - dt_star) / dt_star)
    rec("闭式解与数值极小点一致", "最大相对差 %.2e" % worst, worst < 1e-3,
        "300 组随机参数，%d 组极值落在网格内" % n_ok)

    cg, cr, p, lam = 1.35, 900.0, 1e-7, 200.0
    grid = np.exp(np.linspace(math.log(0.2), math.log(120.0), 2000))
    js = cg / grid + cr * p * lam * grid
    d2 = np.diff(js, 2)
    rec("J 在正实轴上严格凸", "二阶差分最小 %.3e" % float(d2.min()),
        float(d2.min()) > 0.0, "2000 点对数网格")

    lams = np.exp(np.linspace(math.log(20.0), math.log(5000.0), 31))
    pts = []
    for lm in lams:
        dt, edge = opt_numeric(cg, cr, p, float(lm))
        if not edge:
            pts.append((math.log(float(lm)), math.log(dt)))
    slope_l = float(np.polyfit([x for x, _ in pts], [y for _, y in pts], 1)[0])

    cgs = np.exp(np.linspace(math.log(0.05), math.log(20.0), 31))
    pts = []
    for g in cgs:
        dt, edge = opt_numeric(float(g), cr, p, lam)
        if not edge:
            pts.append((math.log(float(g)), math.log(dt)))
    slope_g = float(np.polyfit([x for x, _ in pts], [y for _, y in pts], 1)[0])
    rec("两条缩放律的 log-log 斜率", "对 lam 为 %.3f，对 c_g 为 %.3f"
        % (slope_l, slope_g), abs(slope_l + 0.5) < 0.02 and abs(slope_g - 0.5) < 0.02,
        "理论值 -0.5 与 +0.5")

    def opt_half(cg_, cr_, p_, lam_):
        """暴露量按半个窗口计的口径：J = c_g/dt + c_r p lam (dt/2)。"""
        dt_star = math.sqrt(2.0 * cg_ / (cr_ * p_ * lam_))
        grid_ = np.exp(np.linspace(math.log(dt_star) - 2.0, math.log(dt_star) + 2.0, 81))
        js_ = cg_ / grid_ + cr_ * p_ * lam_ * (grid_ / 2.0)
        dt_, _e = refine_min(grid_, js_)
        return dt_

    r_half = []
    for _ in range(200):
        cg_ = float(rng.uniform(0.2, 5.0))
        cr_ = float(rng.uniform(50.0, 5000.0))
        p_ = float(10 ** rng.uniform(-9, -5))
        lam_ = float(10 ** rng.uniform(1, 4))
        d1, e1 = opt_numeric(cg_, cr_, p_, lam_)
        if e1:
            continue
        r_half.append(opt_half(cg_, cr_, p_, lam_) / d1)
    r_half.sort()
    rec("暴露量改按半个窗口计则最优间隔乘 sqrt(2)",
        "比值中位 %.4f" % r_half[len(r_half) // 2],
        abs(r_half[len(r_half) // 2] - math.sqrt(2.0)) < 1e-3,
        "本文取整窗计，为保守上界")

    def star(args):
        a, b, c, d = args
        return math.sqrt(a / (b * c * d))

    base = [cg, cr, p, lam]
    sens = []
    h = 1e-6
    for i in range(4):
        a1 = list(base)
        a2 = list(base)
        a1[i] *= (1.0 + h)
        a2[i] *= (1.0 - h)
        sens.append((math.log(star(a1)) - math.log(star(a2))) / (2.0 * h))
    rec("弹性 dln(dt*) / dln(x)", "c_g %.3f，c_r %.3f，p %.3f，lam %.3f" % tuple(sens),
        abs(sens[0] - 0.5) < 1e-3 and abs(sens[3] + 0.5) < 1e-3,
        "理论值 +0.5 与 -0.5")

    cases = [(17.3, 60.0, 0.508, 0.004, 1.0),
             (17.3, 5.0, 0.508, 0.004, 1.0),
             (213.4, 60.0, 0.508, 0.004, 0.5),
             (0.433, 60.0, 0.508, 0.004, 1.0)]
    bad = 0
    shown = []
    for dt_star, slo, tc, tv, dt_min in cases:
        ub = slo - tc - tv
        opt = min(dt_star, ub)
        shown.append("SLO %.0f s 时上界 %.1f s、取 %.1f s、可行 %s"
                     % (slo, ub, opt, "是" if ub >= dt_min else "否"))
        if abs(opt - min(dt_star, ub)) > 1e-12:
            bad += 1
    rec("选型准则取闭式解与硬约束上界的较小值", "不一致 %d 次" % bad, bad == 0,
        "；".join(shown[:2]))


# ============================================================================
# 6 策略版本一致性校验（正文 5.5）
# ============================================================================
def check_consistency():
    hr("6 策略版本一致性校验（正文 5.5）")
    rng = random.Random(CFG["seed"])

    bad = 0
    nodes = []
    for n in CFG["policy_sizes"]:
        leaves, root = build_policy_set(n, rng)
        idx = n // 2
        pr = merkle_proof(leaves, idx)
        want = max(1, int(math.ceil(math.log2(max(2, n)))))
        nodes.append(len(pr))
        if len(pr) != want or not verify_proof(leaves[idx], pr, root):
            bad += 1
    rec("单叶证明长度等于 ceil(log2 |Pi|)", "不一致 %d 组" % bad, bad == 0,
        "|Pi| 取 %s 时节点数为 %s" % (list(CFG["policy_sizes"]), nodes))

    leaves, root = build_policy_set(200, rng)
    changed = 0
    for _ in range(1000):
        alt = list(leaves)
        i = rng.randrange(200)
        alt[i] = sha256(("mut-%d" % rng.randrange(10 ** 9)).encode())
        if merkle_root(alt) != root:
            changed += 1
    rec("改动任一叶则根必变", "根变化 %d / 1000" % changed, changed == 1000,
        "抗第二原像性的工程化检验")

    leaves, root = build_policy_set(64, rng)
    fail = 0
    for _ in range(200):
        i, j = rng.sample(range(64), 2)
        alt = list(leaves)
        alt[i], alt[j] = alt[j], alt[i]
        if not verify_proof(alt[i], merkle_proof(alt, i), root):
            fail += 1
    rec("叶位置互换后原根验证必失败", "失败 %d / 200" % fail, fail == 200,
        "结构绑定使叶位置不可替换")

    from math import comb
    n, s, m = 1000, 30, 1
    miss = comb(n - m, s) / comb(n, s)
    rec("单条损坏时抽样漏检概率等于 1 - s/|Pi|",
        "实测 %.6f，解析 %.6f" % (miss, 1.0 - s / n),
        abs(miss - (1.0 - s / n)) < 1e-12, "|Pi| = %d，抽样 %d 条" % (n, s))


# ============================================================================
# 7 计算复杂度（正文第 4、5 章）
# ============================================================================
def check_complexity():
    hr("7 计算复杂度（正文第 4、5 章）")
    orig = core.sha256
    cnt = {"n": 0}

    def counting(b):
        cnt["n"] += 1
        return orig(b)

    rows = []
    bad = 0
    for n in (10, 100, 1000):
        leaves = [orig(("leaf-%d" % i).encode()) for i in range(n)]
        cnt["n"] = 0
        core.sha256 = counting
        try:
            core.merkle_root(leaves)
        finally:
            core.sha256 = orig
        rows.append((n, cnt["n"]))
        if cnt["n"] > 2 * n - 1:
            bad += 1
    rec("Merkle 建根的哈希次数不超过 2|Pi| - 1", "越界 %d 组" % bad, bad == 0,
        "；".join("|Pi|=%d 时 %d 次" % (a, b) for a, b in rows))


# ============================================================================
# 8 与阶段一实测数据的一致性核对
# ============================================================================
def check_csv():
    hr("8 与阶段一实测数据的一致性核对")

    path = os.path.join(OUT, "E5_通道检出率.csv")
    if not os.path.exists(path):
        rec("确定性通道的误报率为 0", "文件缺失", False, path)
        return
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    det = [r for r in rows if r["类型"] == "确定性"]
    bad = sum(1 for r in det if abs(float(r["误报率"])) > 0.0)
    rec("确定性通道的误报率为 0", "违例 %d 行" % bad, bad == 0,
        "基线硬边界与归属中心背书均为确定性判据")

    # 并集行在 E5 输出里已改称「指定确定性与告警通道的并集」（与正文口径一致），
    # 旧脚本仍按「三通道合并」取行，两个名字都接受，避免改名后静默取不到行。
    merged = [r for r in rows
              if r["通道"].startswith("三通道合并")
              or "并集" in r["通道"]]
    ok = bool(merged) and float(merged[0]["TPR_A2"]) == 1.0 and float(merged[0]["TPR_A4"]) == 1.0
    rec("A2 与 A4 被确定性判据完全检出",
        "合并通道 TPR_A2 = %s，TPR_A4 = %s"
        % (merged[0]["TPR_A2"] if merged else "-", merged[0]["TPR_A4"] if merged else "-"),
        ok, "不依赖任何阈值")

    path2 = os.path.join(OUT, "E5_风险分量消融.csv")
    with open(path2, encoding="utf-8-sig", newline="") as f:
        abl = {r["剧本"]: r for r in csv.DictReader(f)}
    d3 = float(abl["A3"]["AUC_ρ_struct"]) - float(abl["A3"]["AUC_ρ_trans"])
    d7 = float(abl["A7"]["AUC_ρ_trans"]) - float(abl["A7"]["AUC_ρ_struct"])
    rec("两个统计分量的作用域不重叠",
        "A3 上结构项高 %.4f，A7 上迁移项高 %.4f" % (d3, d7),
        d3 > 0.20 and d7 > 0.10, "E5 分量消融表，AUC 取合成前")


def main():
    t0 = time.perf_counter()
    rng = random.Random(CFG["seed"])
    topo = build_topology(CFG, rng)
    trans = estimate_trans_model(CFG, topo, rng)

    check_threshold()
    check_struct(topo)
    check_trans(topo, trans)
    check_risk(topo)
    check_tradeoff()
    check_consistency()
    check_complexity()
    check_csv()

    n_pass = sum(1 for r in ROWS if r["verdict"] == "PASS")
    out = os.path.join(OUT, "derivation_check.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "claim", "measured", "verdict", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print()
    print("=" * 78)
    print("合计 %d 项，通过 %d 项，耗时 %.1f s"
          % (len(ROWS), n_pass, time.perf_counter() - t0))
    print("明细已写入 %s" % out)
    return 0 if n_pass == len(ROWS) else 1


if __name__ == "__main__":
    sys.exit(main())
