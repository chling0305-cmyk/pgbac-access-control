# -*- coding: utf-8 -*-
"""冻结训练—校准—评估协议，生成 results/frozen_config.json（第五轮审稿 R2）。

问题
----
v3.9.24 的书面方法与提交代码不一致：

* 稿件称「共同训练流、三条校准流选点、评估流不参与选点」；
* 实际 `run_experiments.exp_e5` 在**评估流**（seed+3）上扫描 FPR/TPR 直接选出工作点，
  `lit11_baseline` 同样在评估流上取 Youden 极大；
* `threshold_streams` 里 θ 是写死的常量，只把四条流的结果列出来，并不构成选点流程。

本脚本把书面方法落实为实现：选点在三条校准流上完成，结果写成一个**冻结配置**，
评估脚本只读该配置，不再接触阈值选择。

数据划分（与正文方法 D 一致）
--------------------------
* 训练流：每条流各有一个只含正常请求的训练流，随机源 seed+off+18；
  即 S0/+3 用 +21、S1/+101 用 +119、S2/+202 用 +220、S3/+303 用 +321。
  这就是「每条流独立重训」——跨流的离散度因此同时覆盖请求样本与模型重估两重不确定性，
  正文按这个更强的口径表述。
* 校准流：off = 101、202、303，只用于选点。
* 评估流：off = 3，只用于报告结果，**不参与任何选点**；在它上面的扫描只作事后诊断留档。

选点规则
--------
* θ_struct：网格 0.05–0.75（步长 0.05）上「自身误报率 ≤ 0.12 的点里检出最高者」，
  三条校准流必须选出同一值，否则脚本报错退出（不许事后挑一条流）。
* θ_trans：与 θ_struct 取同一值，全篇只有一个运行参数。
* θ_scalar：文献 [11] 式单标量基线按 A1∪A3 的 Youden 极大选点，同样只在校准流上选，
  三条流须一致。

产出
----
results/frozen_config.json，含 sources、rule、values 与 cfg_digest（对除 digest 外的
规范化 JSON 求 sha256），供 exp_e5 / threshold_streams / lit11_baseline 直接读取。

用法：
    python freeze_config.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pgbac_core as C            # noqa: E402
from lit11_baseline import lit11_risk   # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "frozen_config.json")

GRID = [round(0.05 + 0.05 * i, 2) for i in range(15)]          # 0.05 .. 0.75
BUDGET = 0.12                                                   # 结构偏离度的误报预算
TRAIN_OFF = 18                                                  # 训练流相对该流随机源的偏移
EVAL_OFF = 3
CAL_OFFS = [101, 202, 303]
LAM, DUR, ATTACK = 300.0, 400.0, 0.06
SCALAR_GRID = sorted(set([round(0.02 + 0.02 * i, 2) for i in range(15)]
                         + [round(0.30 + 0.05 * i, 2) for i in range(10)]))


def build_stream(cfg, topo, off):
    """按 off 生成训练流（仅正常）→ 迁移模型 → 该 off 的请求流。"""
    trans = C.estimate_trans_model(cfg, topo, random.Random(cfg["seed"] + off + TRAIN_OFF))
    reqs = C.gen_requests(cfg, topo, random.Random(cfg["seed"] + off), LAM, DUR,
                          attack_rate=ATTACK, trans=trans)
    return trans, reqs


def struct_budget_point(reqs):
    norm = [r for r in reqs if r.label == "normal"]
    a13 = [r for r in reqs if r.label in ("A1", "A3")]
    ok = []
    for th in GRID:
        fpr = sum(1 for r in norm if r.rho_struct >= th) / max(1, len(norm))
        tpr = sum(1 for r in a13 if r.rho_struct >= th) / max(1, len(a13))
        if fpr <= BUDGET:
            ok.append((tpr, th, fpr))
    if not ok:                      # 预算内无可行点
        return None
    # 检出最高；同检出取更严的阈值
    return max(ok, key=lambda x: (x[0], x[1]))


def scalar_youden_point(reqs):
    vals = [lit11_risk(r) for r in reqs]
    norm = [v for v, r in zip(vals, reqs) if r.label == "normal"]
    a13 = [v for v, r in zip(vals, reqs) if r.label in ("A1", "A3")]
    best = None
    for th in SCALAR_GRID:
        tpr = sum(1 for v in a13 if v >= th) / max(1, len(a13))
        fpr = sum(1 for v in norm if v >= th) / max(1, len(norm))
        J = tpr - fpr
        if best is None or J > best[0]:
            best = (J, th, tpr, fpr)
    return best


def main() -> int:
    cfg = dict(C.CFG)
    topo = C.build_topology(cfg, random.Random(cfg["seed"] + 11))

    per_stream = []
    for off in CAL_OFFS:
        print("校准流 off=+%d：训练 +%d …" % (off, off + TRAIN_OFF), end="", flush=True)
        _, reqs = build_stream(cfg, topo, off)
        n_norm = sum(1 for r in reqs if r.label == "normal")
        sp = struct_budget_point(reqs)
        sl = scalar_youden_point(reqs)
        print(" 正常 %d 条 | θ_struct=%.2f (FPR=%.4f) | θ_scalar=%.2f (J=%.4f)"
              % (n_norm, sp[1], sp[2], sl[1], sl[0]))
        per_stream.append({
            "offset": off, "normal": n_norm,
            "theta_struct": sp[1], "struct_fpr": round(sp[2], 4), "struct_tpr": round(sp[0], 4),
            "theta_scalar": sl[1], "scalar_J": round(sl[0], 4),
            "scalar_tpr": round(sl[2], 4), "scalar_fpr": round(sl[3], 4),
        })

    th_s = sorted({p["theta_struct"] for p in per_stream})
    th_c = sorted({p["theta_scalar"] for p in per_stream})
    assert len(th_s) == 1, "三条校准流选出的 θ_struct 不一致：%s" % th_s
    assert len(th_c) == 1, "三条校准流选出的 θ_scalar 不一致：%s" % th_c

    conf = {
        "protocol": "train-per-stream / select-on-calibration / evaluate-on-holdout",
        "seed_base": cfg["seed"],
        "streams": {
            "evaluation": {"offset": EVAL_OFF, "train_offset": EVAL_OFF + TRAIN_OFF,
                           "used_for_selection": False},
            "calibration": [{"offset": o, "train_offset": o + TRAIN_OFF,
                             "used_for_selection": True} for o in CAL_OFFS],
        },
        "stream_params": {"lambda": LAM, "duration": DUR, "attack_rate": ATTACK},
        "rule": {
            "theta_struct": "grid 0.05-0.75 step 0.05; highest detection among points whose own "
                            "false-positive rate on the calibration stream does not exceed %g" % BUDGET,
            "theta_trans": "same value as theta_struct (single run parameter)",
            "theta_scalar": "Youden maximum of A1 union A3 on the calibration stream",
            "fpr_budget": BUDGET,
            "grid": GRID,
            "scalar_grid": SCALAR_GRID,
        },
        "values": {
            "theta_struct": th_s[0],
            "theta_trans": th_s[0],
            "theta_scalar": th_c[0],
        },
        "calibration_detail": per_stream,
        "note": "The evaluation stream (offset +3) is never used for selection. Any scan reported "
                "on it is a post-hoc diagnostic and is stored separately from these values.",
    }
    canon = json.dumps(conf, ensure_ascii=False, sort_keys=True).encode("utf-8")
    conf["cfg_digest"] = hashlib.sha256(canon).hexdigest()

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(conf, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print("\n冻结配置：%s" % OUT)
    print("  theta_struct = theta_trans = %.2f，theta_scalar = %.2f"
          % (conf["values"]["theta_struct"], conf["values"]["theta_scalar"]))
    print("  cfg_digest = %s" % conf["cfg_digest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
