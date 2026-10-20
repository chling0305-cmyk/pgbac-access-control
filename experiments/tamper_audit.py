# -*- coding: utf-8 -*-
"""篡改审计：策略一致性校验的合法通过侧与篡改拒绝侧（论文一 算法 3 / 命题 1）。

背景（2026-10-18 第五轮审稿 R1/R6）
----------------------------------
上一版补充材料对 A5 只报了「25 次根比对一致、1000 条路径全部通过」。这证明的是
**合法输入能通过**与**计算开销**，不能支持「不一致策略被拒绝」；A6 又把这层
构造性质说成「检测率 1」。两份证据被混用了。

本脚本把两侧分开测：
  · 合法侧：同一叶集合上重复建根（跨进程同种子，等价于重复运行）与逐条路径验证；
  · 篡改侧：六类篡改各 M 次，按算法 3 第 11 步的三项判据判定是否被拒绝。

同时给出一个**判据对照**：只比根（旧写法）与「版本 ∧ 根 ∧ 叶数」（现协议）
在同一批篡改样本上的接受数。二者在「重复末项」与「版本滞后」两类上不同，
这正是本轮 R1 所指的漏洞：只比根时，3 叶的批准集合与「3 叶再加一条等于末叶」
的本地集合同根，算法会接受一个叶数不符的策略集。

输出：results/E3_篡改审计.csv
"""
import csv
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pgbac_core import merkle_root, merkle_proof, verify_proof, sha256  # noqa: E402

RES = os.path.join(HERE, "results")

TRIALS = 200          # 每类篡改的试验次数
LEGIT_ROOTS = 25      # 合法侧：重复建根次数（沿用上一版口径）
LEGIT_PROOFS = 1000   # 合法侧：逐条路径验证条数


def leaves_of(n, tag=b"policy"):
    """构造 n 条互不相同的叶（模拟策略集的规范化编码）。"""
    return [sha256(tag + b"-%d" % i) for i in range(n)]


def verdict(leaves, root_approved, n_approved, ver_local, ver_chain):
    """按算法 3 第 11 步返回 (是否接受, 仅比根时是否接受)。"""
    root_ok = merkle_root(leaves) == root_approved
    len_ok = len(leaves) == n_approved
    ver_ok = ver_local == ver_chain
    return (root_ok and len_ok and ver_ok), root_ok


def main():
    os.makedirs(RES, exist_ok=True)
    print("=" * 78)
    print("篡改审计（算法 3 / 命题 1）")
    print("=" * 78)

    # ---------------- 最小算例：3 叶 vs 4 叶（确定性，不含随机） ----------------
    print("\n[0] 填充约定歧义的最小算例（命题 1 反例）")
    a, b, c = sha256(b"x1"), sha256(b"x2"), sha256(b"x3")
    r3 = merkle_root([a, b, c])
    r4 = merkle_root([a, b, c, c])
    r4b = merkle_root([a, b, c, sha256(b"x4")])
    print("    3 叶 [x1,x2,x3]        根 %s" % r3.hex()[:16])
    print("    4 叶 [x1,x2,x3,x3]     根 %s   %s" % (
        r4.hex()[:16], "相同 ← 填充吸收" if r4 == r3 else "不同"))
    print("    4 叶 [x1,x2,x3,x4]     根 %s   %s" % (
        r4b.hex()[:16], "相同" if r4b == r3 else "不同 ← 内容不同必改根"))
    # 偶数叶数下追加一条不再同根
    e4 = merkle_root([a, b, c, sha256(b"x4")])
    e5 = merkle_root([a, b, c, sha256(b"x4"), sha256(b"x4")])
    print("    偶数参照：4 叶与 5 叶(末项自复制) %s" % ("相同" if e4 == e5 else "不同"))
    case = {
        "n3_root": r3.hex()[:16],
        "n4_pad_root": r4.hex()[:16],
        "n4_changed_root": r4b.hex()[:16],
        "same_3_4pad": r4 == r3,
        "same_4_5pad": e5 == e4,
    }

    # ---------------- 合法侧 ----------------
    print("\n[1] 合法输入侧（应全部通过）")
    n = 201                                  # 奇数叶数，与真实批次规模同量级
    L = leaves_of(n)
    root = merkle_root(L)
    ok_roots = sum(1 for _ in range(LEGIT_ROOTS) if merkle_root(L) == root)
    t0 = time.perf_counter()
    ok_proofs = 0
    for k in range(LEGIT_PROOFS):
        i = (k * 7) % n                      # 覆盖不同位置，含首末
        if verify_proof(L[i], merkle_proof(L, i), root):
            ok_proofs += 1
    dt_proof = (time.perf_counter() - t0) * 1000.0 / LEGIT_PROOFS
    print("    重复建根一致 %d/%d；逐条路径通过 %d/%d；单条验证 %.4f ms"
          % (ok_roots, LEGIT_ROOTS, ok_proofs, LEGIT_PROOFS, dt_proof))

    # ---------------- 篡改侧：六类 ----------------
    print("\n[2] 篡改输入侧（按算法 3 第 11 步判定）")
    rows = []
    for np_ in (200, 201):
        LL = leaves_of(np_)
        R = merkle_root(LL)
        nP = len(LL)
        for kind in ("修改单条", "删除尾部", "新增随机条目", "交换相邻两条",
                     "版本号滞后", "追加末项副本"):
            acc_new = acc_root = 0
            for t in range(TRIALS):
                Lp = list(LL)
                ver_local, ver_chain = "v7", "v7"
                if kind == "修改单条":
                    i = (t * 13) % nP
                    Lp[i] = sha256(b"tampered-%d-%d" % (i, t))
                elif kind == "删除尾部":
                    Lp = Lp[:-1]
                elif kind == "新增随机条目":
                    Lp = Lp + [sha256(b"extra-%d" % t)]
                elif kind == "交换相邻两条":
                    i = (t * 11) % (nP - 1)
                    Lp[i], Lp[i + 1] = Lp[i + 1], Lp[i]
                elif kind == "版本号滞后":
                    ver_local = "v6"
                elif kind == "追加末项副本":
                    Lp = Lp + [Lp[-1]]
                ok_new, ok_root = verdict(Lp, R, nP, ver_local, ver_chain)
                acc_new += 1 if ok_new else 0
                acc_root += 1 if ok_root else 0
            rows.append({
                "叶数N_P": nP, "篡改类型": kind, "试验次数": TRIALS,
                "仅比根时被接受": acc_root, "三项判据下被接受": acc_new,
                "检出数": TRIALS - acc_new,
                "检出率": round((TRIALS - acc_new) / TRIALS, 4),
            })
            print("    N_P=%-4d %-14s 仅比根接受 %3d，三项判据接受 %3d，检出率 %.4f"
                  % (nP, kind, acc_root, acc_new, (TRIALS - acc_new) / TRIALS))

    # ---------------- 落盘 ----------------
    out = os.path.join(RES, "E3_篡改审计.csv")
    with io.open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out2 = os.path.join(RES, "E3_篡改审计_汇总.csv")
    with io.open(out2, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["项目", "值", "说明"])
        w.writerow(["合法侧·重复建根", "%d/%d" % (ok_roots, LEGIT_ROOTS),
                    "同一叶集合重复建根，等价于跨运行一致性"])
        w.writerow(["合法侧·逐条路径", "%d/%d" % (ok_proofs, LEGIT_PROOFS),
                    "不同位置的 Merkle 路径验证"])
        w.writerow(["单条路径验证耗时", "%.4f ms" % dt_proof, "本机，N_P=201"])
        w.writerow(["最小算例·3叶根", case["n3_root"], ""])
        w.writerow(["最小算例·4叶末项自复制根", case["n4_pad_root"],
                    "与 3 叶同根" if case["same_3_4pad"] else "不同根"])
        w.writerow(["最小算例·4叶内容不同根", case["n4_changed_root"], "内容不同必改根"])
        w.writerow(["偶数叶数追加末项", "同根" if case["same_4_5pad"] else "不同根",
                    "等式只在 n 为奇数时成立"])
        w.writerow(["篡改样本总数", str(TRIALS * 6 * 2), "六类 × 两个叶数 × 每类 200 次"])
    print("\n写出：%s\n      %s" % (out, out2))


if __name__ == "__main__":
    main()
