# -*- coding: utf-8 -*-
"""
PGB-AC 实验内核（阶段一：无链仿真）

对应论文《多中心证券业务的区块链访问控制：信任根锚定与策略治理》第 III 章机制与第 IV 章实验方案。
本模块不依赖任何区块链节点，用可复现的随机过程模拟链的锚定行为，
用于在真实链环境搭建完成之前，先行验证：
  - III-C 锚定粒度—开销权衡模型（J(Δt) 的凸性与 Δt* 闭式解）
  - III-B 四步决策函数与结构偏离度（RWR）
  - III-B 状态迁移偏离度（主体操作状态的一阶马尔可夫链）
  - III-C 策略版本一致性校验的 O(log|Π|) 增长
  - IV-C 六类攻击的可检出性

所有参数集中在 CFG 中，便于按实际链上资源消耗重新标定。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------- 冻结的选点配置（R2）

FROZEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "frozen_config.json")


def load_frozen() -> dict:
    """读取由 freeze_config.py 生成的选点配置。

    评估脚本**只**从这里取阈值。文件缺失时直接报错退出，不回退到「在评估流上扫一遍」
    ——那正是审稿意见 R2 指出的、书面方法与提交代码不一致的那条路径。
    """
    if not os.path.exists(FROZEN_PATH):
        raise SystemExit("缺少 %s：请先运行 experiments/freeze_config.py" % FROZEN_PATH)
    with io.open(FROZEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 全局配置
CFG = {
    "seed": 20260913,

    # ---- 拓扑 ----
    "n_centers": 3,                       # 多中心数 n
    "threshold_t": 2,                     # 门限签名阈值 t（n=3 时容忍 1 个中心失效）
    "objects_per_domain": 12,             # 每个业务域的客体数
    "rw_k_hop": 2,                        # 历史可达子图 k 跳闭包
    "rwr_alpha": 0.15,                    # 带重启随机游走的重启概率
    "rwr_iters": 60,                      # 幂迭代次数
    "rwr_power": 1.2,                     # 入口权重指数（模拟入口热度差异）
    # 结构偏离度归一化的标定参数。论文 5.2.1 节给出 rho_struct = -log(max(p(o), eps))
    # 并规定「结果归一化至 [0,1]」，但未固定归一化常数；7.3 节要求分值分布按表 11
    # 的阈值区间设定。此处按角色标定：取该角色历史可达集合内 -log p 的 q 分位为
    # 下界（习惯区域边界，映射到 0），取该集合最大值为上界（最罕见客体，映射到 1）。
    "rho_eps": 1e-6,                      # 论文 5.2.1 节的平滑常数 epsilon
    "rho_norm_q": 0.90,                   # 习惯区域边界所取的分位数

    # ---- 状态迁移偏离度（论文 5.2.2 节）----
    # 状态由 (业务域, 操作类别) 确定，操作类别按是否需要高敏授权归为两类，
    # 故状态数为 5 × 2 = 10。转移矩阵由仅含正常请求的训练窗口估计并作
    # Laplace 平滑；归一化区间按角色标定，口径与 5.2.1 节的结构偏离度一致。
    "rho_trans_norm_q": 0.90,             # 转移不似然度的习惯区域边界分位数
    "trans_laplace": 1.0,                 # 转移计数的 Laplace 平滑量
    "trans_train_lam": 200.0,             # 训练窗口的请求到达率（req/s）
    "trans_train_duration": 600.0,        # 训练窗口时长（s）
    "trans_window": 5,                    # 序列似然所取的转移步数 W

    # ---- 决策 ----
    "security_min_trust": {1: 1, 2: 2, 3: 3, 4: 4},   # L_min(sec(o))
    "risk_bands": [(0.25, "低"), (0.5, "中"), (0.8, "高"), (1.01, "极高")],
    "ctx_anom_limit": 0.60,               # 上下文异常度阈值，超过则触发增强认证（第 2 步）
    "hysteresis_up": 0.50,                # 升级阈值
    "hysteresis_down": 0.40,              # 降级阈值（低于升级阈值，形成滞回带）
    "damping_windows": 2,                 # 连续 m 个窗口超阈值才继续升级

    # ---- 链上锚定成本模型（委托参数，实施时按实际资源消耗折算）----
    "c_tx_base": 1.00,                    # 单次锚定交易的固定成本
    "c_root": 0.35,                       # 每个 Merkle 根（32 字节承诺）的写入与留存成本
    "c_meta_per_1k": 0.0,                 # 批次元数据随批次规模的边际成本（默认关闭；用于检验理想化假设三）
    "t_v": 0.004,                         # 批量构建与验证时间（s）
    # 共识确认时延：取真实联盟链上的实测值，不再是委托参数。
    # 测量对象为 FISCO BCOS 2.11.1 单群组四节点 PBFT，最小出块间隔 500 ms，
    # 单笔串行提交 100 次的「提交 -> 回执」墙钟时延：均值 507.8 ms、标准差 3.7 ms、
    # P50 508 ms、P99 519 ms（见 chain/阶段二_锚定时延与吞吐实测.md）。
    # 该值与出块间隔近似满足 t_c = min_block_generation_time + 7 ms，
    # 已由 100 / 500 / 1000 ms 三档对照验证；改动出块参数时应同步重标定。
    "t_c_mean": 0.508,
    "t_c_sigma": 0.0037,

    # ---- 暴露风险模型 ----
    # p_tamper 的量纲为「单条已执行但尚未锚定的记录，在单位暴露时间内的篡改危险率（1/s）」。
    # 基准取值 1e-7 的含义是：一条记录暴露 1 秒被篡改的概率为 1e-7。
    # 该标定使 Δt* = sqrt(c_g/(c_r·p·λ)) 在 λ∈[50,800] 时落在 4~22 s，
    # 覆盖真实业务可能的锚定间隔，从而可在同一量级上比较链上成本与暴露风险。
    "p_tamper": 1.0e-7,
    "c_r": 900.0,                         # 单条记录被篡改且未及时检出时的损失（元，用于归一化）
    "c_r_by_level": {1: 120.0, 2: 400.0, 3: 1200.0, 4: 4000.0},  # 按客体等级分段（用于检验理想化假设二）
    # 稀有事件的加速抽样倍数：对泊松过程将强度放大 K 倍、计数回缩 K 倍，
    # 期望不变而方差降至 1/K，用于在有限仿真时长内得到收敛的 J_risk 估计。
    "hazard_scale": 20000,

    # ---- 一致性校验 ----
    "policy_sizes": [10, 30, 100, 300, 1000],

    # ---- 仿真规模 ----
    "duration_s": 600.0,                  # 单次仿真时长（仿真秒）
    "lambda_list": [200.0, 800.0, 3200.0],  # 请求到达率（req/s）
    "dt_grid": [0.5, 1, 2, 4, 8, 16, 30, 60],  # 锚定间隔（s）
}


# ================================================================ 工具函数
def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """标准二叉 Merkle 根；奇数层最后一个节点自复制。空集返回全零。"""
    if not leaves:
        return b"\x00" * 32
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def merkle_proof(leaves: Sequence[bytes], idx: int) -> List[Tuple[str, bytes]]:
    """返回第 idx 叶的 Merkle 路径。每项为 (位置, 兄弟哈希)，位置取 'L' 或 'R'。"""
    layer = list(leaves)
    path: List[Tuple[str, bytes]] = []
    i = idx
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        sib = i ^ 1
        path.append(("L" if sib < i else "R", layer[sib]))
        layer = [sha256(layer[k] + layer[k + 1]) for k in range(0, len(layer), 2)]
        i //= 2
    return path


def verify_proof(leaf: bytes, path: Sequence[Tuple[str, bytes]], root: bytes) -> bool:
    h = leaf
    for side, sib in path:
        h = sha256(sib + h) if side == "L" else sha256(h + sib)
    return h == root


def calibration_us(repeats: int = 5, n: int = 100000) -> float:
    """参考运算耗时（微秒/次）：对固定 64 字节输入做 n 次 SHA-256，取 repeats 轮的最小值。

    本机为混合架构 CPU（性能核 + 能效核），操作系统在持续负载之后会把进程降级调度，
    实测使同一段与实验无关的纯计算耗时增加到约 2 倍，且该档位在一次运行内保持不变。
    这会让同一张表里的不同行落在不同的档位上，进而把档位差误读成被测对象随自变量的变化。
    参考运算在同一进程、同一时刻测得，用于标定各测量块所处的档位：
    跨块比较时应以「耗时 / 参考运算耗时」作归一化。
    """
    b = b"x" * 64
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            hashlib.sha256(b).digest()
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    return best


def refine_min(dts, js) -> Tuple[float, bool]:
    """在 log Δt 空间对网格极小邻域作二次拟合，得到亚网格精度的极值位置。

    返回 (极值估计, 是否落在网格边界)。落在边界说明真实极值在网格范围之外。
    放在内核里是为了让运行器与摘要生成器共用同一实现：两处各写一套数值例程时，
    口径会在后续修改中悄悄分叉，摘要与控制台就会对同一指标给出不同的数。
    """
    i = int(np.argmin(js))
    if i == 0 or i == len(js) - 1:
        return float(dts[i]), True
    x = np.log(np.asarray(dts, dtype=float)[i - 1:i + 2])
    y = np.asarray(js, dtype=float)[i - 1:i + 2]
    a, b, _ = np.linalg.solve(np.vstack([x ** 2, x, np.ones(3)]).T, y)
    if a <= 0:
        return float(dts[i]), True
    return float(math.exp(-b / (2 * a))), False


# ================================================================ 拓扑与资源图
ROLE_SPEC = [
    # 角色, 员工数, 基线客体域（RBAC 硬边界）, 入口域, 基础信任等级
    ("柜台经纪", 40, ["交易", "行情", "客户信息"], ["交易"], 4),
    ("投资顾问", 25, ["交易", "行情", "客户信息", "风控"], ["行情"], 4),
    ("清算员", 12, ["清算", "交易", "客户信息"], ["清算"], 4),
    ("风控分析", 10, ["风控", "交易", "清算"], ["风控"], 4),
    ("合规审计", 6, ["客户信息", "交易", "清算", "风控", "行情"], ["客户信息"], 3),
    ("运维", 8, ["交易", "行情", "清算", "客户信息", "风控"], ["清算"], 3),
    ("外包运维", 6, ["行情", "清算"], ["行情"], 2),
    ("批处理服务", 4, ["清算", "风控"], ["清算"], 4),
    ("新入职经纪", 6, ["行情"], ["行情"], 1),
]

DOMAINS = ["交易", "行情", "清算", "客户信息", "风控"]

# ---------------------------------------------------------- 主体操作状态
# 论文 5.2.2 节把主体的访问操作归为离散状态，状态由 (业务域, 操作类别) 确定。
# 操作类别按是否需要高敏授权二分：读取、写入与执行属常规操作；导出触及数据外带、
# 删除触及不可逆变更，二者属高敏操作。状态数 5 × 2 = 10。
# 状态空间里没有「异常态」这一项，异常由低概率转移表示，否则判据会退化为循环定义。
ACTION_CLASS = {
    "read": "常规", "write": "常规", "execute": "常规",
    "export": "高敏", "delete": "高敏",
}
STATE_CLASSES = ["常规", "高敏"]
STATES = [(d, c) for d in DOMAINS for c in STATE_CLASSES]
STATE_INDEX = {s: i for i, s in enumerate(STATES)}


def state_of(domain: str, action: str) -> int:
    """访问操作到离散状态的映射：状态 = (业务域, 操作类别)。"""
    return STATE_INDEX[(domain, ACTION_CLASS[action])]


@dataclass
class TransModel:
    """主体操作状态的一阶马尔可夫转移模型（论文 5.2.2 节）。

    P[role] 为 K × K 行随机转移矩阵，由该角色在训练窗口内的正常访问记录估计，
    计数作 Laplace 平滑以避免真实业务中未出现过的转移被赋零概率。
    单步条件概率的方差较大，故判据取最近 window 步转移的对数似然均值，
    即文献中「以正常档案的马尔可夫链评估观测活动序列」的做法；
    窗口逐主体维护，在一主体的访问序列上滑动，不跨主体聚合。

    scale[role] = (lo, hi) 是该均值在**请求样本**上的归一化区间，
    取法与 5.2.1 节结构偏离度平行：q 分位为下界（习惯转移边界），最大值为上界。
    分位取在样本而非去重后的转移类型上，是因为 5.2.1 节要求归一化把
    该角色合规访问中常态的部分整体压入低风险区间，而「常态的部分」指请求
    分布的主体；按去重转移类型取分位会得到过窄的区间，使取值坍缩到端点。

    无前序状态时不产生转移证据，返回 0，即把分量整体让给其余两项，
    不引入与「首次访问」无关的惩罚。
    """

    P: Dict[str, np.ndarray] = field(default_factory=dict)
    scale: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    window: int = 5
    eps: float = 1e-12

    def step_raw(self, role: str, prev: Optional[int], cur: int) -> Optional[float]:
        """单步转移的对数不似然度 -log P(s'|s)。无前序状态时返回 None。"""
        P = self.P.get(role)
        if P is None or prev is None:
            return None
        return -math.log(max(float(P[prev, cur]), self.eps))

    def normalize(self, role: str, mean_raw: float) -> float:
        """窗口内序列对数不似然度的均值到 [0,1] 的按角色映射。"""
        lo, hi = self.scale.get(role, (0.0, 1e-9))
        return float(np.clip((mean_raw - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class Topology:
    """资源关系图 G_R 与角色—客体权限基线。"""
    objects: List[str] = field(default_factory=list)
    domain_of: Dict[str, str] = field(default_factory=dict)
    sec_of: Dict[str, int] = field(default_factory=dict)
    adj: Optional[np.ndarray] = None            # 行归一化邻接矩阵（列随机）
    role_perm: Dict[str, set] = field(default_factory=dict)
    role_entry: Dict[str, List[int]] = field(default_factory=dict)
    role_trust: Dict[str, int] = field(default_factory=dict)
    rho_scale: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def build_topology(cfg: dict, rng: random.Random) -> Topology:
    t = Topology()
    per = cfg["objects_per_domain"]
    for d in DOMAINS:
        for i in range(per):
            name = f"{d}-{i:02d}"
            t.objects.append(name)
            t.domain_of[name] = d
            # 安全等级：清算与客户信息偏高，行情偏低
            base = {"交易": 2, "行情": 1, "清算": 3, "客户信息": 3, "风控": 2}[d]
            lvl = min(4, max(1, base + (1 if i % 5 == 0 else 0)))
            t.sec_of[name] = lvl

    n = len(t.objects)
    idx_of = {o: i for i, o in enumerate(t.objects)}
    A = np.zeros((n, n), dtype=float)

    # 1) 业务正当关联路径（域内 + 跨域主干）
    for d in DOMAINS:
        ids = [idx_of[o] for o in t.objects if t.domain_of[o] == d]
        for k in range(len(ids)):
            for j in (k + 1, k + 2):
                if j < len(ids):
                    A[ids[k], ids[j]] = 1.0
                    A[ids[j], ids[k]] = 1.0
    trunks = [("交易", "清算"), ("清算", "客户信息"), ("客户信息", "风控"),
              ("交易", "客户信息"), ("行情", "交易"), ("风控", "清算")]
    for a, b in trunks:
        ia = [idx_of[o] for o in t.objects if t.domain_of[o] == a][:3]
        ib = [idx_of[o] for o in t.objects if t.domain_of[o] == b][:3]
        for x in ia:
            for y in ib:
                A[x, y] = 1.0
                A[y, x] = 1.0
    # 2) 少量噪声长边：使图更接近真实（存在少数跨域罕见路径）
    for _ in range(n // 4):
        x, y = rng.randrange(n), rng.randrange(n)
        if x != y:
            A[x, y] = 1.0
            A[y, x] = 1.0

    col = A.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    t.adj = A / col                      # 列随机

    # 3) 角色权限基线与入口
    for role, _cnt, domains, entries, trust in ROLE_SPEC:
        t.role_perm[role] = {o for o in t.objects if t.domain_of[o] in domains}
        t.role_entry[role] = [idx_of[o] for o in t.objects if t.domain_of[o] in entries]
        t.role_trust[role] = trust

    # 4) 结构偏离度的归一化标定
    eps = cfg["rho_eps"]
    q = cfg["rho_norm_q"]
    for role, *_ in ROLE_SPEC:
        p = rwr(t, t.role_entry[role], cfg)
        sc = sorted(-math.log(max(float(p[idx_of[o]]), eps)) for o in t.role_perm[role])
        lo = sc[min(len(sc) - 1, int(q * (len(sc) - 1)))]
        t.rho_scale[role] = (lo, max(sc[-1], lo + 1e-9))
    return t


def rwr(topology: Topology, seeds: Sequence[int], cfg: dict) -> np.ndarray:
    """带重启的随机游走（Personalized PageRank），返回稳态分布 p。"""
    n = len(topology.objects)
    e = np.zeros(n)
    e[list(seeds)] = 1.0
    e = e / max(e.sum(), 1e-12)
    alpha = cfg["rwr_alpha"]
    p = e.copy()
    W = topology.adj
    for _ in range(cfg["rwr_iters"]):
        p = alpha * e + (1.0 - alpha) * (W @ p)
    return p / max(p.sum(), 1e-12)


# ================================================================ 请求流
@dataclass
class Request:
    t: float
    s_id: str
    role: str
    center: int                  # 本次请求发起所在中心
    home: int                    # 该主体的归属中心（由工号确定性导出）
    obj: str
    obj_i: int
    sec: int
    action: str
    in_baseline: bool
    ctx_anom: float              # 上下文异常度（位置／网络域／设备一致性，属 ABAC 环境属性）
    rho_struct: float
    rho_trans: float
    rho_behav: float
    trust_L: int
    conf: float
    label: str = "normal"        # normal / A1..A6
    tampered: bool = False
    rho_struct_raw: float = 0.0  # 未标定的原始结构偏离度 -log p，用于归一化方式的消融对照

    @property
    def center_ok(self) -> bool:
        """门限签名背书校验：请求须由主体归属中心发起。

        论文 6.3 节对 A4（跨中心越界）的处置不依赖风险分值，
        而由「门限签名背书要求 + 目标中心独立校验」保证，故此处为确定性判据。
        """
        return self.center == self.home


ACTIONS = ["read", "write", "execute", "export", "delete"]
ACTION_WEIGHT = [0.62, 0.16, 0.09, 0.09, 0.04]

# ρ_behav（时序行为偏离度）属论文三的产出。在论文一的实验中它没有实现，
# 此处用一组彼此区分的 Beta 分布占位，仅用于展示「结构项单独判别」与
# 「结构项 + 行为项联合判别」的差别。真实实现由论文三提供。
RHO_BEHAV_PARAMS = {
    "normal": (1.6, 6.0),
    "A1": (2.2, 5.0),
    "A2": (3.0, 3.5),
    "A3": (3.2, 3.2),
    "A4": (2.4, 4.5),
    # A7 与正常流取同一组参数：该剧本的判别力必须完全归因于状态迁移偏离度，
    # 占位过程不得为它预置响应；参数相同也使 A7 分支与正常分支消耗的随机数一致。
    "A7": (1.6, 6.0),
}


def home_center_of(s_id: str, n_centers: int) -> int:
    """工号 → 归属中心。模拟证券机构按工号固定归属某交易中心的人员管理事实。"""
    return int(hashlib.sha256(s_id.encode()).hexdigest(), 16) % n_centers


# A7 剧本的暴露窗口占仿真时长的比例。窗口内高敏操作占比按窗口内进度线性爬升。
A7_SPAN = 0.25
A7_VICTIM_PCT = 4          # 被控主体占全部主体的百分比
A7_HI_FROM, A7_HI_TO = 0.15, 0.65   # 高敏操作占比的起点与终点


def a7_onset(s_id: str, duration: float) -> Optional[float]:
    """A7 剧本：返回该主体的被控起始时刻，非被控主体返回 None。

    判据只依赖主体身份与仿真时长，不含随机数，因此该剧本的注入不会改变
    A1~A4 与正常流所占用的随机数序列，三类样本的生成保持原样。
    """
    h = int(hashlib.sha256(s_id.encode()).hexdigest()[:8], 16)
    if (h % 100) >= A7_VICTIM_PCT:
        return None
    return ((h >> 8) % 1000) / 1000.0 * 0.5 * duration


def a7_highmix(s_id: str, k: int) -> float:
    """A7 的高敏操作抽样值，取自身份与请求序号的散列。

    该剧本由正常样本改判而来，若在动作选择时再取一次随机数，会使随机数流相对
    A7 关闭的情形多消耗一次抽样，其后所有样本整体位移，A1~A4 与正常流的样本数
    随之改变。改用散列导出的确定性量后，A7 分支与正常分支消耗的随机数完全一致，
    A1~A4 的样本与正常流的其余部分保持原样。
    """
    h = hashlib.sha256(("%s|%d" % (s_id, k)).encode()).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def gen_requests(cfg: dict, topo: Topology, rng: random.Random,
                 lam: float, duration: float,
                 attack_rate: float = 0.0,
                 trans: Optional[TransModel] = None) -> List[Request]:
    """生成证券业务请求流。

    attack_rate 为攻击请求占比（0 表示纯正常流）。四类统计可判别的攻击剧本严格按
    论文 3.4 节威胁模型构造：
      A1 凭证窃取与重放 —— 异地上下文 + 入口域内罕见路径，身份与权限看似合法；
      A2 内部人员越权   —— 访问职责范围外客体，可由第一步基线硬边界确定性拒绝；
      A3 终端被控横向移动 —— 基线内但远离入口域的跨客体关联访问；
      A4 跨中心越界     —— 由非归属中心发起，可由门限签名背书确定性拒绝。
      A7 状态迁移型低频渐进越权 —— 被控主体在暴露窗口内把操作模式持续移向
         高敏操作（导出与删除），客体仍取自该角色基线内的常见客体，
         上下文条件亦维持正常范围，故可分辨的只有状态迁移。
    正常流中含少量「合法但罕见」的路径访问，构成统计通道的真实误报来源。

    trans 为状态转移模型。为 None 时不产生转移证据（rho_trans 恒为 0），
    合成风险退化为结构项与行为项的二分量形式。这一默认值使不考察风险分量的
    实验（E1~E4）的结果与加入本分量前逐位一致，便于对照。
    """
    rwr_cache: Dict[str, np.ndarray] = {}
    for role, *_ in ROLE_SPEC:
        rwr_cache[role] = rwr(topo, topo.role_entry[role], cfg)

    eps = 1e-6
    norm = -math.log(eps)
    by_domain: Dict[str, List[str]] = {d: [] for d in DOMAINS}
    for o in topo.objects:
        by_domain[topo.domain_of[o]].append(o)

    # 每角色·每域内按 RWR 质量升序排列，低质量档即「罕见路径」
    rare_pool: Dict[Tuple[str, str], List[str]] = {}
    entry_dom: Dict[str, str] = {}
    for role, *_ in ROLE_SPEC:
        pr = rwr_cache[role]
        entry_dom[role] = topo.domain_of[topo.objects[topo.role_entry[role][0]]]
        for d in DOMAINS:
            objs = sorted(by_domain[d], key=lambda o: float(pr[topo.objects.index(o)]))
            rare_pool[(role, d)] = objs[: max(1, len(objs) // 3)]

    # 角色基线内全局最罕见的客体（按 RWR 质量取最低十分位），用于构造 A3 的跨客体关联访问
    base_rare: Dict[str, List[str]] = {}
    for role, *_ in ROLE_SPEC:
        pr = rwr_cache[role]
        objs = sorted(topo.role_perm[role],
                      key=lambda o: float(pr[topo.objects.index(o)]))
        base_rare[role] = objs[: max(1, len(objs) // 10)]

    role_domains = {r[0]: r[2] for r in ROLE_SPEC}
    roles_all = [r[0] for r in ROLE_SPEC]
    # A2 需要角色权限范围之外的域；基线覆盖全部五个域的角色（运维、合规审计）
    # 在权限模型上无法构造越权客体，故 A2 只在这些存在权限缺口的角色上取样。
    gap_roles = [r for r in roles_all if set(DOMAINS) - set(role_domains[r])]
    # A1 假设攻击者具备部分上下文伪装能力，其上下文异常度不必然超过阈值；
    # 未越过阈值的那部分只能依靠结构偏离度与行为偏离度识别。
    A1_CTX = (0.35, 0.92)

    n_req = int(lam * duration)
    out: List[Request] = []
    centers = cfg["n_centers"]
    last_state: Dict[str, int] = {}
    trans_hist: Dict[str, List[float]] = {}

    for k in range(n_req):
        t = k / lam
        is_attack = attack_rate > 0 and rng.random() < attack_rate
        label = rng.choice(["A1", "A2", "A3", "A4"]) if is_attack else "normal"
        role = rng.choice(gap_roles) if label == "A2" else rng.choice(roles_all)
        domains = role_domains[role]
        s_id = f"{role}#{rng.randrange(1, 60):02d}"

        # A7 的覆盖在角色与主体确定之后进行，且只改判本会成为正常请求的样本：
        # 既不动 A1~A4 的样本，也不改变已经消耗的随机数序列。
        # 上下文条件仍取正常分布，故该剧本不经条件通道被判出。
        onset = a7_onset(s_id, duration) if attack_rate > 0.0 else None
        if onset is not None and label == "normal" and onset <= t <= onset + A7_SPAN * duration:
            is_attack, label = True, "A7"
        home = home_center_of(s_id, centers)
        center = home
        ctx = float(np.clip(rng.betavariate(1.8, 9.0), 0, 1))   # 正常上下文

        if label == "A1":
            # 凭证窃取与重放：攻击者持有有效凭证，其访问的客体分布与该角色的
            # 正常流量完全一致，因此在结构上不可分辨；可分辨的只有上下文条件。
            dom = domains[rng.randrange(len(domains))]
            if rng.random() < 0.12:
                obj = rng.choice(rare_pool[(role, dom)])
            else:
                obj = rng.choice(by_domain[dom])
            ctx = float(rng.uniform(*A1_CTX))
        elif label == "A2":
            # 内部人员越权：直接访问职责范围外的域
            other = [d for d in DOMAINS if d not in domains]
            dom = rng.choice(other)
            obj = rng.choice(by_domain[dom])
            ctx = float(rng.uniform(0.30, 0.60))
        elif label == "A3":
            # 终端被控后的横向移动：以合规终端为跳板访问基线内最罕见的客体，
            # 形成跨客体关联访问模式
            obj = rng.choice(base_rare[role])
            dom = topo.domain_of[obj]
            ctx = float(rng.uniform(0.45, 0.75))
        elif label == "A4":
            # 跨中心越界：由非归属中心发起，目标为高敏域
            center = (home + 1) % centers
            hi = [d for d in ("清算", "客户信息") if d in domains] or ["清算"]
            dom = rng.choice(hi)
            obj = rng.choice(by_domain[dom])
            ctx = float(rng.uniform(0.30, 0.60))
        else:
            # 正常流与 A7 共用客体选择：88% 基线域内均匀访问，
            # 12% 为基线域内的合法罕见路径访问。A7 的客体分布与正常流一致，
            # 结构通道对其失明是待测结果，不是靠设定预先保证的。
            dom = domains[rng.randrange(len(domains))]
            if rng.random() < 0.12:
                obj = rng.choice(rare_pool[(role, dom)])
            else:
                obj = rng.choice(by_domain[dom])

        obj_i = topo.objects.index(obj)
        p = rwr_cache[role]
        raw = -math.log(max(float(p[obj_i]), eps))
        lo, hi = topo.rho_scale[role]
        rho_s = float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))
        a, b = RHO_BEHAV_PARAMS[label]
        rho_b = float(np.clip(rng.betavariate(a, b), 0, 1))
        act = rng.choices(ACTIONS, weights=ACTION_WEIGHT, k=1)[0]
        if is_attack and label in ("A2", "A3"):
            act = rng.choice(["export", "delete", "read"])
        if label == "A7":
            # 渐进：高敏操作占比按暴露窗口内的进度由 A7_HI_FROM 线性升到 A7_HI_TO。
            # 用散列导出的确定性量代替随机抽样，使该分支不额外消耗随机数，
            # 抽样概率与原先的权重 [1-p_hi, 0.6*p_hi, 0.4*p_hi] 一致。
            u = min(1.0, max(0.0, (t - onset) / (A7_SPAN * duration)))
            p_hi = A7_HI_FROM + (A7_HI_TO - A7_HI_FROM) * u
            x = a7_highmix(s_id, k)
            if x < p_hi * 0.6:
                act = "export"
            elif x < p_hi:
                act = "delete"
            else:
                act = "read"

        # 状态迁移偏离度（论文 5.2.2 节）：由该主体上一次请求的状态到本次状态
        # 的一阶转移概率给出。转移概率是真实生成的，不由标签设定，
        # 因此该分量对四类剧本的判别力是待测的结果而非前提。
        st = state_of(topo.domain_of[obj], act)
        if trans is None:
            rho_t = 0.0
        else:
            raw_t = trans.step_raw(role, last_state.get(s_id), st)
            last_state[s_id] = st
            if raw_t is None:
                rho_t = 0.0
            else:
                h = trans_hist.setdefault(s_id, [])
                h.append(raw_t)
                if len(h) > trans.window:
                    del h[:-trans.window]
                rho_t = trans.normalize(role, sum(h) / len(h))
        out.append(Request(
            t=t, s_id=s_id, role=role, center=center, home=home, obj=obj, obj_i=obj_i,
            sec=topo.sec_of[obj], action=act,
            in_baseline=obj in topo.role_perm[role],
            ctx_anom=ctx,
            rho_struct=rho_s, rho_trans=rho_t, rho_behav=rho_b,
            rho_struct_raw=raw,
            trust_L=topo.role_trust[role],
            conf=float(np.clip(rng.betavariate(6, 2), 0, 1)),
            label=label,
        ))
    return out


def estimate_trans_model(cfg: dict, topo: Topology, rng: random.Random) -> TransModel:
    """用仅含正常请求的训练流估计各角色的状态转移矩阵与归一化区间。

    与 5.2.1 节按角色估计 a_role / b_role 的做法平行：转移矩阵与标定区间
    都逐角色给出，使同一分量的取值可在角色之间比较。训练流不含攻击剧本，
    故攻击对转移概率的影响不会泄漏到检测阈值中。

    标定取训练流在滑动窗口下的均值序列，分位口径与 5.2.1 节的结构偏离度平行：
    结构偏离度在角色权限范围内的各客体上取分位，本分量在同一角色的各次请求上取分位。
    """
    K = len(STATES)
    train = gen_requests(cfg, topo, rng, cfg["trans_train_lam"],
                         cfg["trans_train_duration"], attack_rate=0.0)
    roles = [r[0] for r in ROLE_SPEC]
    lap = cfg["trans_laplace"]
    counts = {r: np.full((K, K), lap) for r in roles}
    last: Dict[str, int] = {}
    for r in train:
        st = state_of(topo.domain_of[r.obj], r.action)
        prev = last.get(r.s_id)
        if prev is not None:
            counts[r.role][prev, st] += 1.0
        last[r.s_id] = st

    model = TransModel()
    model.window = int(cfg["trans_window"])
    for role in roles:
        C = counts[role]
        model.P[role] = C / C.sum(axis=1, keepdims=True)

    # 第二遍按与检测时相同的滑动窗口重放训练流，取样本上的均值序列做标定。
    raw_by_role = {r: [] for r in roles}
    last2: Dict[str, int] = {}
    hist2: Dict[str, List[float]] = {}
    for r in train:
        st = state_of(topo.domain_of[r.obj], r.action)
        raw = model.step_raw(r.role, last2.get(r.s_id), st)
        last2[r.s_id] = st
        if raw is None:
            continue
        h = hist2.setdefault(r.s_id, [])
        h.append(raw)
        if len(h) > model.window:
            del h[:-model.window]
        raw_by_role[r.role].append(sum(h) / len(h))

    q = cfg["rho_trans_norm_q"]
    for role in roles:
        vals = sorted(raw_by_role[role])
        if not vals:
            model.scale[role] = (0.0, 1e-9)
        else:
            lo = vals[min(len(vals) - 1, int(q * (len(vals) - 1)))]
            model.scale[role] = (lo, max(vals[-1], lo + 1e-9))
    return model


# ================================================================ 四步决策函数
@dataclass
class Decision:
    d: str                  # permit / refine / deny
    reason: str
    rho: float
    level: str
    step: int
    # 算法 2 第 19 步合并后的附加约束集合 C ⊆ {增强认证, 权限降级}。
    # 放行的充要条件是 C = ∅；默认空集使全部旧调用点仍可构造。
    cons: frozenset = frozenset()


def composite_risk(r_s: float, r_t: float, r_b: float) -> float:
    """三类风险来源的合成（第 5.2.1、5.2.2 节）。

    结构可达性、状态迁移次序与逐请求行为残差刻画的对象互不重叠，
    因而按相互独立的失效事件作联合。任一分量取 0 时该项不参与合成，
    保证在缺少某一来源证据时不引入额外惩罚。
    """
    return 1.0 - (1.0 - r_s) * (1.0 - r_t) * (1.0 - r_b)


class DecisionEngine:
    """论文第 5.2 节四步决策函数：基线过滤 → 动态映射 → 信任分级准入 → 风险干预。"""

    #: 同一请求标识允许的「不同输入」求值次数上限。论文的终止规则是「每个约束类
    #: 每个 (s,o) 只 refine 一次，两类各至多一次 ⇒ 任一条请求至多求值三次」，
    #: 这里把它落成实现的硬上限，超限按拒绝处理。
    MAX_EVALS = 3
    #: 缓存条目上限。原实现无淘汰规则，完整 E5 结束后保留 120000 项，
    #: 不能据此声称持续服务的内存可被这轮流试验证。取 1024 的理由是去重的
    #: 实际用途——覆盖一次认证往返内的在途请求；同时把常驻内存压在几十 KB，
    #: 避免 38 万条身份留在字典里把 GC 压力带进被测路径（实测 20000 条时
    #: λ=3200 的墙钟约翻倍、尾部出现 60 ms 级停顿）。被淘汰的标识若再次出现，
    #: 按新求值处理：安全性不受影响（快照检查仍在），只是不再计入幂等查询。
    MEMO_MAX = 1024
    #: 固定策略模板集的版本标识。它随请求一起进快照：策略版本一变，
    #: 旧缓存条目自动不再被认为是同一输入。
    POLICY_VERSION = "pv-fixed-permit-1"
    # 计数随缓存条目一起保存，条目淘汰即丢失，因此 MAX_EVALS 只是**条目存活期间**
    # 的局部上限，不是跨淘汰、跨连接的全局上限；协议级的终止要由认证票据与
    # 一次性消耗承担，那部分没有实现。

    def __init__(self, cfg: dict, topo: Topology):
        self.cfg = cfg
        self.topo = topo
        self._state: Dict[Tuple[str, str], dict] = {}
        # 缓存：(请求标识) -> [输入快照, Decision, 已用求值次数]。
        # 注意：第六轮审稿 R1：原缓存键只有 (主体, 客体, 动作, 请求时刻)，命中即返回
        # 整个 Decision，**不校验**授权所依赖的输入是否仍是原来的那份。实测里
        # 基线外、跨中心、信任不足三种变化都会被旧 permit 遮住。现在的语义是：
        #   * 快照一致  → 幂等查询，原样返回，不推进 ℓ／streak；
        #   * 快照不同  → 不做「复用」，按重新求值走完整检查并计入求值次数，
        #                次数超过 MAX_EVALS 则拒绝。
        self._memo: Dict[Tuple[str, str, str, float], list] = {}
        # 注意：第七轮审稿 R2：快照只覆盖了**请求自带**的字段，没有覆盖按 (s, o)
        # 维护的 ℓ 与 streak。别的请求推进了同一 (s, o) 的状态之后，旧条目仍然
        # 「快照一致」，于是被当成同一问题原样返回——实测里那条低风险请求在
        # 状态已被推到 ℓ=3 之后仍拿到旧 permit。现在条目里再存一份**求值结束
        # 时的 (ℓ, streak) 快照**，命中还要求当前状态与之相同。
        #
        # 不另设「状态版本」字典：那会让**每一次**求值都多付一次字典查找，实测
        # 把 E1 的 P50 从 1.8 μs 抬到 2.2 μs。改为直接存状态快照后，常规路径
        # （绝大多数请求是首次求值、没有条目可命中）只在登记时多一次元组构造。
        # 早于第 15 步返回的确定性分支（基线外、非归属中心、信任不足）与
        # ℓ 无关，它们的条目把状态快照记为 None，命中只看输入快照。


    # ---- 请求标识与安全输入快照 -------------------------------------------
    @staticmethod
    def _ident(r: "Request") -> Tuple[str, str, str, float]:
        """请求标识。原型把它当作网关下发的可信标识（三元组 + 请求时刻）。"""
        return (r.s_id, r.obj, r.action, round(float(r.t), 9))

    def _digest(self, r: "Request") -> tuple:
        """决定授权结果所需的全部输入的一份快照。

        这些字段一旦不同，旧判定就不再是同一问题的答案，必须重新求值。
        取浮点原值而不是四舍五入后的值：本流的输入由确定性脚本生成，
        同一请求的同一输入逐位相等；把「接近但不相等」当成同一输入才是错的。
        """
        return (r.in_baseline, r.center, r.home, r.sec, r.trust_L, r.conf,
                r.ctx_anom, r.rho_struct, r.rho_trans, r.rho_behav,
                self.POLICY_VERSION)

    def _state_snapshot(self, r: "Request"):
        """当前 (s, o) 的 (ℓ, streak)；该二元组尚无状态时返回 (0, 0)。"""
        st = self._state.get((r.s_id, r.obj))
        return (0, 0) if st is None else (st["band"], st["streak"])

    def _band(self, rho: float) -> Tuple[str, int]:
        for i, (hi, name) in enumerate(self.cfg["risk_bands"]):
            if rho < hi:
                return name, i
        return self.cfg["risk_bands"][-1][1], len(self.cfg["risk_bands"]) - 1

    def decide(self, r: Request, step_idx: int) -> Decision:
        """对一条请求求值一次，返回处置建议。

        注意：状态推进的单位是**请求（求值步）**，不是时间窗口。原参数名 `win` 会让人
        以为 ℓ 与 streak 按墙钟窗口推进，而实现从未使用过它，也没有按时间分窗、
        没有同窗口聚合、没有「无请求窗口」的处理。第五轮审稿 R3.1 指出这一点后，
        参数改名为 `step_idx`（该请求在流中的序号，仅用于审计与复现），语义按实际
        实现写明：每来一个新请求观察一次状态转移，m 约束的是**连续请求数**，
        因此**不能**用来声称限制了单位时间内的档位切换频率。

        本方法只承担**当前重新求值**这一种语义：它按当前输入与当前状态判定
        「现在能不能执行」。要查「当时为什么这样判定」，走 recorded()，
        那条路径不校验当前状态，也不构成授权。

        去重的范围以**实现能力**为准，不外推成「任何见过的快照只计一次」。
        缓存对每个请求标识**只保留最近一份输入摘要**，不保存该标识下出现过的
        全部快照集合（第九轮审稿 R1 指出：正文原先写的「(标识, 输入快照) 只计一次」
        超出了这个能力，u → v → u 的回放会被再计一次观察）。四条分支与实现一一对应：

        * 标识首次出现 ⇒ 新观察，按第 15 步推进 ℓ／streak；
        * 标识已出现过、输入与**保留的那一份**相同、且该 (s, o) 的 ℓ／streak 仍停在
          登记值 ⇒ 幂等重复调用，原样返回，不动状态、不计求值次数；
        * 标识已出现过、输入与保留值相同、但该 (s, o) 已被别的请求推进 ⇒ 按**当前
          状态**重算（旧判定作废），不推进 ℓ／streak（同一次观察不重复累计）；
        * 标识已出现过、输入与保留值不同 ⇒ 按新观察处理并推进状态——网关正是新
          风险证据的入口，换一份输入或换一个标识都不靠「缓存是否失效」来判定。

        注意：**历史快照回放不被识别**：v 覆盖 u 的摘要之后回放 u，u 会被当成新观察
        再推进一次。区分「回放旧快照」与「新签发的证据」需要已实现接口之外的观察
        生命周期管理，原型没有做；`E16_历史快照回放.csv` 把这条边界记了下来。

        去重只在条目存活期间有效：条目被淘汰后同一标识再出现会被当作新观察，
        这一限定与 MAX_EVALS 相同。
        """
        ident = self._ident(r)
        dg = self._digest(r)
        rec = self._memo.get(ident)
        same_input = rec is not None and rec[0] == dg
        if same_input:
            if rec[1] is None or rec[1] == self._state_snapshot(r):
                # 输入快照一致，且（判定与状态有关时）(s, o) 的 ℓ 与 streak
                # 仍是登记时那一份：同一次求值的幂等重复调用，原样返回，
                # 不推进 ℓ／streak，也不计入求值次数。
                return rec[2]
            # 输入没变、但同一 (s, o) 的状态被别的请求推进过：按当前状态重算，
            # 不推进状态（同一次观察不重复累计），也不占再评估额度。
        elif rec is not None and rec[3] >= self.MAX_EVALS:
            # 换了输入、且该标识的不同输入求值次数已到上限
            rho_cap = composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)
            return self._memoize(ident, dg, rec, Decision(
                "deny", "再评估次数超限", rho_cap, self._band(rho_cap)[0], 4))
        # 本次求值是否构成一次新的风险观察
        new_obs = rec is None or not same_input

        # 步 1 准入硬边界：角色基线客体范围 + 主体归属中心背书。
        # 两项均为确定性判据，不受信任分与风险分影响，因而不会因分值漂移而失效。
        if not r.in_baseline:
            return self._memoize(ident, dg, rec, Decision("deny", "基线外客体",
                                                    composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 1),
                            st=None)
        if not r.center_ok:
            return self._memoize(ident, dg, rec, Decision("deny", "非归属中心发起，背书校验失败",
                                                    composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 1),
                            st=None)

        # 步 2 动态映射（ABAC）：安全等级、操作类型与上下文条件。
        # 这两项在论文里是**约束**而非否决（高敏操作与该上下文要求增强认证），
        # 因此并入 C 后继续求值，不再提前返回——否则这些请求会绕过第 15 步的
        # 状态累积，等于只对低风险路径检验档位语义。
        cons = set()
        via_cond = False
        if r.sec == 4 and r.action in ("delete", "export") and r.conf < 0.6:
            cons.add("增强认证")
            via_cond = True
        if r.ctx_anom >= self.cfg["ctx_anom_limit"]:
            cons.add("增强认证")
            via_cond = True

        # 步 3 信任分级准入：不足且置信度高时把「增强认证」并入约束集合，
        # **不提前返回**——请求继续进入风险分段，两侧约束在第 19 步合并。
        # 这与论文算法 2 第 7~11 步一一对应；只有 conf 不足才在此直接拒绝。
        trust_side = False
        need = self.cfg["security_min_trust"][r.sec]
        if r.trust_L < need:
            if r.conf >= 0.7:
                cons.add("增强认证")
                trust_side = True
            else:
                return self._memoize(ident, dg, rec, Decision("deny", "信任等级不足",
                                                   composite_risk(r.rho_struct, r.rho_trans, r.rho_behav), "—", 3),
                            st=None)

        # 步 4 风险干预（含迟滞与阻尼）。档位一律计算：约束集合的并集要求
        # 风险侧动作总被求出，不能因前面任何一条约束而跳过。
        rho = composite_risk(r.rho_struct, r.rho_trans, r.rho_behav)
        _name, b = self._band(rho)
        key = (r.s_id, r.obj)
        st = self._state.setdefault(key, {"band": 0, "streak": 0, "last": 0.0})

        # 第 15 步：ℓ 与 streak 的更新，逐条对应论文伪码的三个分支。
        # 注意：只有构成**新的风险观察**（(标识, 输入) 首次出现）的求值才推进；
        # 同一观察的重试不推进。
        if new_obs:
            if b > st["band"] and (rho >= self.cfg["hysteresis_up"] or b - st["band"] > 1):
                # 升级：连续 m 个窗口满足升级条件才前进一步，越级也只前进一步
                st["streak"] += 1
                if st["streak"] >= self.cfg["damping_windows"] or b - st["band"] > 1:
                    st["band"] = min(st["band"] + 1, len(self.cfg["risk_bands"]) - 1)
                    st["streak"] = 0
            elif b < st["band"] and rho <= self.cfg["hysteresis_down"]:
                # 降级
                st["band"] = b
                st["streak"] = 0
            else:
                # 其余：ℓ 不变、streak 清零（含 b > ℓ 但 ρ < h_up 且未越级）
                st["streak"] = 0

        # 第 15 步之后，动作只取自**稳定档位 ℓ**：b 只驱动 ℓ 的更新，
        # 本身不再直接给动作。否则迟滞与阻尼只会改状态标签、不改处置，
        # 越级与连续窗口确认都失去意义（与论文第 17 步「由档位给出」不符）。
        # 重试路径跳过了上面的更新，直接取当前 ℓ。
        band = st["band"]
        name = self.cfg["risk_bands"][band][1]
        # 第 17 步：风险侧动作对 (d_risk, C_risk)，只产生风险侧的一份约束，不回写 C
        if band == 3:
            return self._memoize(ident, dg, rec, Decision("deny", "干预与处置", rho, name, 4, frozenset()), st=st)
        if band == 1:
            cons.add("增强认证")          # 收紧约束
        elif band == 2:
            cons.add("权限降级")          # 权限降级

        # 第 19 步：C ← C ∪ C_risk；d ← (C = ∅ ? permit : refine)
        C = frozenset(cons)
        if not C:
            return self._memoize(ident, dg, rec, Decision("permit", "正常放行", rho, name, 4, C), st=st)
        head = "权限降级" if band == 2 else ("收紧约束" if band == 1 else "需增强认证")
        tail = "＋信任不足" if trust_side else ("＋条件通道" if via_cond else "")
        return self._memoize(ident, dg, rec, Decision("refine", head + tail, rho, name, 4, C), st=st)

    def _memoize(self, ident, dg, rec, d: "Decision", st=None) -> "Decision":
        """登记本次判定、输入快照与**求值之后**的 (s, o) 状态快照。

        登记时必须取求值结束时的状态，而不是入口处的状态：本请求自己可能
        刚刚推进了 (s, o)，取入口状态会让它的重复调用立刻失配、又被当成
        新求值再推进一次——那正是「同一请求被重复计成新观察」。

        第 15 步之前的确定性拒绝（`st=None`）与 ℓ 无关，条目把状态快照记为
        None，命中时只看输入快照，省掉一次状态查找；其余路径把调用处**已经
        持有**的状态字典 `st` 直接传进来，登记时不必再按 (s, o) 查一次。

        计数存在条目里（`rec[3]`），随条目淘汰而丢失。缓存按插入顺序淘汰
        最旧条目，上限 MEMO_MAX。计数只统计**不同输入**的求值次数（MAX_EVALS
        的口径就是「同一标识允许的不同输入」）：输入没变、只是状态被别的请求
        推进后的重试不是一次再评估，不占额度。
        """
        if rec is None:
            n = 1
        elif rec[0] != dg:
            n = rec[3] + 1
        else:
            n = rec[3]
        self._memo[ident] = [dg, None if st is None else (st["band"], st["streak"]), d, n]
        if len(self._memo) > self.MEMO_MAX:
            self._memo.pop(next(iter(self._memo)))
        return d


    # ---- 历史查询：只读当时的判定，不作为再次执行的授权 -------------------
    def recorded(self, ident) -> Optional["Decision"]:
        """按请求标识查**当时**记下的判定。

        这是一条**历史查询**路径，语义与 decide 严格分开：它不重新求值、
        不校验当前输入、不推进 ℓ／streak，返回的 Decision 只回答「当时为什么
        这样判定」。它**不能**被当作当前再次执行的授权——要授权就调 decide，
        由 decide 按当前输入与当前状态重新求值。
        """
        rec = self._memo.get(ident)
        return None if rec is None else rec[2]


# ================================================================ 链后端（模拟）
@dataclass
class AnchorRecord:
    t_submit: float
    t_confirmed: float
    n_leaves: int
    g: int
    cost: float
    root: bytes
    leaves: List[bytes] = field(default_factory=list)


class MockChain:
    """用可复现随机过程模拟联盟链的锚定交易：成本按 g 折算，确认时延服从正态分布。"""

    def __init__(self, cfg: dict, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        self.records: List[AnchorRecord] = []

    def cost_of(self, g: int, n_leaves: int) -> float:
        c = self.cfg["c_tx_base"] + (1 + g) * self.cfg["c_root"]
        if self.cfg["c_meta_per_1k"] > 0:
            c += math.ceil(n_leaves / 1000.0) * self.cfg["c_meta_per_1k"]
        return c

    def submit(self, t: float, g: int, leaves: List[bytes]) -> AnchorRecord:
        root = merkle_root(leaves)
        cost = self.cost_of(g, len(leaves))
        dt = max(0.05, self.rng.gauss(self.cfg["t_c_mean"], self.cfg["t_c_sigma"]))
        rec = AnchorRecord(t_submit=t, t_confirmed=t + dt, n_leaves=len(leaves),
                           g=g, cost=cost, root=root, leaves=leaves)
        self.records.append(rec)
        return rec


def anchor_payload_leaves(decisions: List[Tuple[str, str, int, str, float]],
                          g: int) -> List[bytes]:
    """按上链信息量 g 构造该批次的叶集合。
    g=0 仅承诺决策结果；g=1 追加策略标识与版本；g=2 再追加风险分量与属性条件摘要。"""
    leaves = []
    for (s_id, obj, ver, d, rho) in decisions:
        base = f"H:{d}|{obj}".encode()
        if g == 0:
            leaves.append(sha256(base))
        elif g == 1:
            leaves.append(sha256(base + f"|pi=read-{obj}|ver={ver}".encode()))
        else:
            leaves.append(sha256(base + f"|pi=read-{obj}|ver={ver}".encode()
                                 + f"|rho={rho:.4f}".encode()))
    return leaves


# ================================================================ 一致性校验
def build_policy_set(n: int, rng: random.Random) -> Tuple[List[bytes], bytes]:
    leaves = [sha256(f"policy-{i}-{rng.randrange(10**9)}".encode()) for i in range(n)]
    return leaves, merkle_root(leaves)


def consistency_check(leaves: List[bytes], root: bytes, sample: int) -> Dict[str, float]:
    """返回本地重算 + 抽样按条校验的耗时与证明大小。"""
    t0 = time.perf_counter()
    local_root = merkle_root(leaves)
    ok_root = (local_root == root)
    idxs = [int(i * (len(leaves) - 1) / max(1, sample - 1)) for i in range(sample)]
    proof_len = 0
    ok_all = True
    for i in idxs:
        pr = merkle_proof(leaves, i)
        proof_len += len(pr)
        if not verify_proof(leaves[i], pr, root):
            ok_all = False
    t1 = time.perf_counter()
    return {
        "root_match": float(ok_root),
        "proof_ok": float(ok_all),
        "elapsed_ms": (t1 - t0) * 1000.0,
        "proof_nodes": float(proof_len) / max(1, sample),
        "theoretical_log2": math.log2(max(2, len(leaves))),
    }


# ================================================================ 门限签名（t,n）
def threshold_available(failed_centers: Sequence[int], n: int, t: int) -> bool:
    return (n - len(failed_centers)) >= t
