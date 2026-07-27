# -*- coding: utf-8 -*-
"""
题目3 第一问：用 ikpy 求解 SO-101 逆运动学，并用正运动学验证
运行前先安装： pip install ikpy

思路（闭环验证）：
  1. 随便选一组关节角 q_true，用第2题的正运动学算出末端位置 P_target
  2. 把 P_target 交给 ikpy 反解出关节角 q_solved
  3. 把 q_solved 再喂回正运动学，看末端位置和 P_target 差多少
"""

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 第一部分：第2题的正运动学（原样复用，作为"标准答案"来验证 ikpy）
# ============================================================

def dh_transform(alpha, a, theta, d):
    """改进型 DH 相邻连杆变换矩阵"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,      -st,      0,    a],
        [st * ca,  ct * ca, -sa, -d * sa],
        [st * sa,  ct * sa,  ca,  d * ca],
        [0,        0,        0,   1],
    ])


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

DH_TABLE = [
    #  alpha_{i-1}    a_{i-1}   d_i      theta偏置
    (  0.0,           0.0,      0.0624,  0.0),
    ( -np.pi / 2,     0.035,    0.0,    -np.pi / 2),   # θ2 = q2 - 90°
    (  0.0,           0.116,    0.0,     0.0),
    (  0.0,           0.135,    0.0,     0.0),
    ( -np.pi / 2,     0.0,      0.061,   0.0),
]


def forward_kinematics(q):
    """输入 q=[q1..q5]（弧度），返回 0_T_5"""
    T = np.eye(4)
    for i, (alpha, a, d, offset) in enumerate(DH_TABLE):
        T = T @ dh_transform(alpha, a, q[i] + offset, d)
    return T


# ============================================================
# 第二部分：把 DH 表翻译成 ikpy 的 Chain
#
# ikpy 的每节 link 变换 = Trans(origin_translation)
#                       · Rot(origin_orientation)
#                       · Rot(关节轴, theta)
# 而改进型 DH 变换可以等价改写为：
#   Trans([a, -d·sinα, d·cosα]) · Rot(x, α) · Rot(z, θ)
# 正好一一对应！（θ 取"含偏置的完整关节角"，解出来后再减掉偏置还原成 q）
# ============================================================

def build_chain():
    links = [OriginLink()]  # 基座（固定，不参与求解）
    for name, (alpha, a, d, offset) in zip(JOINT_NAMES, DH_TABLE):
        links.append(URDFLink(
            name=name,
            origin_translation=[a, -d * np.sin(alpha), d * np.cos(alpha)],
            origin_orientation=[alpha, 0, 0],   # 绕 x 轴的固定扭转 α
            rotation=[0, 0, 1],                 # 关节绕本地 z 轴转动
            bounds=(-np.pi, np.pi),
        ))
    # active_links_mask：OriginLink 固定，其余5个关节可动
    return Chain(links, active_links_mask=[False] + [True] * 5)


OFFSETS = np.array([off for (_, _, _, off) in DH_TABLE])


def q_to_ikpy(q):
    """我们的关节角 q → ikpy 的关节向量（含基座0位 + 偏置）"""
    return np.concatenate([[0.0], np.asarray(q) + OFFSETS])


def ikpy_to_q(theta_full):
    """ikpy 的关节向量 → 我们的关节角 q（去掉基座位、减掉偏置）"""
    return np.asarray(theta_full[1:]) - OFFSETS


# ============================================================
# 第三部分：主流程 —— 生成目标 → IK 求解 → FK 验证
# ============================================================

if __name__ == "__main__":
    chain = build_chain()

    # --- 步骤0：交叉验证 —— ikpy 链条的 FK 必须和我们 DH 的 FK 一致 ---
    q_test = np.deg2rad([10, 20, -30, 40, -50])
    T_dh = forward_kinematics(q_test)
    T_ik = chain.forward_kinematics(q_to_ikpy(q_test))
    fk_err = np.abs(T_dh - T_ik).max()
    print(f"[交叉验证] ikpy链条FK vs DH公式FK 最大差异: {fk_err:.2e}")
    assert fk_err < 1e-9, "链条构建有误！请检查 DH → URDF 的转换"

    # --- 步骤1：用正运动学生成目标位姿 ---
    q_true = np.deg2rad([30, -20, 45, -30, 60])   # 随意设定的一组"真值"
    T_target = forward_kinematics(q_true)
    P_target = T_target[:3, 3]
    print(f"\n[目标] 由 q_true(度)={np.rad2deg(q_true)} 生成")
    print(f"[目标] 末端位置 P_target = {P_target}")

    # --- 步骤2：ikpy 求解逆运动学（位置IK） ---
    theta_solved = chain.inverse_kinematics(
        target_position=P_target,
        initial_position=q_to_ikpy(np.zeros(5)),   # 从零位开始迭代
    )
    q_solved = ikpy_to_q(theta_solved)
    print(f"\n[IK解] q_solved(度) = {np.rad2deg(q_solved)}")

    # --- 步骤3：把解喂回正运动学，验证准确性 ---
    T_check = forward_kinematics(q_solved)
    P_check = T_check[:3, 3]
    pos_err = np.linalg.norm(P_target - P_check)
    print(f"\n[验证] IK解到达的位置 P = {P_check}")
    print(f"[验证] 位置误差 = {pos_err * 1000:.4f} 毫米")

    if pos_err < 1e-3:
        print("✓ 求解成功：误差在1毫米以内")
    else:
        print("✗ 误差偏大，可尝试换初值 initial_position 或检查目标是否在工作空间内")

    # 注意：q_solved 不一定等于 q_true！
    # 5个自由度去够3维位置，解不唯一——只要末端到达同一位置就算对。
    print(f"\n[对比] q_true(度)   = {np.rad2deg(q_true)}")
    print(f"[对比] q_solved(度) = {np.rad2deg(q_solved)}")
    print("（两者不同是正常的：位置IK存在无穷多组解）")