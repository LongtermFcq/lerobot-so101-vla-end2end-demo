# -*- coding: utf-8 -*-
"""
题目2 编程练习：SO-101 机械臂正运动学 (Forward Kinematics)
采用改进型 DH (Modified DH / Craig's Convention)
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)  # 打印保留4位小数，不用科学计数法


def dh_transform(alpha, a, theta, d):
    """
    根据改进型 DH 参数计算相邻连杆的齐次变换矩阵 i-1_T_i

    参数:
        alpha : 连杆扭转角 α_{i-1} (弧度)
        a     : 连杆长度   a_{i-1} (米)
        theta : 关节角     θ_i     (弧度)
        d     : 连杆偏距   d_i     (米)

    公式: T = Rot(x, α) · Trans(x, a) · Rot(z, θ) · Trans(z, d)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,      -st,      0,    a],
        [st * ca,  ct * ca, -sa, -d * sa],
        [st * sa,  ct * sa,  ca,  d * ca],
        [0,        0,        0,   1],
    ])


# SO-101 的 DH 参数表：每行 = (alpha_{i-1}, a_{i-1}, d_i, theta 偏置)
# 关节角 θ_i = q_i + offset（第2关节有 -90° 的偏置）
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

DH_TABLE = [
    #  alpha_{i-1}    a_{i-1}   d_i      theta偏置
    (  0.0,           0.0,      0.0624,  0.0),          # 1 shoulder_pan
    ( -np.pi / 2,     0.035,    0.0,    -np.pi / 2),    # 2 shoulder_lift (θ2 = q2 - 90°)
    (  0.0,           0.116,    0.0,     0.0),          # 3 elbow_flex
    (  0.0,           0.135,    0.0,     0.0),          # 4 wrist_flex
    ( -np.pi / 2,     0.0,      0.061,   0.0),          # 5 wrist_roll
]


def forward_kinematics(q, verbose=True):
    """
    正运动学：输入关节角列表 q = [q1..q5] (弧度)，
    返回末端相对基座的总变换矩阵 0_T_5。
    verbose=True 时打印每一个相邻变换矩阵 i-1_T_i。
    """
    assert len(q) == len(DH_TABLE), "关节角数量必须为 5"

    T = np.eye(4)  # 从单位阵开始累乘，相当于 0_T_0

    for i, (alpha, a, d, offset) in enumerate(DH_TABLE):
        theta = q[i] + offset            # 实际关节角 = 输入角 + 偏置
        Ti = dh_transform(alpha, a, theta, d)
        T = T @ Ti                       # 连乘: 0_T_i = 0_T_{i-1} · i-1_T_i

        if verbose:
            print(f"--- {i}_T_{i+1}  ({JOINT_NAMES[i]}) ---")
            print(Ti, "\n")

    return T


if __name__ == "__main__":
    # 示例输入：全零位姿（也可以改成任意角度试试）
    q = np.deg2rad([0, 0, 0, 0, 0])   # 角度转弧度！

    T_end = forward_kinematics(q)

    print("=" * 40)
    print("末端执行器相对基座的总变换矩阵 0_T_5:")
    print(T_end, "\n")

    R = T_end[:3, :3]   # 左上 3x3：旋转姿态
    P = T_end[:3, 3]    # 右侧 3x1：位置
    print("末端旋转矩阵 R:")
    print(R, "\n")
    print(f"末端位置 P (米): x={P[0]:.4f}, y={P[1]:.4f}, z={P[2]:.4f}")