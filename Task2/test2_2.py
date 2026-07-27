# -*- coding: utf-8 -*-
"""
题目3 第二问（思考题）：用梯度下降 (Gradient Descent) 求解逆运动学
框架：PyTorch 自动微分
运行前确认已安装： pip install torch

核心思想：把 IK 当成优化问题
    Loss(q) = || P_target - P_current(q) ||^2
  只要正运动学 P_current(q) 是用可微运算写的，
  PyTorch 就能自动算出 dLoss/dq（本质上是链式法则算雅可比），
  然后沿负梯度方向迭代更新 q，让末端一步步"走"向目标。
"""

import torch

torch.set_printoptions(precision=4, sci_mode=False)

# ============================================================
# 第一部分：可微的正运动学（把第2题的 NumPy 版翻译成 PyTorch 版）
# 关键区别：所有运算必须用 torch 的函数，且不能破坏计算图
# ============================================================

# SO-101 改进型 DH 参数表: (alpha_{i-1}, a_{i-1}, d_i, theta偏置)
DH_TABLE = [
    (  0.0,                0.0,    0.0624,  0.0),
    ( -torch.pi / 2,       0.035,  0.0,    -torch.pi / 2),   # θ2 = q2 - 90°
    (  0.0,                0.116,  0.0,     0.0),
    (  0.0,                0.135,  0.0,     0.0),
    ( -torch.pi / 2,       0.0,    0.061,   0.0),
]


def dh_transform(alpha, a, theta, d):
    """
    改进型 DH 变换矩阵（PyTorch 可微版）
    注意：theta 是带梯度的张量，所以矩阵要用 torch.stack 拼，
    不能用 torch.tensor([[...]]) —— 那样会切断梯度传播！
    """
    ct, st = torch.cos(theta), torch.sin(theta)
    ca = torch.cos(torch.tensor(alpha))
    sa = torch.sin(torch.tensor(alpha))
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)

    row0 = torch.stack([ct,      -st,      zero,     zero + a])
    row1 = torch.stack([st * ca,  ct * ca, -sa + zero, zero - d * sa])
    row2 = torch.stack([st * sa,  ct * sa,  ca + zero, zero + d * ca])
    row3 = torch.stack([zero,     zero,     zero,     one])
    return torch.stack([row0, row1, row2, row3])


def forward_kinematics(q):
    """可微正运动学：q 是 shape=(5,) 的张量，返回末端位置 P (shape=(3,))"""
    T = torch.eye(4, dtype=q.dtype)
    for i, (alpha, a, d, offset) in enumerate(DH_TABLE):
        T = T @ dh_transform(alpha, a, q[i] + offset, d)
    return T[:3, 3]


# ============================================================
# 第二部分：梯度下降求解 IK
# ============================================================

def solve_ik(P_target, q_init=None, lr=0.5, max_iters=2000, tol=1e-6):
    """
    梯度下降逆运动学求解器

    P_target  : 目标末端位置，shape=(3,)
    q_init    : 迭代初值（默认零位）
    lr        : 学习率
    max_iters : 最大迭代次数
    tol       : 收敛阈值（位置误差，米）
    """
    if q_init is None:
        q_init = torch.zeros(5, dtype=torch.float64)

    # requires_grad=True：告诉 PyTorch "q 是优化变量，请对它求导"
    q = q_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([q], lr=lr)

    for it in range(max_iters):
        optimizer.zero_grad()                    # 清空上一轮的梯度
        P_current = forward_kinematics(q)        # 前向：算当前末端位置
        loss = torch.sum((P_target - P_current) ** 2)   # Loss = ||ΔP||²
        loss.backward()                          # 反向：自动微分算 dLoss/dq

        if it % 200 == 0:
            err_mm = torch.norm(P_target - P_current).item() * 1000
            grad_str = q.grad.detach().numpy().round(4)
            print(f"iter {it:4d} | 位置误差 {err_mm:9.4f} mm | 梯度 {grad_str}")

        if loss.item() < tol ** 2:               # 误差足够小，提前结束
            print(f"在第 {it} 次迭代收敛")
            break

        optimizer.step()                         # 沿梯度方向更新 q

    return q.detach()


# ============================================================
# 第三部分：主流程 —— 和第一问相同的闭环验证
# ============================================================

if __name__ == "__main__":
    # 步骤1：用正运动学生成目标（和 ikpy 那问用同一组真值，方便对比）
    q_true = torch.deg2rad(torch.tensor([30., -20., 45., -30., 60.], dtype=torch.float64))
    P_target = forward_kinematics(q_true).detach()
    print(f"[目标] P_target = {P_target.numpy()}\n")

    # 步骤2：梯度下降求解
    q_solved = solve_ik(P_target, lr=0.5)

    # 步骤3：验证
    P_check = forward_kinematics(q_solved)
    pos_err_mm = torch.norm(P_target - P_check).item() * 1000
    print(f"\n[IK解] q_solved(度) = {torch.rad2deg(q_solved).numpy().round(4)}")
    print(f"[验证] 到达位置 = {P_check.detach().numpy()}")
    print(f"[验证] 位置误差 = {pos_err_mm:.6f} 毫米")
    print("\n[观察] 注意打印出来的梯度向量：第5个分量始终是 0 ——")
    print("       wrist_roll 不影响末端位置，梯度下降对它'无能为力'，")
    print("       这就是上一问 ikpy 把 q5 留在初值不动的数学原因。")