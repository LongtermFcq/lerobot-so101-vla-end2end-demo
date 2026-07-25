#!/usr/bin/env python3
"""
题目1:你好,机械臂 (Hello Robot) —— Home 复位脚本
====================================================
功能:让 SO-101 从臂从当前(未知)姿态,平滑插值运动到指定姿态,并校验是否到位。

用法(先 conda activate lerobot):
    python homing.py                # 默认:移动到 Home 姿态
    python homing.py --pose zero    # 移动到全 0 姿态(查看校准中心位置)
    python homing.py --pose custom  # 移动到自定义姿态(可自行修改角度)

    可选参数:
    python homing.py --pose home --duration 5.0 --steps 100   # 更慢更平滑
"""

import argparse
import time

# ------------------------------------------------------------------
# 兼容不同版本 lerobot 的导入(新版类名为 SOFollower,旧版为 SO101Follower)
# ------------------------------------------------------------------
try:
    from lerobot.robots.so_follower import SO101Follower as Follower
    from lerobot.robots.so_follower import SO101FollowerConfig as FollowerConfig
except ImportError:
    from lerobot.robots.so101_follower import SO101Follower as Follower
    from lerobot.robots.so101_follower import SO101FollowerConfig as FollowerConfig

# ------------------------------------------------------------------
# 你的机械臂信息(与校准时保持一致!)
# ------------------------------------------------------------------
PORT = "/dev/tty.usbmodem5B3D0458541"
ROBOT_ID = "monica_follower_arm"

# ------------------------------------------------------------------
# 姿态定义(单位:度)
# ------------------------------------------------------------------
POSES = {
    # 标准 Home 姿态(作业示例给出的安全待机姿势)
    "home": {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": -20,
        "elbow_flex.pos": 40,
        "wrist_flex.pos": 0,
        "wrist_roll.pos": 30,
        "gripper.pos": 40,
    },
    # 全 0 姿态:查看校准定义的"中心位置"
    "zero": {
        "shoulder_pan.pos": 0,
        "shoulder_lift.pos": 0,
        "elbow_flex.pos": 0,
        "wrist_flex.pos": 0,
        "wrist_roll.pos": 0,
        "gripper.pos": 0,
    },
    # 自定义姿态:作业要求"尝试新姿态",可修改下面的角度
    # 修改前想清楚:会不会撞桌子?角度是否在校准范围内?
    "custom": {
        "shoulder_pan.pos": 20,
        "shoulder_lift.pos": -30,
        "elbow_flex.pos": 60,
        "wrist_flex.pos": -20,
        "wrist_roll.pos": 0,
        "gripper.pos": 10,
    },
}

POS_TOL_DEG = 3.0  # 到位容差(度):实际角度与目标相差小于此值即视为到位
SETTLE_SEC = 1.0   # 运动结束后等待电机稳定的时间(秒)
FINE_TUNE_RETRIES = 3  # 超差时补发目标位置的最大次数


def move_smooth(arm, target: dict, duration: float, steps: int):
    """
    平滑插值运动:把"当前位置 -> 目标位置"的路程切成 steps 小步,
    每隔 duration/steps 秒发送一步,实现匀速平滑运动。
    """
    # 1. 读取当前姿态(插值的起点)
    obs = arm.get_observation()
    start = {k: obs[k] for k in target}

    print("\n当前姿态(起点):")
    for k, v in start.items():
        print(f"  {k:22s} {v:8.2f}°  ->  目标 {target[k]:.2f}°")

    # 2. 逐步插值:第 i 步的目标 = 起点 + (终点-起点) * i/steps
    dt = duration / steps
    print(f"\n开始平滑运动:{duration:.1f} 秒,{steps} 步,每步 {dt*1000:.0f} ms ...")
    for i in range(1, steps + 1):
        alpha = i / steps  # 插值系数:0 -> 1
        action = {k: start[k] + (target[k] - start[k]) * alpha for k in target}
        arm.send_action(action)
        time.sleep(dt)
    print("运动指令发送完毕。")


def verify(arm, target: dict, tol: float, settle: float = SETTLE_SEC) -> bool:
    """到位校验:读回实际角度,逐关节与目标比较。"""
    time.sleep(settle)
    obs = arm.get_observation()

    print(f"\n到位校验(容差 ±{tol}°):")
    all_ok = True
    for k, goal in target.items():
        actual = obs[k]
        err = actual - goal
        ok = abs(err) <= tol
        all_ok &= ok
        mark = "OK " if ok else "FAIL"
        print(f"  [{mark}] {k:22s} 目标 {goal:8.2f}°  实际 {actual:8.2f}°  误差 {err:+.2f}°")
    return all_ok


def fine_tune(arm, target: dict, tol: float, max_retries: int = FINE_TUNE_RETRIES) -> bool:
    """对未达标的关节补发精确目标,最多重试 max_retries 次。"""
    for attempt in range(1, max_retries + 1):
        if verify(arm, target, tol):
            if attempt > 1:
                print(f"补到位成功(第 {attempt} 次)。")
            return True
        if attempt < max_retries:
            print(f"部分关节超差,补发目标位置({attempt}/{max_retries})...")
            arm.send_action(target)
    return False


def main():
    parser = argparse.ArgumentParser(description="SO-101 从臂姿态复位脚本")
    parser.add_argument("--pose", choices=list(POSES.keys()), default="home",
                        help="目标姿态:home(默认)/ zero(全0)/ custom(自定义)")
    parser.add_argument("--duration", type=float, default=3.0, help="运动总时长(秒)")
    parser.add_argument("--steps", type=int, default=60, help="插值步数")
    args = parser.parse_args()

    target = POSES[args.pose]
    print(f"目标姿态:{args.pose}")

    # 1. 配置并连接机械臂
    cfg = FollowerConfig(
        port=PORT,
        id=ROBOT_ID,           # 与校准时相同的 id,自动加载校准文件
        use_degrees=True,      # 用"度"作为角度单位
        max_relative_target=15.0,  # 安全限速:单步目标不得偏离当前位置超过 15°
    )
    arm = Follower(cfg)
    print(f"连接机械臂 {ROBOT_ID} @ {PORT} ...")
    arm.connect(calibrate=False)  # 已校准过,直接加载本地校准文件
    print("连接成功。")

    try:
        # 2. 平滑运动到目标姿态
        move_smooth(arm, target, duration=args.duration, steps=args.steps)

        # 3. 校验并补到位
        if fine_tune(arm, target, POS_TOL_DEG):
            print("\n✅ 复位成功:所有关节均在容差范围内。")
        else:
            print("\n⚠️ 部分关节未到位:可能被遮挡、负载过大或目标超出校准范围。")

        # 4. 断开前提示:断开时电机会卸力,机械臂会软下来
        input("\n按回车断开连接(断开后电机卸力,请确认机械臂处于安全低位)...")
    finally:
        arm.disconnect()
        print("已断开连接。")


if __name__ == "__main__":
    main()