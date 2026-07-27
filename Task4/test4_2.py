import numpy as np
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("monica/so101_bottle_to_flag_20260714_000034")

# 按 episode 分组收集 timestamp 和 state
ts_by_ep = defaultdict(list)
st_by_ep = defaultdict(list)
for i in range(ds.num_frames):
    item = ds[i]
    ep = item["episode_index"].item()
    ts_by_ep[ep].append(item["timestamp"].item())
    st_by_ep[ep].append(item["observation.state"].numpy())

for ep in sorted(ts_by_ep):
    ts = np.array(ts_by_ep[ep])
    st = np.stack(st_by_ep[ep])
    assert np.all(np.diff(ts) > 0), f"ep{ep}: timestamp 非单调"
    assert np.allclose(np.diff(ts), 1/ds.fps, atol=1e-3), f"ep{ep}: 帧间隔异常"
    assert -120 <= st.min() and st.max() <= 120, f"ep{ep}: 范围超界 [{st.min():.1f}, {st.max():.1f}]"
    jump = np.abs(np.diff(st, axis=0)).max()
    assert jump < 15, f"ep{ep}: 帧间跳变过大 {jump:.1f}"
    print(f"ep{ep}: {len(ts)} 帧, ts [{ts[0]:.3f} → {ts[-1]:.3f}], "
          f"state 范围 [{st.min():.1f}, {st.max():.1f}], 最大帧间跳变 {jump:.2f} ✓")

print("质量检查通过")