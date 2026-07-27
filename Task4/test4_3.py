from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("~/.cache/huggingface/lerobot/feng0724821/so101_test_record_20260721_165456").expanduser()

ds = LeRobotDataset(
    "feng0724821/so101_test_record",
    root=ROOT,
)
print(f"episodes: {ds.num_episodes}, frames: {ds.num_frames}, fps: {ds.fps}")
