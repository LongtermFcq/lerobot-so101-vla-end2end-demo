from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("monica/so101_bottle_to_flag_20260714_000034")

print(ds)                          # 总览：episode 数、总帧数、fps、特征
print("episodes:", ds.num_episodes)
print("frames:", ds.num_frames)

frame = ds[0]
print(frame.keys())
print("state:", frame["observation.state"].shape)
print("action:", frame["action"].shape)
print("timestamp:", frame["timestamp"])
print("episode_index:", frame["episode_index"], "frame_index:", frame["frame_index"])