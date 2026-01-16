from lerobot.common.datasets.video_utils import encode_video_frames
from pathlib import Path

imgs_dir = Path("/home/zundong/.cache/huggingface/lerobot/dataset_0405/pick_and_put_pot/images/observation.depths.cam_front/episode_000042")
video_path= Path("/home/zundong/.cache/huggingface/lerobot/dataset_0405/pick_and_put_pot/videos/chunk-000/observation.depths.cam_front/episode_000042.mp4")
fps =30
encode_video_frames(imgs_dir,video_path,fps,overwrite=True)