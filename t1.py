import pandas as pd

path = '/home/zundong/.cache/huggingface/lerobot/dataset_0316/pick_place_teapot/data/chunk-000/episode_000001.parquet'
df = pd.read_parquet(path)

print(df.head())