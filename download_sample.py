# # from huggingface_hub import hf_hub_download

# # hf_hub_download(repo_id="pretraining-playground/pythia-pile-presampled", filename="data/checkpoint_steps.parquet", repo_type="dataset", cache_dir="./sample")

# import pandas as pd
# import pyarrow.parquet as pq

# parquet_file = pq.ParquetFile("/home/bmg44/pretraining-playground/sample/datasets--pretraining-playground--pythia-pile-presampled/snapshots/58bf5d7014f48f7601ff3f26c0a6888e2bd84672/data/checkpoint_steps.parquet")

# first_batch = next(parquet_file.iter_batches())
# df = first_batch.to_pandas()
# print(parquet_file.schema)
# print(df.head(1024))