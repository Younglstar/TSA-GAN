import pandas as pd

static_path = "./cache_encdec/final_synthetic_output/synthetic_static.csv"
temporal_path = "./cache_encdec/final_synthetic_output/synthetic_temporal.csv"
merged_path = "./cache_encdec/final_synthetic_output/synthetic_merged_long.csv"

subject_col = "dataid"
seq_col = "SEQ_POS"
chunksize = 200000

static_df = pd.read_csv(static_path)

dup_static = static_df.duplicated(subset=[subject_col]).sum()
if dup_static > 0:
    raise ValueError(f"静态表中 {subject_col} 不是唯一键，不能直接 merge。")

first_chunk = True
for chunk_id, temporal_chunk in enumerate(pd.read_csv(temporal_path, chunksize=chunksize), start=1):
    merged_chunk = temporal_chunk.merge(
        static_df,
        on=subject_col,
        how="left",
        validate="many_to_one",
    )

    if seq_col in merged_chunk.columns:
        merged_chunk = merged_chunk.sort_values([subject_col, seq_col])

    merged_chunk.to_csv(
        merged_path,
        index=False,
        mode="w" if first_chunk else "a",
        header=first_chunk,
    )
    first_chunk = False
    print(f"finished chunk {chunk_id}, shape={merged_chunk.shape}")

print(f"saved -> {merged_path}")
