import os
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

HF_DATA_REPO = "henribonamy/chess-puzzles-data"
OUTPUT_PATH = "data/high_rated_indices.npy"
RATING_THRESHOLD = 1500


def main() -> None:
    """Stream Lichess puzzle dataset, collect indices with Rating > threshold, upload to HF Hub."""
    os.makedirs("data", exist_ok=True)

    print(f"Streaming Lichess/chess-puzzles, collecting Rating > {RATING_THRESHOLD} indices...")
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)

    high_rated: list[int] = []
    for i, row in enumerate(tqdm(ds, desc="Scanning")):
        if row["Rating"] > RATING_THRESHOLD:
            high_rated.append(i)
        if i % 500_000 == 0 and i > 0:
            print(f"  {i:,} rows scanned, {len(high_rated):,} high-rated so far")

    arr = np.array(high_rated, dtype=np.int32)
    np.save(OUTPUT_PATH, arr)
    print(f"Saved {len(arr):,} indices to {OUTPUT_PATH}")

    print(f"Uploading to {HF_DATA_REPO}...")
    api = HfApi()
    api.create_repo(HF_DATA_REPO, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=OUTPUT_PATH,
        path_in_repo="high_rated_indices.npy",
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
