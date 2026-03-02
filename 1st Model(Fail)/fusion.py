"""
Data fusion: merge internet_data + realtime_data for PatchTST.

- internet_data.csv = built by build_internet_data.py (BMKG + lettuce: weather/temp + soil moisture/pH).
- realtime_data.csv = dummy data when no live sensors; used as last input window for prediction.

Training: load_and_merge() → merged series → preprocess → train PatchTST.
Prediction: realtime_data (last INPUT_LEN rows) → PatchTST → outcome.
"""
import pandas as pd

def load_and_merge():
    internet = pd.read_csv("data/internet_data.csv")
    realtime = pd.read_csv("data/realtime_data.csv")
    merged = pd.concat([internet, realtime], ignore_index=True)
    return merged


if __name__ == "__main__":
    """Run with: python fusion.py (merges CSVs and prints summary)."""
    try:
        df = load_and_merge()
        print(f"Merged shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  (internet_data + realtime_data concatenated)")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nLast 3 rows:")
        print(df.tail(3))
        # Optional: save merged result for inspection
        out_path = "data/merged_data.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved merged data to {out_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure data/internet_data.csv and data/realtime_data.csv exist.")
        print("Run: python build_internet_data.py")
    except Exception as e:
        print(f"Error: {e}")
