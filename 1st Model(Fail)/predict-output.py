import pandas as pd

def load_and_merge():
    internet = pd.read_csv("data/internet_data.csv")
    realtime = pd.read_csv("data/realtime_data.csv")

    # Gabungkan (historical + realtime)
    merged = pd.concat([internet, realtime], ignore_index=True)

    return merged
