import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df = df.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_daynight"]
    )
    df = df[df["confidence"] >= 0]
    df = df.dropna(subset=["frp_mw", "brightness_k"])
    df = df[df["brightness_k"].between(280, 400)]
    df = df[df["frp_mw"].between(0, 200)]
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"[data_processing] Cleaned {before} -> {after} records")
    return df

if __name__ == "__main__":
    from generate_mock_data import generate
    raw = generate()
    cleaned = clean(raw)
    cleaned.to_csv("cleaned_firms.csv", index=False)
    print("Saved cleaned_firms.csv")
