import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["frp_mw"] = pd.to_numeric(
        df["frp_mw"],
        errors="coerce",
    )

    df["brightness_k"] = pd.to_numeric(
        df["brightness_k"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
            "frp_mw",
            "brightness_k",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
            "acq_daynight",
        ]
    )

    df = df[
        df["latitude"].between(6.0, 37.5)
        & df["longitude"].between(68.0, 98.0)
    ]

    df = df[
        df["confidence"].between(0, 100)
    ]

    df = df[
        df["brightness_k"].between(280, 400)
    ]

    df = df[
        df["frp_mw"].between(0, 200)
    ]

    return df.reset_index(drop=True)
