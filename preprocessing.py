import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]


def load_data(path="data/spotify_dataset.csv"):
    df = pd.read_csv(path)

    use_cols = [
        "track_name",
        "artists",
        "popularity",
        "track_genre"
    ] + FEATURES

    df = df[use_cols]
    df = df.dropna()
    df = df.drop_duplicates(subset=["track_name", "artists"])
    df = df.reset_index(drop=True)

    return df


def scale_features(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    return X_scaled, scaler