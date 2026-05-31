import pandas as pd
from sklearn.neighbors import NearestNeighbors
from preprocessing import FEATURES, load_data, scale_features


def build_recommendation_model(df):
    X_scaled, scaler = scale_features(df)

    model = NearestNeighbors(
        n_neighbors=11,
        metric="cosine"
    )

    model.fit(X_scaled)
    return model, X_scaled


def recommend_songs(song_title, df, model, X_scaled, n_recommendations=10):
    song_indices = df[df["track_name"] == song_title].index

    if len(song_indices) == 0:
        return pd.DataFrame()

    song_index = song_indices[0]

    distances, indices = model.kneighbors(
        [X_scaled[song_index]],
        n_neighbors=n_recommendations + 1
    )

    recommended_indices = indices[0][1:]

    result = df.iloc[recommended_indices][
        ["track_name", "artists", "track_genre", "popularity"]
    ].copy()

    result["similarity_score"] = 1 - distances[0][1:]

    return result