import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from preprocessing import scale_features


def build_recommendation_model(df):
    X_scaled, scaler = scale_features(df)

    model = NearestNeighbors(
        n_neighbors=100,
        metric="cosine"
    )

    model.fit(X_scaled)

    return model, X_scaled


def recommend_songs(
    selected_songs,
    df,
    model,
    X_scaled,
    n_recommendations=10,
    same_genre=False
):
    if isinstance(selected_songs, str):
        selected_songs = [selected_songs]

    selected_rows = df[df["track_name"].isin(selected_songs)]

    if selected_rows.empty:
        return pd.DataFrame()

    selected_positions = selected_rows.index.tolist()

    user_profile = X_scaled[selected_positions].mean(axis=0).reshape(1, -1)

    distances, indices = model.kneighbors(
        user_profile,
        n_neighbors=100
    )

    selected_genres = selected_rows["track_genre"].unique()

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]

        if row["track_name"] in selected_songs:
            continue

        if same_genre and row["track_genre"] not in selected_genres:
            continue

        results.append({
            "track_name": row["track_name"],
            "artists": row["artists"],
            "track_genre": row["track_genre"],
            "popularity": row["popularity"],
            "similarity_score": round(1 - distance, 3)
        })

        if len(results) == n_recommendations:
            break

    return pd.DataFrame(results)