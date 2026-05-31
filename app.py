import streamlit as st
import matplotlib.pyplot as plt

from preprocessing import load_data, FEATURES
from recommendation import build_recommendation_model, recommend_songs


st.set_page_config(
    page_title="Spotify Music Recommendation System",
    layout="wide"
)

st.title("Spotify Music Recommendation System")
st.write("This app recommends songs based on Spotify audio features using KNN.")

df = load_data()
model, X_scaled = build_recommendation_model(df)

song_list = sorted(df["track_name"].unique())

selected_song = st.selectbox(
    "Choose a song:",
    song_list
)

n_recommendations = st.slider(
    "Number of recommendations:",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend Songs"):
    selected_info = df[df["track_name"] == selected_song].iloc[0]

    st.subheader("Selected Song")
    st.write(f"**Title:** {selected_info['track_name']}")
    st.write(f"**Artist:** {selected_info['artists']}")
    st.write(f"**Genre:** {selected_info['track_genre']}")
    st.write(f"**Popularity:** {selected_info['popularity']}")

    recommendations = recommend_songs(
        selected_song,
        df,
        model,
        X_scaled,
        n_recommendations
    )

    st.subheader("Recommended Songs")
    st.dataframe(recommendations)

    st.subheader("Audio Features of Selected Song")

    feature_values = selected_info[FEATURES]

    fig, ax = plt.subplots()
    ax.bar(FEATURES, feature_values)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_ylabel("Feature Value")
    st.pyplot(fig)