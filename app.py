import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from preprocessing import load_data, FEATURES
from recommendation import build_recommendation_model, recommend_songs
from streamlit_option_menu import option_menu


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spotify Music Intelligence",
    page_icon="🎧",
    layout="wide"
)


# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    div[data-testid="stAppViewContainer"] {
        background:linear-gradient(
                180deg,
                #0F172A 0%,
                #111827 40%,
                #0B1120 100%
        );
    }

    div[data-testid="stHeader"] {
        background-color: #0E1117;
    }

    h1, h2, h3 {
        color: #F9FAFB !important;
        font-weight: 800;
    }

    p, label, span {
        color: #D1D5DB;
    }

    div[data-testid="stMarkdownContainer"] {
    color: #F9FAFB !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        color: #D1D5DB !important;
    }

    div[data-testid="stMarkdownContainer"] li {
        color: #D1D5DB !important;
    }

    div[data-testid="stMarkdownContainer"] strong {
        color: #FFFFFF !important;
    }

    .top-header {
        background-color: #191414;
        padding: 28px 42px;
        border-radius: 0px 0px 28px 28px;
        margin-bottom: 22px;
    }

    .brand-title {
        font-size: 44px;
        font-weight: 900;
        color: #1DB954;
        margin-bottom: 4px;
        letter-spacing: 1px;
    }

    .brand-subtitle {
        font-size: 19px;
        color: #D1D5DB;
        margin-bottom: 0px;
    }

    .hero-box {
        background: linear-gradient(135deg, #1DB954 0%, #1E3A8A 45%, #7C3AED 100%);
        padding: 55px;
        border-radius: 30px;
        margin-bottom: 35px;
        color: white;
    }

    .hero-title {
        font-size: 58px;
        font-weight: 900;
        margin-bottom: 10px;
        color: white;
    }

    .hero-subtitle {
        font-size: 22px;
        color: #E5E7EB;
        max-width: 850px;
        line-height: 1.6;
    }

    .song-title {
        font-size: 28px;
        font-weight: 900;
        color: #FFFFFF;
        margin-bottom: 8px;
    }

    .song-meta {
        font-size: 16px;
        color: #D1D5DB;
        line-height: 1.7;
    }

    .tag {
        background: linear-gradient(
            135deg,
            #1DB954 0%,
            #22C55E 100%
        );
        color:white;
        padding:10px 16px;
        border-radius:999px;
        display:inline-block;
        margin:6px;
        font-weight:700;
        font-size:14px;
        box-shadow: 0 4px 12px rgba(29,185,84,0.25);
        transition: 0.2s ease;
    }

    .tag:hover {
        transform: scale(1.05);
    }

    .match-badge {
        background: linear-gradient(
            135deg,
            #1DB954 0%,
            #22C55E 100%
        );
        color:white;
        padding:6px 14px;
        border-radius:999px;
        display:inline-block;
        font-weight:800;
        margin-top:10px;
        box-shadow: 0 4px 12px rgba(29,185,84,0.4);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        backdrop-filter: blur(12px);
        transition: 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        border: 1px solid #1DB954;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #1DB954 0%,
            #1ED760 100%
        );
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 700;
        padding: 10px 18px;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# DATA FUNCTIONS
# =========================
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_recommender(df):
    return build_recommendation_model(df)


@st.cache_data
def evaluate_knn(df):
    top_genres = df["track_genre"].value_counts().head(8).index
    eval_df = df[df["track_genre"].isin(top_genres)].copy()

    X = eval_df[FEATURES]
    y = eval_df["track_genre"]

    scaler = StandardScaler()
    X_scaled_eval = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_eval,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    k_values = [3, 5, 7, 9, 11, 15, 21, 31]
    results = []

    best_k = None
    best_acc = 0
    best_pred = None

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = accuracy_score(y_test, pred)

        results.append({"k": k, "accuracy": acc})

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_pred = pred

    cm = confusion_matrix(y_test, best_pred, labels=top_genres)

    return pd.DataFrame(results), cm, list(top_genres), best_k, best_acc


@st.cache_data
def normalize_features_for_visualization(df):
    visual_df = df.copy()

    for feature in FEATURES:
        min_value = visual_df[feature].min()
        max_value = visual_df[feature].max()

        visual_df[feature] = (
            visual_df[feature] - min_value
        ) / (
            max_value - min_value
        )

    return visual_df


# =========================
# LOAD DATA
# =========================
df = get_data()
model, X_scaled = get_recommender(df)
visual_df = normalize_features_for_visualization(df)

song_display_df = df.copy()
song_display_df["song_label"] = (
    song_display_df["track_name"].astype(str)
    + " — "
    + song_display_df["artists"].astype(str)
    + " | "
    + song_display_df["track_id"].astype(str)
)

song_display_df = song_display_df.sort_values(
    ["popularity", "track_name"],
    ascending=[False, True]
)


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="top-header">
        <div class="brand-title">SONG RECOMMENDER</div>
        <div class="brand-subtitle">
        Build your music taste profile and discover similar songs
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Discover Your Next Favorite Song</div>
        <div class="hero-subtitle">
            AI-powered music recommendation system using Spotify audio features,
            similarity analysis, and machine learning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# NAVIGATION
# =========================
page = option_menu(
    menu_title=None,
    options=[
        "Recommender",
        "Dataset Insights",
        "Model Evaluation",
        "About Project",
    ],
    icons=[
        "music-note-beamed",
        "bar-chart",
        "graph-up",
        "info-circle"
    ],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "10px",
            "background-color": "#FFFFFF",
            "border-radius": "18px",
            "box-shadow": "0 6px 18px rgba(0,0,0,0.08)"
        },
        "icon": {
            "color": "#1DB954",
            "font-size": "18px"
        },
        "nav-link": {
            "font-size": "15px",
            "font-weight": "600",
            "color": "#374151",
            "text-align": "center",
            "margin": "0px 4px",
            "border-radius": "14px"
        },
        "nav-link-selected": {
            "background-color": "#1DB954",
            "color": "white"
        }
    }
)

st.divider()


# =========================
# RECOMMENDER
# =========================
if page == "Recommender":
    st.title("🎵 Personalized Music Recommender")

    st.markdown(
        """
        This is the main recommendation engine of the project.  
        Users can select up to **10 favorite songs**, and the model builds a personalized taste profile
        by averaging their Spotify audio features.
        """
    )

    st.divider()

    if "selected_song_labels" not in st.session_state:
        st.session_state.selected_song_labels = []

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.subheader("Step 1. Select Your Favorite Songs")

        search_text = st.text_input(
            "Search song or artist",
            placeholder="Example: BTS, Taylor Swift, The Weeknd, Ariana Grande"
        )

        filtered_options_df = song_display_df.copy()

        if search_text:
            filtered_options_df = filtered_options_df[
                filtered_options_df["song_label"].str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            ]

        filtered_options_df = (
            filtered_options_df
            .drop_duplicates(subset=["song_label"])
            .head(300)
        )

        song_options = filtered_options_df["song_label"].tolist()

        song_to_add = st.selectbox(
            "Search results",
            [""] + song_options,
            format_func=lambda x: "Select a song to add" if x == "" else x.split(" | ")[0]
        )

        col_add, col_clear = st.columns([1, 1])

        with col_add:
            if st.button("Add Song"):
                if song_to_add == "":
                    st.warning("Please choose a song first.")
                elif song_to_add in st.session_state.selected_song_labels:
                    st.info("This song is already selected.")
                elif len(st.session_state.selected_song_labels) >= 10:
                    st.warning("You can select up to 10 songs.")
                else:
                    st.session_state.selected_song_labels.append(song_to_add)
                    st.success("Song added.")

        with col_clear:
            if st.button("Clear Selected Songs"):
                st.session_state.selected_song_labels = []
                st.info("Selected songs cleared.")

        st.divider()

        st.subheader("Step 2. Recommendation Settings")

        n_recommendations = st.slider(
            "Number of recommendations",
            5,
            30,
            10
        )

        same_genre = st.checkbox(
            "Prefer same genres",
            value=False
        )

        min_popularity = st.slider(
            "Minimum popularity",
            0,
            100,
            30
        )

    selected_labels = st.session_state.selected_song_labels

    with right_col:
        st.subheader("Selected Songs")

        if len(selected_labels) == 0:
            st.info("No songs selected yet.")
        else:
            for label in selected_labels:
                display_label = label.split(" | ")[0]
                st.markdown(
                    f"""
                    <div class="tag">{display_label}</div>
                    """,
                    unsafe_allow_html=True
                )

        st.caption(
            "The recommendation model creates a user taste profile by averaging the audio features of the selected songs."
        )

    st.divider()

    if len(selected_labels) > 0:
        selected_track_ids = [label.split(" | ")[-1] for label in selected_labels]
        selected_songs = df[df["track_id"].isin(selected_track_ids)]["track_name"].tolist()

        selected_df = df[df["track_id"].isin(selected_track_ids)].copy()
        selected_visual_df = visual_df[visual_df["track_id"].isin(selected_track_ids)].copy()

        st.subheader("Selected Favorite Songs")

        st.dataframe(
            selected_df[
                ["track_name", "artists", "track_genre", "popularity"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("User Taste Profile")

        profile_values = selected_visual_df[FEATURES].mean()

        profile_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "average_value": profile_values.values
            }
        )

        fig = px.line_polar(
            profile_df,
            r="average_value",
            theta="feature",
            line_close=True,
            title="Average Audio Feature Profile of Selected Songs",
            template="plotly_dark"
        )
        fig.update_traces(fill="toself")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Select at least 2 songs to build a personalized recommendation profile.")

    if st.button("Generate Personalized Recommendations", type="primary"):
        if len(selected_labels) < 2:
            st.warning("Please select at least 2 songs to generate personalized recommendations.")
        else:
            selected_track_ids = [label.split(" | ")[-1] for label in selected_labels]
            selected_songs = df[df["track_id"].isin(selected_track_ids)]["track_name"].tolist()

            with st.spinner("Analyzing your music taste..."):
                recommendations = recommend_songs(
                    selected_songs,
                    df,
                    model,
                    X_scaled,
                    n_recommendations=100,
                    same_genre=same_genre
                )

            if recommendations.empty:
                st.warning("No recommendations found. Try selecting different songs or turning off same-genre preference.")
            else:
                recommendations = recommendations[
                    recommendations["popularity"] >= min_popularity
                ].head(n_recommendations)

                if recommendations.empty:
                    st.warning("No songs matched the minimum popularity filter.")
                else:
                    st.divider()
                    st.subheader("Step 3. Recommendation Results")

                    avg_similarity = recommendations["similarity_score"].mean()
                    avg_popularity = recommendations["popularity"].mean()
                    genre_count = recommendations["track_genre"].nunique()
                    dominant_genre = recommendations["track_genre"].mode()[0]

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Average Similarity", f"{avg_similarity:.3f}")
                    m2.metric("Average Popularity", f"{avg_popularity:.1f}")
                    m3.metric("Genre Diversity", genre_count)
                    m4.metric("Dominant Genre", dominant_genre)

                    st.markdown("### Recommended Songs")

                    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
                        spotify_url = f"https://open.spotify.com/track/{row['track_id']}"

                        with st.container(border=True):
                            embed_col, info_col, link_col = st.columns([1.6, 5.2, 1.2])

                            with embed_col:
                                components.iframe(
                                    f"https://open.spotify.com/embed/track/{row['track_id']}",
                                    height=120,
                                    scrolling=False
                                )

                            with info_col:
                                st.markdown(
                                    f"""
                                    <div class="song-title">#{rank} {row['track_name']}</div>
                                    <div class="song-meta">
                                        <b>Artist:</b> {row['artists']}<br>
                                        <b>Genre:</b> {row['track_genre']} &nbsp;&nbsp;
                                        <b>Popularity:</b> {row['popularity']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                                match_score = int(row["similarity_score"] * 100)

                                st.markdown(
                                    f"""
                                    <div class="match-badge">
                                        Match {match_score}%
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            with link_col:
                                st.link_button("Open in Spotify", spotify_url)

                    st.markdown("### Recommendation Table")
                    st.dataframe(
                        recommendations,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.divider()
                    st.subheader("Recommendation Analysis")

                    chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
                        [
                            "Similarity Ranking",
                            "Popularity vs Similarity",
                            "Genre Distribution",
                            "Feature Comparison"
                        ]
                    )

                    with chart_tab1:
                        fig = px.bar(
                            recommendations.sort_values("similarity_score"),
                            x="similarity_score",
                            y="track_name",
                            orientation="h",
                            color="similarity_score",
                            hover_data=["artists", "track_genre", "popularity"],
                            title="Similarity Score of Recommended Songs",
                            template="plotly_dark"
                        )
                        fig.update_layout(height=650)
                        st.plotly_chart(fig, use_container_width=True)

                    with chart_tab2:
                        fig = px.scatter(
                            recommendations,
                            x="similarity_score",
                            y="popularity",
                            size="popularity",
                            color="track_genre",
                            hover_name="track_name",
                            hover_data=["artists"],
                            title="Similarity vs Popularity of Recommended Songs",
                            template="plotly_dark"
                        )
                        fig.update_layout(height=650)
                        st.plotly_chart(fig, use_container_width=True)

                    with chart_tab3:
                        genre_counts = recommendations["track_genre"].value_counts().reset_index()
                        genre_counts.columns = ["genre", "count"]

                        fig = px.pie(
                            genre_counts,
                            names="genre",
                            values="count",
                            title="Genre Distribution of Recommended Songs",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with chart_tab4:
                        selected_df = df[df["track_id"].isin(selected_track_ids)].copy()

                        rec_full_df = df[
                            df["track_name"].isin(recommendations["track_name"])
                        ].copy()

                        input_profile = selected_df[FEATURES].mean()
                        recommendation_profile = rec_full_df[FEATURES].mean()

                        compare_profile_df = pd.DataFrame(
                            {
                                "feature": FEATURES,
                                "Selected Songs Profile": input_profile.values,
                                "Recommended Songs Profile": recommendation_profile.values
                            }
                        )

                        fig = px.bar(
                            compare_profile_df,
                            x="feature",
                            y=["Selected Songs Profile", "Recommended Songs Profile"],
                            barmode="group",
                            title="Input Taste Profile vs Recommended Songs Profile",
                            template="plotly_dark"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)

                    st.success(
                        "Recommendations were generated by comparing the user's averaged audio feature profile with all songs in the dataset."
                    )


# =========================
# DATASET INSIGHTS
# =========================
elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")

    st.write(
        """
        This page summarizes the Spotify dataset as a continuous analytical report.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Unique Genres", df["track_genre"].nunique())

    st.divider()

    st.subheader("Sample of Dataset")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Genre Distribution")
    genre_count = df["track_genre"].value_counts().head(25).reset_index()
    genre_count.columns = ["genre", "count"]

    fig = px.bar(
        genre_count,
        x="count",
        y="genre",
        orientation="h",
        title="Top 25 Genres by Number of Songs",
        template="plotly_dark"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=800)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("This chart shows which genres are most represented in the dataset.")

    st.divider()

    st.subheader("Popularity Distribution")
    fig = px.histogram(
        df,
        x="popularity",
        nbins=50,
        marginal="box",
        title="Distribution of Song Popularity",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("The popularity distribution helps identify whether the dataset is biased toward highly popular songs.")

    st.divider()

    st.subheader("Average Popularity by Genre")
    genre_pop = (
        df.groupby("track_genre")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    fig = px.bar(
        genre_pop,
        x="popularity",
        y="track_genre",
        orientation="h",
        title="Top 20 Genres by Average Popularity",
        template="plotly_dark"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Dataset Interpretation")
    st.info(
        """
        The dataset contains a wide range of genres and audio feature values.
        This diversity makes it suitable for building a content-based recommendation system.
        """
    )


# =========================
# MODEL EVALUATION
# =========================
elif page == "Model Evaluation":
    st.title("📈 Model Evaluation")

    st.markdown(
        """
        Recommendation systems do not have one fixed correct answer for each input song.
        Therefore, this project evaluates the usefulness of audio features through an additional KNN genre classification task.
        """
    )

    result_df, cm, labels, best_k, best_acc = evaluate_knn(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluation Task", "Genre Classification")
    c2.metric("Best k", best_k)
    c3.metric("Best Accuracy", f"{best_acc:.3f}")

    st.divider()

    st.subheader("Accuracy by k-value")
    fig = px.line(
        result_df,
        x="k",
        y="accuracy",
        markers=True,
        title="KNN Accuracy by Number of Neighbors",
        template="plotly_dark"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Confusion Matrix Heatmap")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        title=f"Confusion Matrix Heatmap, Best k = {best_k}",
        aspect="auto",
        template="plotly_dark"
    )
    fig.update_layout(height=750)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Evaluation Interpretation")
    st.markdown(
        """
        The main recommendation model uses similarity search, so traditional accuracy is not directly applicable.
        However, if the same audio features can classify genres with reasonable accuracy, it suggests that these features contain useful musical information.

        This supports the logic of the recommendation system:
        songs with similar audio feature vectors are likely to share similar musical characteristics.
        """
    )

    st.success(
        "The evaluation result supports the use of Spotify audio features for content-based music recommendation."
    )


# =========================
# ABOUT PROJECT
# =========================
elif page == "About Project":
    st.title("📝 About This Project")

    st.markdown(
        """
        ### Project Title
        Spotify Music Intelligence: Content-Based Music Recommendation System

        ### Machine Learning Method
        - K-Nearest Neighbors
        - Cosine similarity
        - StandardScaler preprocessing

        ### Main Input
        - Up to 10 songs selected by the user

        ### Main Output
        - Recommended songs with similar audio feature patterns
        """
    )

    st.subheader("Features Used")
    st.table(pd.DataFrame({"Audio Feature": FEATURES}))

    st.subheader("Limitations")
    st.markdown(
        """
        - The model does not analyze raw MP3 audio.
        - The model does not use lyrics.
        - The model does not use individual user listening history.
        - Recommendations are based only on structured Spotify audio features.
        """
    )

    st.subheader("Why Cosine Similarity?")

    st.markdown(
        """
        Cosine similarity measures how similar two songs are based on the direction of their audio feature vectors.
        This is useful because the recommendation model focuses on the overall pattern of musical characteristics,
        rather than only the absolute size of each feature value.

        For example, songs with similar danceability, energy, valence, and acousticness patterns are likely to feel musically similar.
        """
    )

    st.subheader("Why This Still Works")

    st.markdown(
        """
        Although this project does not use raw audio files, Spotify audio features already summarize important
        musical characteristics. Therefore, they can be used to build a meaningful content-based recommendation system.
        """
    )
