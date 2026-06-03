# Spotify Music Intelligence

This project is a content-based music recommendation system using Spotify audio features.

## Project Overview

Users select songs they like, and the system recommends similar songs based on audio feature similarity.

## Main Features

- Personalized song recommendation
- Spotify track embed
- Audio feature visualization
- Dataset insights
- KNN-based genre classification evaluation
- Business use case analysis

## Dataset

The dataset contains Spotify track information including:

- track name
- artist
- genre
- popularity
- danceability
- energy
- valence
- tempo
- acousticness
- instrumentalness
- liveness
- loudness
- speechiness

## Model

The recommendation system uses:

- StandardScaler for feature scaling
- K-Nearest Neighbors
- Cosine similarity

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py

Project Structure
spotify-recommendation-project/
├── app.py
├── preprocessing.py
├── recommendation.py
├── requirements.txt
├── README.md
└── data/

Limitations
* The model does not use lyrics.
* The model does not use user listening history.
* The model does not analyze raw audio files.
* Recommendations are based only on structured audio features.

Future Improvements
* Add collaborative filtering
* Add lyrics-based NLP analysis
* Improve duplicate song handling
* Add user profile saving
* Deploy using Streamlit Cloud


