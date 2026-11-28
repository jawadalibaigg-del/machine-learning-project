"""
Movie Recommendation System - a beginner machine learning project.

This project recommends movies to a user based on what similar users liked.
It uses "user-based collaborative filtering": first it finds users whose
taste is similar to yours (cosine similarity), then it recommends movies
those similar users rated highly but you have not watched yet.

The ratings data is a small hand-built table, so the project runs fully
offline - no downloads are needed.

Concepts used: pandas DataFrames, NumPy arrays, cosine similarity,
collaborative filtering, train/test split, and RMSE as an evaluation metric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# The data: a small ratings table.
# Rows are users, columns are movies, values are ratings from 1 to 5.
# A 0 means "this user has not rated this movie yet".
# ---------------------------------------------------------------------------

MOVIES = [
    "The Dark Knight",
    "Inception",
    "Interstellar",
    "Titanic",
    "The Notebook",
    "Avengers",
    "Iron Man",
    "Toy Story",
    "Finding Nemo",
    "The Godfather",
]

RATINGS = [
    # Dark Knight, Inception, Interstellar, Titanic, Notebook, Avengers,
    # Iron Man, Toy Story, Finding Nemo, Godfather
    [5, 5, 4, 1, 1, 5, 4, 2, 0, 3],   # Ali
    [4, 5, 5, 2, 1, 4, 5, 0, 1, 0],   # Bilal
    [1, 2, 1, 5, 5, 1, 0, 4, 5, 2],   # Sara
    [2, 1, 0, 4, 5, 2, 1, 5, 4, 1],   # Ayesha
    [5, 4, 5, 1, 0, 5, 5, 2, 1, 0],   # Usman
    [1, 1, 2, 5, 4, 0, 1, 5, 5, 2],   # Hira
    [4, 4, 3, 2, 1, 5, 4, 3, 0, 0],   # Hamza
    [2, 2, 1, 4, 5, 2, 0, 4, 5, 1],   # Zainab
]

USERS = ["Ali", "Bilal", "Sara", "Ayesha", "Usman", "Hira", "Hamza", "Zainab"]


def build_ratings_table():
    """Put the ratings list into a pandas DataFrame with user and movie names."""
    ratings = pd.DataFrame(RATINGS, index=USERS, columns=MOVIES)
    return ratings


def compute_user_similarity(ratings):
    """Compute how similar every user is to every other user.

    Cosine similarity measures the angle between two users' rating vectors.
    A value close to 1 means very similar taste; close to 0 means very
    different taste. The result is a square table: user x user.
    """
    similarity = cosine_similarity(ratings)
    similarity_table = pd.DataFrame(
        similarity, index=ratings.index, columns=ratings.index
    )
    return similarity_table


def most_similar_users(similarity_table, user, top_n=3):
    """Return the top_n users most similar to the given user (excluding the user)."""
    scores = similarity_table[user].drop(user)
    return scores.sort_values(ascending=False).head(top_n)


def predict_rating(ratings, similarity_table, user, movie):
    """Predict how much a user would like a movie they have not rated.

    The prediction is a weighted average of the ratings that similar users
    gave this movie. Users who are more similar to you count more
    (their similarity score is the weight).
    """
    similar_users = similarity_table[user].drop(user)
    total_weight = 0.0
    weighted_sum = 0.0
    for other_user, similarity in similar_users.items():
        rating = ratings.loc[other_user, movie]
        if rating > 0 and similarity > 0:
            weighted_sum += similarity * rating
            total_weight += similarity
    if total_weight == 0:
        # Nobody similar has rated this movie, so fall back to its average rating.
        rated = ratings[movie][ratings[movie] > 0]
        return float(rated.mean()) if len(rated) > 0 else 0.0
    return weighted_sum / total_weight


def recommend_movies(ratings, similarity_table, user, top_n=3):
    """Recommend the top_n movies the user has not rated yet.

    Every unrated movie gets a predicted rating, then the movies are sorted
    from the highest predicted rating to the lowest.
    """
    predictions = {}
    for movie in ratings.columns:
        if ratings.loc[user, movie] == 0:
            predictions[movie] = predict_rating(ratings, similarity_table, user, movie)
    ranked = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def evaluate(ratings):
    """Test the recommender by hiding some known ratings and predicting them.

    We flatten all known ratings into (user, movie, rating) rows, hide 25%
    of them as a test set, predict each hidden rating from the other ratings,
    and report the RMSE (root mean squared error): the average size of our
    prediction mistakes. Lower RMSE means better predictions.
    """
    known = []
    for user in ratings.index:
        for movie in ratings.columns:
            rating = ratings.loc[user, movie]
            if rating > 0:
                known.append((user, movie, rating))

    train_rows, test_rows = train_test_split(known, test_size=0.25, random_state=42)

    # Rebuild a ratings table that only contains the training ratings.
    train_ratings = pd.DataFrame(0, index=ratings.index, columns=ratings.columns)
    for user, movie, rating in train_rows:
        train_ratings.loc[user, movie] = rating

    similarity_table = compute_user_similarity(train_ratings)

    squared_errors = []
    for user, movie, actual in test_rows:
        predicted = predict_rating(train_ratings, similarity_table, user, movie)
        squared_errors.append((predicted - actual) ** 2)

    rmse = float(np.sqrt(np.mean(squared_errors)))
    return rmse, len(test_rows)


def main():
    """Run the whole recommendation demo and print the results."""
    ratings = build_ratings_table()
    print("========== MOVIE RATINGS TABLE ==========")
    print(ratings)
    print()

    similarity_table = compute_user_similarity(ratings)
    print("========== USER SIMILARITY TABLE ==========")
    print(similarity_table.round(2))
    print()

    # Show the most similar users for one example user.
    example_user = "Ali"
    print(f"Users most similar to {example_user}:")
    for user, score in most_similar_users(similarity_table, example_user).items():
        print(f"  {user}: similarity {score:.2f}")
    print()

    # Recommend movies for every user.
    print("========== RECOMMENDATIONS ==========")
    for user in USERS:
        recommendations = recommend_movies(ratings, similarity_table, user, top_n=3)
        print(f"\nFor {user}:")
        for movie, predicted in recommendations:
            print(f"  {movie}  (predicted rating: {predicted:.2f} / 5)")
    print()

    # Evaluate how accurate the predictions are.
    rmse, test_count = evaluate(ratings)
    print("========== EVALUATION ==========")
    print(f"Tested on {test_count} hidden ratings.")
    print(f"RMSE: {rmse:.2f} (on a 1-5 scale, lower is better)")


if __name__ == "__main__":
    main()
