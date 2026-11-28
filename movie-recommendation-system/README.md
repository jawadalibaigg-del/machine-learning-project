# Movie Recommendation System

A beginner machine learning project that recommends movies using
**user-based collaborative filtering**: it finds users with similar taste to
yours and suggests movies those users loved but you have not watched yet.

## How it works

1. Every user rates some movies from 1 to 5 (0 means "not rated yet").
2. **Cosine similarity** measures how similar two users' tastes are.
3. To predict your rating for a movie, it takes a **weighted average** of
   the ratings similar users gave that movie — more similar users count more.
4. The movies with the highest predicted ratings become your recommendations.

## Features

- Small hand-built ratings table (8 users x 10 movies) — runs fully offline
- User similarity table so you can see who has similar taste
- Top-3 movie recommendations for every user
- Evaluation: hides 25% of the known ratings, predicts them, and reports
  the **RMSE** (average prediction error)

## How to run

```bash
pip install -r requirements.txt
python movie_recommendation_system.py
```

## Example output

```
Users most similar to Ali:
  Usman: similarity 0.94
  Hamza: similarity 0.94
  Bilal: similarity 0.92

For Ali:
  Finding Nemo  (predicted rating: 2.85 / 5)
```

## What was learned

- **Collaborative filtering** — recommending from other people's behaviour
- **Cosine similarity** — measuring the angle between two rating vectors
- **Weighted averages** — similar users count more in a prediction
- **pandas DataFrames** — storing and reading the ratings table
- **NumPy** — fast maths on arrays (mean, square root)
- **Train/test split** — hiding some data to honestly test the model
- **RMSE** — measuring how far predictions are from real ratings
