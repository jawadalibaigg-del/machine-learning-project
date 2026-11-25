# Customer Segmentation with K-Means Clustering

A shop wants to understand its customers so it can send the right offers to the right people. This project uses **K-Means clustering** — an unsupervised machine learning algorithm — to split 200 customers into groups with similar income and spending habits.

## Features

- Self-generated customer dataset (no downloads needed): Annual Income and Spending Score for 200 customers
- **Elbow method** to choose the best number of clusters (K)
- **K-Means clustering** to find the customer groups
- **Silhouette score** to measure how clean the split is
- Plain-word summary of each group (size, average income, average spending score)
- Scatter chart of the groups saved as `customer_segments.png`

## How to Run

```bash
pip install -r requirements.txt
python customer_segmentation.py
```

## Example Output

```
Dataset: 200 customers, 2 features
Elbow method — inertia for each K:
  K =  1  inertia =    400.0
  ...
Silhouette score: 0.55 (closer to 1 is better)

Customer groups found:
  Group 0:  40 customers | avg income $25k | avg spending score 80/100
  Group 1:  40 customers | avg income $30k | avg spending score 25/100
  ...
Chart saved to customer_segments.png
```

## What Was Learned

- **Unsupervised learning** — finding groups in data with no labels to learn from
- **K-Means clustering** — how the algorithm picks centres and assigns each point to the nearest one
- **The elbow method** — choosing K by watching where the inertia curve bends
- **Inertia** — how spread out the points are inside their clusters
- **Silhouette score** — a single number that says how clean the clusters are
- **Feature scaling** — why `StandardScaler` is needed before measuring distances
