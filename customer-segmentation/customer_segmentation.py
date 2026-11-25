"""
Customer Segmentation with K-Means Clustering.

A shop wants to understand its customers so it can send the right
offers to the right people. This script creates a small pretend
customer dataset (age, annual income, spending score), then uses
K-Means clustering — an unsupervised learning algorithm — to split
the customers into groups with similar behaviour.

It also uses the "elbow method" to decide how many groups (clusters)
make sense, and the silhouette score to check how clean the split is.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def make_customer_data(seed=42):
    """Create a pretend dataset of 200 customers.

    Each customer has two numbers we cluster on:
    - Annual Income (in thousands of dollars)
    - Spending Score (1-100, how much they like to spend)

    The customers are generated around 5 hidden centres so that
    real groups exist for K-Means to find.
    """
    rng = np.random.default_rng(seed)

    # The 5 hidden groups: (income, spending score) centres.
    centres = np.array([
        [25, 80],   # low income, loves to spend
        [30, 25],   # low income, careful spender
        [60, 50],   # medium income, medium spender
        [85, 80],   # high income, loves to spend
        [90, 25],   # high income, careful spender
    ])
    sizes = [40, 40, 45, 35, 40]  # customers in each group

    chunks = []
    for centre, size in zip(centres, sizes):
        # Random points scattered around the centre (spread of 6).
        chunk = rng.normal(loc=centre, scale=6, size=(size, 2))
        chunks.append(chunk)

    data = np.vstack(chunks)
    data[:, 0] = np.clip(data[:, 0], 15, 140)   # income stays realistic
    data[:, 1] = np.clip(data[:, 1], 1, 100)    # spending score is 1-100
    return data


def elbow_method(data, max_k=10):
    """Try K = 1..max_k and print the inertia (within-cluster spread).

    Inertia always falls as K grows. The "elbow" is the K where the
    improvement suddenly slows down — that is a good number of clusters.
    """
    print("Elbow method — inertia for each K:")
    inertias = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)
        print(f"  K = {k:2d}  inertia = {model.inertia_:8.1f}")
    return inertias


def main():
    # 1. Build the data.
    data = make_customer_data()
    print(f"Dataset: {data.shape[0]} customers, {data.shape[1]} features")
    print("Features: Annual Income (k$) and Spending Score (1-100)")

    # 2. Scale the features so income and score count equally.
    #    Without this, income (15-140) would outweigh the score (1-100).
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # 3. Elbow method to choose K.
    inertias = elbow_method(scaled)
    best_k = 5
    print(f"\nThe elbow is around K = {best_k} (the curve flattens after it).")

    # 4. Train the final K-Means model with the chosen K.
    model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = model.fit_predict(scaled)

    # 5. Score how clean the split is (silhouette: -1 bad, +1 perfect).
    score = silhouette_score(scaled, labels)
    print(f"Silhouette score: {score:.3f} (closer to 1 is better)")

    # 6. Describe each cluster in plain words.
    print("\nCustomer groups found:")
    for cluster in range(best_k):
        members = data[labels == cluster]
        print(
            f"  Group {cluster}: {len(members):3d} customers | "
            f"avg income ${members[:, 0].mean():.0f}k | "
            f"avg spending score {members[:, 1].mean():.0f}/100"
        )

    # 7. Save a scatter chart of the groups.
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", s=40)
    plt.scatter(
        scaler.inverse_transform(model.cluster_centers_)[:, 0],
        scaler.inverse_transform(model.cluster_centers_)[:, 1],
        c="red", marker="X", s=200, label="Group centres",
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments (K-Means, K=5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("customer_segments.png")
    print("\nChart saved to customer_segments.png")


if __name__ == "__main__":
    main()
