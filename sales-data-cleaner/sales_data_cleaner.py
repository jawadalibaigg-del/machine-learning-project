"""
Sales Data Cleaner — Portfolio Project 3
=========================================
A beginner data-cleaning project using pandas.

Real-world data is messy: missing values, duplicate rows, inconsistent
spelling ("laptop" vs "Laptop "), and bad dates. This script:
  1. Creates a small messy sales CSV (if none exists yet)
  2. Loads it with pandas
  3. Cleans it step by step
  4. Analyses the clean data (revenue by product and by month)
  5. Saves the cleaned data to cleaned_sales.csv

Run it with:
    python sales_data_cleaner.py

Requires: pandas  (pip install pandas)
"""

import os
import pandas as pd

RAW_FILE = "sales_data.csv"
CLEAN_FILE = "cleaned_sales.csv"

# Messy sample data: duplicates, missing prices, inconsistent names,
# extra spaces, and one bad date — exactly the problems real data has.
SAMPLE_ROWS = [
    ("2026-11-01", "laptop", 2, 1200),
    ("2026-11-01", "Laptop ", 1, 1200),
    ("2026-11-02", "mouse", 5, 25),
    ("2026-11-02", "Mouse", 3, None),      # missing price
    ("2026-11-03", "keyboard", 4, 60),
    ("2026-11-03", "keyboard", 4, 60),      # exact duplicate row
    ("2026-11-04", "monitor", 1, 300),
    ("2026-11-05", "mouse", 2, 25),
    ("2026-11-06", "laptop", 1, None),      # missing price
    ("2026-11-07", "headphones", 6, 80),
    ("2026-11-08", "monitor", 2, 300),
    ("2026-11-09", "keyboard", 1, 60),
    ("bad-date", "mouse", 4, 25),           # bad date, will be dropped
    ("2026-11-10", "headphones", 3, 80),
    ("2026-11-11", "laptop", 1, 1200),
]


def create_sample_data():
    """Create the messy sample CSV if it does not exist yet."""
    if os.path.exists(RAW_FILE):
        return
    df = pd.DataFrame(
        SAMPLE_ROWS, columns=["date", "product", "quantity", "price"]
    )
    df.to_csv(RAW_FILE, index=False)
    print(f"Created sample file: {RAW_FILE}")


def load_data():
    """Load the raw CSV and show how messy it is."""
    df = pd.read_csv(RAW_FILE)
    print(f"\nLoaded {len(df)} rows from {RAW_FILE}")
    print("Missing values per column:")
    print(df.isna().sum())
    return df


def clean_data(df):
    """Clean the data, one problem at a time."""
    # 1. Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"\nRemoved {before - len(df)} duplicate row(s).")

    # 2. Fix product names: remove extra spaces, make them consistent
    df["product"] = df["product"].str.strip().str.title()

    # 3. Fix dates: bad dates become NaT, then drop those rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    print(f"Removed {before - len(df)} row(s) with bad dates.")

    # 4. Fill missing prices with the median price of the same product
    df["price"] = df.groupby("product")["price"].transform(
        lambda s: s.fillna(s.median())
    )

    # 5. Drop rows still missing quantity or price
    df = df.dropna(subset=["quantity", "price"])

    # 6. Add a revenue column: quantity x price
    df["revenue"] = df["quantity"] * df["price"]
    return df


def analyse(df):
    """Print simple insights from the clean data."""
    total = df["revenue"].sum()
    print(f"\n--- Results ({len(df)} clean rows) ---")
    print(f"Total revenue: ${total:,.0f}")

    print("\nRevenue by product:")
    by_product = df.groupby("product")["revenue"].sum().sort_values(
        ascending=False
    )
    for product, revenue in by_product.items():
        print(f"  {product:<12} ${revenue:,.0f}")

    top_product = by_product.index[0]
    print(f"\nTop-selling product: {top_product}")

    print("\nRevenue by month:")
    monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
    for month, revenue in monthly.items():
        print(f"  {month}  ${revenue:,.0f}")

    print(f"\nAverage order value: ${df['revenue'].mean():.2f}")


def main():
    create_sample_data()
    df = load_data()
    df = clean_data(df)
    analyse(df)
    df.to_csv(CLEAN_FILE, index=False)
    print(f"\nClean data saved to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
