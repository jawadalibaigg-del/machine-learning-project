# Sales Data Cleaner

Portfolio Project 3 — clean messy sales data with **pandas** and find the
best-selling products.

Real-world data is never clean: this project works with missing prices,
duplicate rows, inconsistent product names (`laptop` vs `Laptop `) and a bad
date, and fixes all of them step by step.

## How to run

```bash
pip install pandas
python sales_data_cleaner.py
```

The first run creates a sample `sales_data.csv` for you. The script then
cleans it and saves the result as `cleaned_sales.csv`.

## What it does

1. **Loads** the messy CSV with pandas and reports missing values
2. **Cleans** it:
   - Drops duplicate rows
   - Standardises product names (trims spaces, consistent capitals)
   - Drops rows with unusable dates
   - Fills missing prices with the median price of the same product
3. **Analyses** the clean data:
   - Total revenue
   - Revenue by product and top-selling product
   - Revenue by month and average order value
4. **Saves** the cleaned data to `cleaned_sales.csv`

## Example output

```
Loaded 15 rows from sales_data.csv
Removed 1 duplicate row(s).
Removed 1 row(s) with bad dates.

--- Results (13 clean rows) ---
Total revenue: $8,170

Revenue by product:
  Laptop       $6,000
  Monitor      $900
  Headphones   $720
  Keyboard     $300
  Mouse        $250

Top-selling product: Laptop
```

## What I learned

- Loading and inspecting a CSV with pandas (`read_csv`, `isna().sum()`)
- Cleaning data: `drop_duplicates`, `dropna`, `fillna`
- Fixing text with `str.strip()` and `str.title()`
- Parsing dates with `pd.to_datetime(errors="coerce")`
- Grouping and summarising with `groupby()`
- Why cleaning data is the first step of every real ML project — a model
  trained on messy data gives messy results

## Next step

Try it on your own data: replace `sales_data.csv` with any CSV that has
`date`, `product`, `quantity` and `price` columns and run the script again.
