# Expense Tracker

My second portfolio project. A small Python program that saves your
spending to a CSV file and shows how much you spent in each category.

## What it does
- Add an expense: date, category, item, and amount
- Saves every expense to `expenses.csv`
- Shows all your expenses, or the total spent by category
- Uses only the Python standard library — no installs needed

## How to run
```bash
python expense_tracker.py
```
Choose `1` to add an expense, `2` to see them all, `3` to see totals.

Try it: add `2026-11-13, food, lunch, 500` and
`2026-11-13, transport, bus, 100`, then choose `3` — total should be 600.

## What I learned
- Reading and writing CSV files with Python
- Saving data so it is still there next time you run the program
- Grouping data and adding up totals by category
- Building a simple menu in a command-line program

## Next step
Add a `4. Delete an expense` option, or a monthly total report.
