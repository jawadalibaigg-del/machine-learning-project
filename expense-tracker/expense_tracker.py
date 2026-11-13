# Expense Tracker - my second portfolio project
# A small Python program that saves your spending to a CSV file
# and shows how much you spent in each category.
# Only uses the Python standard library - no installs needed.

import csv
import os

FILE = "expenses.csv"

def load_expenses():
    """Read all saved expenses from the CSV file (empty list if none)."""
    expenses = []
    if not os.path.exists(FILE):
        return expenses
    with open(FILE, newline="") as f:
        for row in csv.DictReader(f):
            expenses.append({
                "date": row["date"],
                "category": row["category"],
                "item": row["item"],
                "amount": float(row["amount"]),
            })
    return expenses

def save_expense(date, category, item, amount):
    """Add one expense to the CSV file."""
    is_new = not os.path.exists(FILE)
    with open(FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "item", "amount"])
        if is_new:
            writer.writeheader()
        writer.writerow({
            "date": date,
            "category": category,
            "item": item,
            "amount": amount,
        })

def show_totals(expenses):
    """Print total spending, and the total for each category."""
    total = sum(e["amount"] for e in expenses)
    print(f"\nTotal spent: {total:.2f}")
    print("By category:")
    categories = {}
    for e in expenses:
        categories[e["category"]] = categories.get(e["category"], 0) + e["amount"]
    for category, amount in categories.items():
        print(f"  {category}: {amount:.2f}")

def main():
    print("Expense Tracker")
    print("1. Add expense")
    print("2. Show all expenses")
    print("3. Show totals")
    choice = input("Choose 1, 2 or 3: ").strip()

    if choice == "1":
        date = input("Date (e.g. 2026-11-13): ").strip()
        category = input("Category (e.g. food, transport): ").strip()
        item = input("What did you buy? ").strip()
        amount = float(input("Amount: ").strip())
        save_expense(date, category, item, amount)
        print(f"Saved: {item} ({category}) - {amount:.2f}")
    elif choice == "2":
        for e in load_expenses():
            print(f"{e['date']} | {e['category']} | {e['item']} | {e['amount']:.2f}")
    elif choice == "3":
        show_totals(load_expenses())
    else:
        print("Please choose 1, 2 or 3.")

if __name__ == "__main__":
    main()
