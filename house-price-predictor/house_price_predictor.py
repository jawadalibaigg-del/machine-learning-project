# House Price Predictor - my first portfolio project
# Lesson 3 idea: linear regression - price = m * size + b
# m = slope (price goes up by m for each extra sq ft)
# b = intercept (predicted price when size is 0)

# Data from ML Episode 1: house size (sq ft) and price
sizes = [1000, 1500, 2000, 2500]
prices = [50, 75, 100, 125]

def train(sizes, prices):
    """Find m (slope) and b (intercept) for the best straight line."""
    n = len(sizes)
    mean_size = sum(sizes) / n
    mean_price = sum(prices) / n
    top = sum((x - mean_size) * (y - mean_price) for x, y in zip(sizes, prices))
    bottom = sum((x - mean_size) ** 2 for x in sizes)
    m = top / bottom
    b = mean_price - m * mean_size
    return m, b

def predict(size, m, b):
    return m * size + b

m, b = train(sizes, prices)
print(f"Model: price = {m} * size + {b}")
print(f"Check: a 1500 sq ft house costs about {predict(1500, m, b)}")

try:
    size = float(input("Enter a house size in sq ft: "))
    print(f"Predicted price: {predict(size, m, b)}")
except ValueError:
    print("Please enter a number, e.g. 1600")
