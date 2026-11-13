# House Price Predictor

My first portfolio project. A small Python program that predicts a house
price from its size using linear regression (Lesson 3).

## What it does
- Learns a straight line from 4 house examples: price = m * size + b
- `m` is the slope, `b` is the intercept (the predicted price when size is 0)
- Asks you for a house size and prints the predicted price

## How to run
```bash
python house_price_predictor.py
```
Try size `1600` — it should predict `80.0`.

## What I learned
- Linear regression, slope and intercept
- Training a model on small data and using it to predict
- Writing clean, commented Python

## Next step
Add more house examples to the `sizes` and `prices` lists and see how
`m` and `b` change.
