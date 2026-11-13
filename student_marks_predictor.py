
# Project 2: Student Marks Predictor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Training data
hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
marks = [15, 20, 35, 40, 50, 60, 65, 75, 85, 95]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    hours, marks, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("===== STUDENT MARKS PREDICTOR =====")
print("Model trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Predict marks
study_hours = float(input("How many hours do you study daily? "))

predicted_marks = model.predict([[study_hours]])

print("Predicted marks:", round(predicted_marks[0], 2))
