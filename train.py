import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("dataset/Housing.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Check target column
if "price" not in df.columns:
    raise ValueError("Dataset lo 'price' column ledu!")

# 3. Prepare data
df = df.drop_duplicates()
X = df.drop(columns=["price"])
y = pd.to_numeric(df["price"], errors="coerce")

valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

# 4. Convert categorical columns into numbers
X = pd.get_dummies(X, drop_first=True)

# 5. Handle missing values
X = X.fillna(X.median())

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed!")

# 8. Evaluate model
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 9. Save model
Path("artifacts").mkdir(exist_ok=True)

joblib.dump(model, "artifacts/house_price_model.joblib")
joblib.dump(X.columns.tolist(), "artifacts/model_columns.joblib")

print("Model saved successfully!")
print("TRAINING COMPLETED!")