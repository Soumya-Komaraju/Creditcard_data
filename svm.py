import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("credit_data.csv")

print("=== Dataset Loaded ===")
print(df.head())
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df["default"].value_counts())
print()

# Features and Target
X = df.drop("default", axis=1)
y = df["default"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train SVM Model
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

print("=== Model Trained ===")
print("Support Vectors per Class:", model.n_support_)
print("Total Support Vectors:", model.n_support_.sum())
print()

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))