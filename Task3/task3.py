import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("bank.csv", sep=";")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
df.info()

# ----------------------------
# Convert Categorical Columns
# ----------------------------

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

# ----------------------------
# Features and Target
# ----------------------------

X = df.drop("y", axis=1)
y = df["y"]

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Model
# ----------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ----------------------------
# Prediction
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# Accuracy
# ----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# ----------------------------
# Confusion Matrix
# ----------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()

# ----------------------------
# Classification Report
# ----------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Feature Importance
# ----------------------------

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.plot(kind="bar")

plt.title("Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()

# ----------------------------
# Decision Tree
# ----------------------------

plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    fontsize=8
)

plt.tight_layout()

plt.savefig("decision_tree.png")
plt.show()

# ----------------------------
# Final Output
# ----------------------------

print("\n" + "=" * 50)
print("TASK COMPLETED SUCCESSFULLY")
print("=" * 50)