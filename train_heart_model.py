import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ================= LOAD DATA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "heart_balanced_dataset.xlsx")

df = pd.read_excel(file_path)

# ================= CLEAN COLUMN NAMES =================
df.columns = df.columns.str.strip().str.replace(" ", "")

print("Columns:", df.columns)

# ================= HANDLE MISSING VALUES =================
df = df.replace("NA", pd.NA)
df = df.fillna(method="ffill")

# ================= TARGET =================
target = "Heart_stroke"

# ================= ENCODE TARGET =================
le_target = LabelEncoder()
df[target] = le_target.fit_transform(df[target])

# ================= HANDLE CATEGORICAL COLUMNS =================
# Convert yes/no → 1/0
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].str.lower()
        df[col] = df[col].replace({"yes": 1, "no": 0})

# Encode remaining text columns (like education)
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ================= SPLIT =================
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TRAIN =================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================= PREDICT =================
y_pred = model.predict(X_test)

# ================= ACCURACY =================
acc = accuracy_score(y_test, y_pred)
print("\n🔥 Accuracy:", acc)

# ================= EXTRA METRICS =================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ================= SAVE =================
pickle.dump(model, open(os.path.join(BASE_DIR, "heart_model.pkl"), "wb"))
pickle.dump(le_target, open(os.path.join(BASE_DIR, "heart_encoder.pkl"), "wb"))

print("\n✅ Model saved successfully")