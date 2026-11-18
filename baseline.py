import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from datetime import datetime

# DO NOT FORGET - Log update code
WHO = "Unknown"
NOTES = "Forgot to set log settings :( )"
LOG_FILE = "res/experiment_log.csv"

# ---- New: read the cleaned file made by data_cleaning.py (run that first) ----
df = pd.read_csv("res/train_clean.csv")

# all 5 outcome classes
y = df["OutcomeType"]
X = df.drop(columns="OutcomeType")

# binary cols go in with the numeric ones, they just get scaled
num_cols = ["age_days", "is_dog", "has_name", "is_mix", "multi_color"]
cat_cols = ["sex", "fixed", "breed_main", "color_main"]

# stratify keeps the outcome proportions the same in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()) ])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)) ])
pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols) ])
pipe = Pipeline([
    ("pre", pre),
    ("model", LogisticRegression(class_weight='balanced', max_iter=1000)) ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro"
}

for name, model in [
    ("dummy", DummyClassifier(strategy="prior")),
    ("logreg", pipe)]:

    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    print(name, {k: round(float(v.mean()), 3) for k, v in res.items() if k.startswith("test_")})

    # ---- Experiment log: one row per model per run, appended to res/experiment_log.csv ----
    log_row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "who": WHO,
        "model": name,
        "target": "OutcomeType (5 class)",
        "features": ", ".join(num_cols + cat_cols),
        "cv": "StratifiedKFold(5), 80/20 split, random_state=42",
        "accuracy": round(float(res["test_accuracy"].mean()), 3),
        "balanced_accuracy": round(float(res["test_balanced_accuracy"].mean()), 3),
        "f1_macro": round(float(res["test_f1_macro"].mean()), 3),
        "notes": NOTES,
    }])
    log_row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)
