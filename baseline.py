import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate

# Get the age of the animal in total days
def age_days(s):
    if pd.isna(s): return np.nan

    n, unit = s.split()[:2]
    mult = {
        "day":1,
        "week":7,
        "month":30,
        "year":365
    } [unit.rstrip("s")]

    return int(n) * mult

df = pd.read_csv("res/train.csv")

df["at_risk"] = df["OutcomeType"].isin(["Euthanasia", "Died"]).astype(int)
# print(df["at_risk"].mean())              # base rate, ~.065
# print(df["AnimalID"].duplicated().sum()) # number of repeat animals, there are none

groups = df["AnimalID"]
y = df["at_risk"]

df["age_days"] = df["AgeuponOutcome"].apply(age_days)
dt = pd.to_datetime(df["DateTime"])

df["hour"] = dt.dt.hour
df["weekday"] = dt.dt.weekday
df["month"] = dt.dt.month

df["has_name"] = df["Name"].notna().astype(int)
df["is_mix"] = df["Breed"].str.contains("Mix|\"").astype(int)
df["color_primary"] = df["Color"].str.split("/").str[0]

# Some do not have a known sex, and they are left as "Unknown"

num_cols = ["age_days", "hour", "weekday", "month", "has_name", "is_mix"]
cat_cols = ["AnimalType", "SexuponOutcome", "Breed", "color_primary"]
X = df[num_cols + cat_cols]

#Exclude the Outcome columns to prevent data leakage

#Group split by animal
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
g_train = groups.iloc[train_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])

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
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000)) ])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True)
scoring = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1"
}

for name, model in [
    ("dummy", DummyClassifier(strategy="prior")),
    ("logreg", pipe)]:
    res = cross_validate(model, X_train, y_train, groups = g_train,
                         cv=cv, scoring=scoring)