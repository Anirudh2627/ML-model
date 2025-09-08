import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
file_path = "451ee8f7-085b-4dc0-b7db-6de151289cbb.csv"  # update path if needed
raw = pd.read_csv(file_path, skiprows=1)
raw.columns = ["index", "id", "age", "salary", "bought_TV"]


df = raw.drop(columns=["index", "id"]).copy()


df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
df["bought_TV"] = pd.to_numeric(df["bought_TV"], errors="coerce").astype("Int64")


df = df.dropna().reset_index(drop=True)
df["bought_TV"] = df["bought_TV"].astype(int)

# Remove invalid values (negative or zero)
df = df[(df["age"] > 0) & (df["salary"] > 0)]
print("After removing invalid values:", df.shape)


def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

df = remove_outliers(df, "age")
df = remove_outliers(df, "salary")
print("After outlier removal:", df.shape)

X = df[["age", "salary"]].values
y = df["bought_TV"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

param_grids = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear"]
    },
    "DecisionTree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    },
    "RandomForest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
}

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42)
}


best_models = {}
for name, model in models.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print("Best Params:", grid.best_params_)


metrics_rows = {}
roc_curves = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)
    
    metrics_rows[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr)

metrics_df = pd.DataFrame(metrics_rows).T
print("\n0Final Model Performance:\n", metrics_df)

# Confusion Matrices
for name, model in best_models.items():
    cm = confusion_matrix(y_test, model.predict(X_test))
    print(f"\nConfusion Matrix - {name}:\n", cm)

# ===============================
# 7) ROC Curves
# ===============================
plt.figure()
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves (Tuned Models)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ===============================
# 8) Feature Importance
# ===============================
# Logistic Regression
lr = best_models["LogisticRegression"].named_steps["clf"]
print("\nLogistic Regression Coefficients:")
for feat, coef in zip(["age", "salary"], lr.coef_[0]):
    print(f"{feat}: {coef:.4f}")

# Decision Tree & Random Forest
for name in ["DecisionTree", "RandomForest"]:
    model = best_models[name]
    print(f"\n{name} Feature Importances:")
    for feat, imp in zip(["age", "salary"], model.feature_importances_):
        print(f"{feat}: {imp:.4f}")

