# app.py
# A minimal OOP ML app for regression/classification with readiness checks, preprocessing,
# model training via GridSearchCV(cv=10), reporting, and optional model persistence.

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# plotting (used for confusion matrices)
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# regressors
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score
)

RANDOM_STATE = 42

class InputHandler:
    """Handles user choices via CLI."""

    def ask_ml_type(self) -> str:
        choice = input("Do you want a regressor or classifier? ").strip().lower()
        if choice not in {"regressor", "classifier"}:
            raise ValueError("Invalid choice. Must be 'regressor' or 'classifier'.")
        return choice

    def ask_file_path(self) -> str:
        path = input("Enter path to your CSV file: ").strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' does not exist.")
        if not path.lower().endswith(".csv"):
            raise ValueError("Only .csv files are supported.")
        return path

    def ask_target(self, df: pd.DataFrame) -> str:
        print("\nColumns in dataset:\n", df.columns.tolist())
        target = input("Enter the target column (y): ").strip()
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in columns.")
        return target

class DataHandler:
    """Loads and exposes the dataframe"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df: pd.DataFrame = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded data: shape={self.df.shape}")
        return self.df

class DataValidator:
    """
    Validates dataset readiness for ML:
    - presence of target
    - missing values
    - encodable dtypes
    - for classifier: target categorical
    - for regressor: target continuous
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target

    def base_issues(self) -> list:
        issues = []
        if self.target not in self.df.columns:
            issues.append(f"Target '{self.target}' not in data.")
            return issues

        # Missing values
        missing = self.df.isna().sum()
        miss_cols = missing[missing > 0].index.tolist()
        if miss_cols:
            issues.append(f"Missing values in: {miss_cols}")

        # String columns (to be encoded by preprocessing; not a hard failure, but we flag)
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            issues.append(f"Categorical/string columns will be one-hot encoded: {cat_cols}")

        return issues

    def ready_for_regression(self) -> Tuple[bool, list]:
        issues = self.base_issues()
        if self.target in self.df.columns:
            if not pd.api.types.is_numeric_dtype(self.df[self.target]):
                issues.append(f"Target '{self.target}' is not numeric; regression expects continuous target.")
        return (len([i for i in issues if "Missing values" in i or "not numeric" in i or "not in data" in i]) == 0, issues)

    def ready_for_classification(self) -> Tuple[bool, list]:
        issues = self.base_issues()
        if self.target in self.df.columns:
            # numeric is OK if few unique values; otherwise warn
            nunique = self.df[self.target].nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(self.df[self.target]) and nunique > 20:
                issues.append(f"Target '{self.target}' looks continuous (unique={nunique}); classification expects categories.")
        # ready if no missing and target not continuous-like
        hard = any(txt.startswith("Missing values") or "not in data" in txt for txt in issues)
        cont_like = any("looks continuous" in txt for txt in issues)
        return (not hard and not cont_like, issues)

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),  # with_mean=False allows sparse stack with OHE
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre

class RegressorTrainer:
    """
    Trains multiple regressors with GridSearchCV(cv=10),
    prints best params + MAE + RMSE + R2, and returns the best.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self.results: Dict[str, Dict[str, Any]] = {}

    def run(self) -> Dict[str, Dict[str, Any]]:
        y = self.df[self.target].astype(float)
        X = self.df.drop(columns=[self.target])
        pre = make_preprocessor(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        models = {
            "LinearRegression": (LinearRegression(), {}),
            "Ridge": (Ridge(random_state=RANDOM_STATE), {"clf__alpha": [0.01, 0.1, 1, 10]}),
            "Lasso": (Lasso(random_state=RANDOM_STATE, max_iter=5000), {"clf__alpha": [0.001, 0.01, 0.1, 1]}),
            "ElasticNet": (ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
                           {"clf__alpha": [0.001, 0.01, 0.1, 1], "clf__l1_ratio": [0.2, 0.5, 0.8]}),
            "SVR": (SVR(),
                    {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"], "clf__gamma": ["scale", "auto"]}),
            "ANN(MLPRegressor)": (MLPRegressor(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True),
                                  {"clf__hidden_layer_sizes": [(64,), (64, 32)], "clf__alpha": [1e-4, 1e-3]}),
        }

        for name, (est, grid) in models.items():
            pipe = Pipeline([("prep", pre), ("clf", est)])
            gs = GridSearchCV(pipe, grid, cv=10, n_jobs=-1, scoring="neg_root_mean_squared_error")
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            y_pred = best.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            self.results[name] = {
                "best_params": gs.best_params_,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "model": best,
            }

            print(f"\n=== {name} ===")
            print("Best params:", gs.best_params_)
            print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        # choose best by RMSE (lowest)
        best_name = min(self.results, key=lambda k: self.results[k]["RMSE"])
        print(f"\n⭐ Best regressor: {best_name} (RMSE={self.results[best_name]['RMSE']:.4f})")
        return self.results


class ClassifierTrainer:
    """
    Trains logistic/KNN/SVC/ANN with GridSearchCV(cv=10),
    prints best params, confusion matrix, classification report, accuracy,
    and returns the best.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self.results: Dict[str, Dict[str, Any]] = {}

    def _train_one(self, name, estimator, param_grid, X_train, y_train, X_test, y_test, pre):
        pipe = Pipeline([("prep", pre), ("clf", estimator)])
        gs = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1, scoring="accuracy")
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        y_pred = best.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        plt.show()

        print(f"\n=== {name} ===")
        print("Best params:", gs.best_params_)
        print(f"Accuracy: {acc:.4f}")
        print("Classification report:\n", report)

        self.results[name] = {
            "best_params": gs.best_params_,
            "accuracy": acc,
            "report": report,
            "model": best,
        }

    def run(self) -> Dict[str, Dict[str, Any]]:
        y = self.df[self.target]
        X = self.df.drop(columns=[self.target])
        pre = make_preprocessor(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        self._train_one(
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1, 10], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]},
            X_train, y_train, X_test, y_test, pre,
        )

        self._train_one(
            "KNN",
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 11], "clf__weights": ["uniform", "distance"]},
            X_train, y_train, X_test, y_test, pre,
        )

        self._train_one(
            "SVC",
            SVC(probability=False, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"], "clf__gamma": ["scale", "auto"]},
            X_train, y_train, X_test, y_test, pre,
        )

        self._train_one(
            "ANN(MLPClassifier)",
            MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
            {"clf__hidden_layer_sizes": [(64,), (64, 32)], "clf__alpha": [1e-4, 1e-3]},
            X_train, y_train, X_test, y_test, pre,
        )

        # choose best by accuracy (highest)
        best_name = max(self.results, key=lambda k: self.results[k]["accuracy"])
        print(f"\n⭐ Best classifier: {best_name} (accuracy={self.results[best_name]['accuracy']:.4f})")
        return self.results


class App:
    def __init__(self):
        self.input = InputHandler()
        self.path = ""
        self.df = pd.DataFrame()
        self.target = ""
        self.ml_type = ""

    def run(self):
        print("=== ML App (OOP) ===")
        self.ml_type = self.input.ask_ml_type()
        self.path = self.input.ask_file_path()
        self.df = DataHandler(self.path).load()
        self.target = self.input.ask_target(self.df)

        validator = DataValidator(self.df, self.target)

        if self.ml_type == "regressor":
            ready, notes = validator.ready_for_regression()
        else:
            ready, notes = validator.ready_for_classification()

        # print readiness report
        print("\n--- Readiness report ---")
        if not notes:
            print("No issues detected.")
        else:
            for n in notes:
                print("-", n)

        if not ready:
            print("\n❌ Data is not ready. Please fix the issues above and rerun the app.")
            return

        # data ready → train and evaluate
        if self.ml_type == "regressor":
            results = RegressorTrainer(self.df, self.target).run()
            best_name = min(results, key=lambda k: results[k]["RMSE"])
            best_model = results[best_name]["model"]
            print(f"\nMy conclusion: Best model is {best_name} (RMSE={results[best_name]['RMSE']:.4f}, "
                  f"MAE={results[best_name]['MAE']:.4f}, R2={results[best_name]['R2']:.4f})")
        else:
            results = ClassifierTrainer(self.df, self.target).run()
            best_name = max(results, key=lambda k: results[k]["accuracy"])
            best_model = results[best_name]["model"]
            print(f"\nMy conclusion: Best model is {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

        # ask user to persist
        try:
            agree = input("Do you agree this is the best model? (yes/no): ").strip().lower()
        except EOFError:
            agree = "no"
        if agree.startswith("y"):
            try:
                fname = input("Enter filename to save (e.g., best_model.joblib): ").strip()
                if not fname:
                    fname = "best_model.joblib"
            except EOFError:
                fname = "best_model.joblib"
            joblib.dump(best_model, fname)
            print(f"💾 Saved best model to '{fname}'")
        else:
            print("Model was not saved.")


if __name__ == "__main__":
    try:
        App().run()
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
