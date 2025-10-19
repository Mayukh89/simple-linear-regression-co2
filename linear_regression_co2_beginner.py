#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Regression on CO2 Emissions — Beginner, Single-File Script
=================================================================
This script teaches AND runs a complete simple linear regression workflow on a real dataset.

Features:
- Loads FuelConsumptionCo2.csv from the web (or a local path via --csv).
- Exploratory data peeks (sample, describe).
- Train/test split (80/20).
- Trains LinearRegression with:
    (A) Single feature: ENGINESIZE
    (B) Single feature: FUELCONSUMPTION_COMB
    (C) Two features: ENGINESIZE + FUELCONSUMPTION_COMB (optional, on by default)
- Evaluates with MAE, MSE, RMSE, R^2.
- Saves helpful plots into ./plots/
- Clean, beginner-friendly printouts.

Run:
-----
python linear_regression_co2_beginner.py
# or specify a local CSV (must have the same column names):
python linear_regression_co2_beginner.py --csv ./FuelConsumptionCo2.csv

CSV columns used:
-----------------
ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB, CO2EMISSIONS
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Utils
# -----------------------------
def ensure_plot_dir(dirpath: str = "plots") -> Path:
    p = Path(dirpath)
    p.mkdir(parents=True, exist_ok=True)
    return p


def eval_and_print(y_true, y_pred, header: str = "Evaluation"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {header} ===")
    print(f"MAE : {mae:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2 : {r2:.3f}")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def scatter_and_line(X, y, model, xlabel: str, ylabel: str, title: str, savepath: Path):
    plt.figure()
    plt.scatter(X, y, s=12)
    # For a clean line plot, sort X:
    order = np.argsort(X.ravel())
    X_sorted = X[order]
    y_line = model.predict(X_sorted)
    plt.plot(X_sorted, y_line, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def scatter_actual_vs_pred(y_true, y_pred, title: str, savepath: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=12)
    # 45-degree reference line:
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


# -----------------------------
# Core training routines
# -----------------------------
def run_single_feature(df: pd.DataFrame, feature_col: str, outdir: Path):
    assert feature_col in df.columns, f"Missing column: {feature_col}"
    target_col = "CO2EMISSIONS"

    X = df[[feature_col]].to_numpy()
    y = df[[target_col]].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    print(f"\n>>> Simple Linear Regression with feature: {feature_col}")
    print(f"Learned line: y = {slope:.3f} * {feature_col} + {intercept:.3f}")

    # Fitted line on train
    scatter_and_line(
        X_train, y_train, model,
        xlabel=feature_col, ylabel=target_col,
        title=f"{feature_col} vs {target_col} (train)",
        savepath=outdir / f"fit_{feature_col.lower()}.png"
    )

    # Predictions & evaluation on test set
    y_pred = model.predict(X_test)
    eval_and_print(y_test, y_pred, header=f"Test set — {feature_col}")

    # Actual vs Predicted
    scatter_actual_vs_pred(
        y_test, y_pred,
        title=f"Actual vs Predicted — {feature_col}",
        savepath=outdir / f"actual_vs_pred_{feature_col.lower()}.png"
    )


def run_two_features(df: pd.DataFrame, features: list, outdir: Path):
    target_col = "CO2EMISSIONS"
    for c in features:
        assert c in df.columns, f"Missing column: {c}"

    X = df[features].to_numpy()
    y = df[[target_col]].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    coefs = [float(c) for c in model.coef_[0]]
    intercept = float(model.intercept_[0])

    terms = " + ".join([f"{coefs[i]:.3f}*{features[i]}" for i in range(len(features))])
    print(f"\n>>> Multiple Linear Regression with features: {features}")
    print(f"Learned plane: y = {terms} + {intercept:.3f}")

    # Evaluate
    y_pred = model.predict(X_test)
    eval_and_print(y_test, y_pred, header=f"Test set — {', '.join(features)}")

    # Actual vs Predicted
    scatter_actual_vs_pred(
        y_test, y_pred,
        title=f"Actual vs Predicted — {', '.join(features)}",
        savepath=outdir / "actual_vs_pred_two_features.png"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Beginner linear regression on CO2 emissions.")
    parser.add_argument("--csv", type=str, default="", help="Path to local FuelConsumptionCo2.csv (optional).")
    parser.add_argument("--skip-multi", action="store_true", help="Skip the 2-feature model.")
    args = parser.parse_args()

    # Load data (web URL by default, local CSV if provided)
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
    if args.csv and Path(args.csv).exists():
        print(f"Loading local CSV: {args.csv}")
        df = pd.read_csv(args.csv)
    else:
        print("Loading CSV from the web URL...")
        df = pd.read_csv(url)

    # Minimal column check
    needed_cols = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Quick peeks
    print("\n=== Random Sample (5 rows) ===")
    print(df.sample(5))
    print("\n=== Describe ===")
    print(df[needed_cols].describe())

    # Out directory for plots
    outdir = ensure_plot_dir("plots")

    # Single-feature: ENGINESIZE
    run_single_feature(df, "ENGINESIZE", outdir)

    # Single-feature: FUELCONSUMPTION_COMB
    run_single_feature(df, "FUELCONSUMPTION_COMB", outdir)

    # Two features (optional, enabled by default)
    if not args.skip_multi:
        run_two_features(df, ["ENGINESIZE", "FUELCONSUMPTION_COMB"], outdir)

    print(f"\nDone! Plots saved to: {outdir.resolve()}")
    print("Files include:")
    for p in sorted(outdir.glob(\"*.png\")):
        print(\" -\", p.name)


if __name__ == "__main__":
    main()
