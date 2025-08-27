#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 03.py

import pandas as pd # type: ignore
import pickle
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore

DATA_CSV = "./data/asl_features.csv"
MODEL_PATH = "./model/asl_model.pkl"

def main():
    df = pd.read_csv(DATA_CSV)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Test Accuracy = {acc:.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf}, f)
    print(f"Model saved → {MODEL_PATH}")

if __name__ == "__main__":
    main()
