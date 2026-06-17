import os
import sys
import time

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(DATASET_DIR)
sys.path.insert(0, ROOT_DIR)

DATASET_PATH = os.path.join(DATASET_DIR, "asl_landmarks.csv")
MODEL_DIR    = os.path.join(ROOT_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

def load_dataset(path: str):

    import csv
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                rows.append(row)

    labels   = [r[0] for r in rows]
    features = np.array([list(map(float, r[1:])) for r in rows], dtype=np.float32)
    return features, labels

def train(n_estimators: int = 200, random_state: int = 42) -> None:
    
    if not os.path.exists(DATASET_PATH):
        print("[train] No dataset found — generating synthetic data …")
        from generate_synthetic import generate_dataset
        generate_dataset(DATASET_PATH)

    print(f"[train] Loading dataset from {DATASET_PATH} …")
    X, y_raw = load_dataset(DATASET_PATH)
    print(f"[train] Loaded {len(X)} samples, {X.shape[1]} features")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    print(f"[train] Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"[train] Training RandomForest (n_estimators={n_estimators}) …")
    t0 = time.perf_counter()
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"[train] Training done in {elapsed:.1f}s")

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n[train] Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0,
    ))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf,     MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[train] Model saved  -> {MODEL_PATH}")
    print(f"[train] Encoder saved-> {ENCODER_PATH}")

if __name__ == "__main__":
    train()
