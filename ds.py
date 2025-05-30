import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump, load

"""
# === CONFIG ===
PCAP_FILE = "input/traffic.pcap"
EXTRACTED_CSV = "output/features.csv"
MODEL_FILE = "output/model.joblib"
PREDICTIONS_FILE = "output/predictions.csv"
LABELS_FILE = "input/ground_truth.csv"  # Optional: wenn vorhanden
"""


# === STEP 1: FEATURE EXTRACTION ===
def extract_features(pcap_path: str, output_csv: str):
    print(f"[+] Extracting features from {pcap_path}...")
    tshark_cmd = [
        "tshark",
        "-r",
        pcap_path,
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-E",
        "header=y",
        "-e",
        "frame.time_relative",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "ip.proto",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "frame.len",
    ]

    with open(output_csv, "w") as f:
        subprocess.run(tshark_cmd, stdout=f)
    print(f"[+] Features written to {output_csv}")


# === STEP 2: DATA PREPROCESSING ===
def preprocess_data(input_csv: str):
    print("[+] Preprocessing extracted features...")
    df = pd.read_csv(input_csv)
    df.fillna(0, inplace=True)  # Replace NaNs
    df["src_port"] = df[["tcp.srcport", "udp.srcport"]].max(axis=1)
    df["dst_port"] = df[["tcp.dstport", "udp.dstport"]].max(axis=1)

    df.drop(
        columns=["tcp.srcport", "udp.srcport", "tcp.dstport", "udp.dstport"],
        inplace=True,
    )

    # Encode IPs as integers (simple encoding)
    df["ip.src"] = df["ip.src"].astype("category").cat.codes
    df["ip.dst"] = df["ip.dst"].astype("category").cat.codes

    return df


# === STEP 3: MODEL TRAINING ===
def train_model(features: pd.DataFrame, labels: pd.Series):
    print("[+] Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("[+] Classification Report:")
    print(classification_report(y_test, y_pred))

    dump(model, MODEL_FILE)
    print(f"[+] Model saved to {MODEL_FILE}")


# === STEP 4: PREDICTION ===
def predict(model_path: str, feature_df: pd.DataFrame, output_csv: str):
    print("[+] Loading model and predicting...")
    model = load(model_path)
    predictions = model.predict(feature_df)
    feature_df["prediction"] = predictions
    feature_df.to_csv(output_csv, index=False)
    print(f"[+] Predictions saved to {output_csv}")


# === MAIN EXECUTION ===
def main():
    os.makedirs("output", exist_ok=True)
    extract_features(PCAP_FILE, EXTRACTED_CSV)

    df = preprocess_data(EXTRACTED_CSV)

    if os.path.exists(LABELS_FILE):
        gt = pd.read_csv(LABELS_FILE)
        df = df.loc[gt.index]  # Ensure alignment
        train_model(df.drop(columns=["frame.time_relative"]), gt["label"])
    else:
        predict(MODEL_FILE, df.drop(columns=["frame.time_relative"]), PREDICTIONS_FILE)


if __name__ == "__main__":
    main()
