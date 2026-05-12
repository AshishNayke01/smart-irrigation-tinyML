"""
01_generate_dataset.py  (PRODUCTION)
=====================================
Dataset generation for the 2‑model irrigation system.

NORMALISATION CONTRACT (same across all files):
  temp_norm  = temp  / 45.0
  hum_norm   = hum   / 100.0
  soil_norm  = soil  / 100.0

HARD DECISION RULES:
  soil < 20  → firmware override (model never sees these)
  soil < 30  → irrigate
  temp > 35 AND hum < 40 → irrigate
  else       → no irrigate
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES  = 4500
NOISE_STD  = {"temp": 0.5, "humidity": 1.0, "soil": 1.0}
SOFT_WIDTH = 5.0

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical bounds
TEMP_MIN, TEMP_MAX = 10.0, 45.0
HUM_MIN,  HUM_MAX  = 20.0, 100.0
SOIL_MIN, SOIL_MAX = 0.0, 100.0

# Normalisation divisors – MUST match firmware NORM_TEMP etc.
NORM_TEMP = 45.0
NORM_HUM  = 100.0
NORM_SOIL = 100.0

# Firmware hard override
SOIL_HARD_OVERRIDE = 20.0

# ── Helpers ────────────────────────────────────────────────────────────
def _sigmoid(x, k=1.0):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-k * x)),
                    np.exp(k * x) / (1.0 + np.exp(k * x)))

# ── Raw sample generation (noise applied before clipping) ─────────────
def generate_raw_samples(n):
    temp     = rng.uniform(TEMP_MIN - 1.0, TEMP_MAX + 1.0, n)
    humidity = rng.uniform(HUM_MIN  - 2.0, HUM_MAX  + 2.0, n)
    soil     = rng.uniform(SOIL_MIN - 2.0, SOIL_MAX + 2.0, n)

    temp     += rng.normal(0, NOISE_STD["temp"],     n)
    humidity += rng.normal(0, NOISE_STD["humidity"], n)
    soil     += rng.normal(0, NOISE_STD["soil"],     n)

    temp     = np.clip(temp,     TEMP_MIN, TEMP_MAX)
    humidity = np.clip(humidity, HUM_MIN,  HUM_MAX)
    soil     = np.clip(soil,     SOIL_MIN, SOIL_MAX)

    return pd.DataFrame({"temp": temp, "humidity": humidity, "soil": soil})

# ── Soft labels ────────────────────────────────────────────────────────
def soft_label(temp, humidity, soil):
    k = 5.0 / SOFT_WIDTH
    p_soil   = _sigmoid(30.0 - soil, k)
    p_temp_h = _sigmoid(temp - 35.0, k)
    p_hum_d  = _sigmoid(40.0 - humidity, k)
    p_hotdry = p_temp_h * p_hum_d
    p_irrigate = np.maximum(p_soil, p_hotdry)
    labels = (rng.uniform(0, 1, len(p_irrigate)) < p_irrigate).astype(np.int32)
    return labels, p_irrigate

# ── Hard‑rule labels (for evaluation only) ─────────────────────────────
def hard_rule_label(temp, humidity, soil):
    return ((soil < 30.0) | ((temp > 35.0) & (humidity < 40.0))).astype(np.int32)

# ── Normalisation ──────────────────────────────────────────────────────
def normalize(df):
    norm = pd.DataFrame()
    norm["temp_norm"] = df["temp"] / NORM_TEMP
    norm["hum_norm"]  = df["humidity"] / NORM_HUM
    norm["soil_norm"] = df["soil"] / NORM_SOIL
    return norm

# ── Calibration grid ───────────────────────────────────────────────────
def build_calibration_grid():
    temp_vals = np.arange(TEMP_MIN, TEMP_MAX + 1, 5)
    hum_vals  = np.arange(HUM_MIN,  HUM_MAX  + 1, 10)
    soil_vals = np.arange(SOIL_MIN, SOIL_MAX + 1, 5)
    rows = []
    for t in temp_vals:
        for h in hum_vals:
            for s in soil_vals:
                rows.append([t / NORM_TEMP, h / NORM_HUM, s / NORM_SOIL])
    return np.array(rows, dtype=np.float32)

# ── Main ───────────────────────────────────────────────────────────────
def main():
    raw = generate_raw_samples(N_SAMPLES)
    labels, probs = soft_label(raw.temp, raw.humidity, raw.soil)
    hard_labels = hard_rule_label(raw.temp, raw.humidity, raw.soil)
    normed = normalize(raw)

    df = pd.concat([raw, normed], axis=1)
    df["label"]      = labels
    df["label_prob"] = probs
    df["hard_label"] = hard_labels

    # Exclude hard‑override zone (soil < 20)
    df_override = df[df.soil < SOIL_HARD_OVERRIDE].copy()
    df_model    = df[df.soil >= SOIL_HARD_OVERRIDE].reset_index(drop=True)

    # 70 / 15 / 15 split
    idx = rng.permutation(len(df_model))
    n_train = int(0.70 * len(df_model))
    n_val   = int(0.15 * len(df_model))
    train = df_model.iloc[idx[:n_train]].reset_index(drop=True)
    val   = df_model.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    test  = df_model.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    # Class weights for imbalance
    n_pos = train.label.sum()
    n_neg = len(train) - n_pos
    w_pos = len(train) / (2.0 * n_pos) if n_pos else 1.0
    w_neg = len(train) / (2.0 * n_neg) if n_neg else 1.0
    class_weights = {0: float(w_neg), 1: float(w_pos)}

    # Calibration grid
    cal_grid = build_calibration_grid()

    # Save artifacts
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(  OUTPUT_DIR / "val.csv",   index=False)
    test.to_csv( OUTPUT_DIR / "test.csv",  index=False)
    df.to_csv(   OUTPUT_DIR / "full.csv",  index=False)
    df_override.to_csv(OUTPUT_DIR / "override_samples.csv", index=False)
    np.save(OUTPUT_DIR / "calibration_grid.npy", cal_grid)

    with open(OUTPUT_DIR / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)

    norm_contract = {
        "NORM_TEMP": NORM_TEMP,
        "NORM_HUM":  NORM_HUM,
        "NORM_SOIL": NORM_SOIL,
        "feature_order": ["temp_norm", "hum_norm", "soil_norm"],
        "SOIL_HARD_OVERRIDE": SOIL_HARD_OVERRIDE,
        "DECISION_THRESHOLD": 0.45,
    }
    with open(OUTPUT_DIR / "normalization_contract.json", "w") as f:
        json.dump(norm_contract, f, indent=2)

    print(f"\n{'='*55}")
    print(" DATASET GENERATED")
    print(f"{'='*55}")
    print(f"  Total samples        : {len(df)}")
    print(f"  Override zone (soil<20): {len(df_override)} (excluded from training)")
    print(f"  Train / val / test   : {len(train)} / {len(val)} / {len(test)}")
    print(f"  Train label balance  : irrigate {train.label.mean()*100:.1f}%")
    print(f"  Class weights        : {class_weights}")
    print(f"  Calibration grid     : {cal_grid.shape[0]} samples")

if __name__ == "__main__":
    main()