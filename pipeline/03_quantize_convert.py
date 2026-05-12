"""
03_quantize_convert.py  (PRODUCTION)
======================================
INT8 quantization + TFLite conversion for Lite and Full irrigation models.

Input  : models/keras/{lite,full}.keras
Output : models/tflite/{lite,full}_int8.tflite
         models/tflite/quantization_results.json

FIXES APPLIED:
  [F1] Calibration dataset from grid (data/calibration_grid.npy) – uniform coverage
  [F2] Normalisation contract verified before conversion
  [F3] Feature order explicitly verified: [temp_norm, hum_norm, soil_norm]
  [F4] Dual accuracy evaluation: vs soft labels AND vs hard rules
        Accuracy gate applied against hard‑rule accuracy (correct ground truth)
  [F5] Tensor shape & dtype verified post‑conversion for each model
  [F6] I/O float32 dtype confirmed (critical for firmware data.f[] access)
  [F7] Quantization scale check – warns if threshold 0.45 maps to non‑representable INT8 value
  [F8] Hard‑override zone samples excluded from accuracy evaluation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import json

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
KERAS_DIR  = Path("models/keras")
TFLITE_DIR = Path("models/tflite")
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

MAX_MODEL_SIZE_KB  = 300       # hard constraint
THRESHOLD          = 0.45      # must match firmware DECISION_THRESHOLD
ACCURACY_GATE      = 0.85      # minimum post‑quant hard‑rule accuracy
FEATURES           = ["temp_norm", "hum_norm", "soil_norm"]

# ── Normalisation contract verification ────────────────────────────────────
def verify_normalization_contract():
    path = DATA_DIR / "normalization_contract.json"
    if not path.exists():
        raise FileNotFoundError("Run 01_generate_dataset.py first")
    with open(path) as f:
        c = json.load(f)
    assert c["NORM_TEMP"]        == 45.0
    assert c["NORM_HUM"]         == 100.0
    assert c["NORM_SOIL"]        == 100.0
    assert c["feature_order"]    == FEATURES
    assert c["DECISION_THRESHOLD"] == THRESHOLD
    assert c["SOIL_HARD_OVERRIDE"] == 20.0
    print("[OK] Normalisation contract verified")
    return c

# ── Load test data (exclude override zone) ──────────────────────────────────
def load_test_data(contract):
    test = pd.read_csv(DATA_DIR / "test.csv")
    soil_ov = contract["SOIL_HARD_OVERRIDE"]
    n_total = len(test)
    test = test[test["soil"] >= soil_ov].copy()
    if len(test) < n_total:
        print(f"  [INFO] Excluded {n_total - len(test)} override‑zone samples from eval")
    X     = test[FEATURES].values.astype(np.float32)
    y_soft = test["label"].values.astype(np.float32)
    y_hard = test["hard_label"].values.astype(np.float32) if "hard_label" in test.columns else y_soft
    return X, y_soft, y_hard

# ── Load calibration grid ──────────────────────────────────────────────────
def load_calibration_data():
    grid_path = DATA_DIR / "calibration_grid.npy"
    if grid_path.exists():
        cal = np.load(grid_path).astype(np.float32)
        print(f"Calibration grid: {len(cal)} samples (from file)")
        return cal

    # Fallback (should not be needed, but safe)
    print("[WARN] calibration_grid.npy not found – building inline fallback")
    temp_vals = np.arange(10, 46, 5)
    hum_vals  = np.arange(20, 101, 10)
    soil_vals = np.arange(0, 101, 5)
    rows = []
    for t in temp_vals:
        for h in hum_vals:
            for s in soil_vals:
                rows.append([t / 45.0, h / 100.0, s / 100.0])
    return np.array(rows, dtype=np.float32)

# ── Rep dataset generator ──────────────────────────────────────────────────
def make_representative_dataset(cal_data):
    def gen():
        for sample in cal_data:
            yield [sample.reshape(1, 3).astype(np.float32)]
    return gen

# ── TFLite evaluator ───────────────────────────────────────────────────────
def run_tflite(tflite_path, X):
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    probs = np.empty(len(X), dtype=np.float32)
    for i, sample in enumerate(X):
        interp.set_tensor(in_idx, sample.reshape(1, 3).astype(np.float32))
        interp.invoke()
        probs[i] = float(interp.get_tensor(out_idx)[0][0])
    return probs

# ── Metrics ────────────────────────────────────────────────────────────────
def evaluate(probs, y_true, threshold, label=""):
    preds = (probs > threshold).astype(int)
    y     = y_true.astype(int)
    tp = np.sum((preds==1) & (y==1))
    tn = np.sum((preds==0) & (y==0))
    fp = np.sum((preds==1) & (y==0))
    fn = np.sum((preds==0) & (y==1))
    acc  = (tp+tn)/len(y)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    if label:
        print(f"    {label}: acc={acc*100:.2f}% prec={prec:.3f} rec={rec:.3f} F1={f1:.4f} FN={fn} FP={fp}")
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "tp":tp, "tn":tn, "fp":fp, "fn":fn}

# ── Tensor contract verification ───────────────────────────────────────────
def verify_tensor_contract(tflite_path, name):
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    assert inp["dtype"] == np.float32, \
        f"[{name}] Input dtype={inp['dtype']}, expected float32. Firmware data.f[] will fail."
    assert list(inp["shape"]) == [1, 3], \
        f"[{name}] Input shape={inp['shape']}, expected [1,3]. Feature mismatch."
    assert out["dtype"] == np.float32, \
        f"[{name}] Output dtype={out['dtype']}, expected float32."
    assert list(out["shape"]) == [1, 1], \
        f"[{name}] Output shape={out['shape']}, expected [1,1]."
    tensors = interp.get_tensor_details()
    n_int8  = sum(1 for t in tensors if t["dtype"] == np.int8)
    n_float = sum(1 for t in tensors if t["dtype"] == np.float32)
    print(f"    [{name}] Tensor contract OK — input=float32[1,3] output=float32[1,1] "
          f"internal: {n_int8} INT8, {n_float} float32")
    # Threshold representability
    int8_outs = [t for t in tensors if t["dtype"]==np.int8 and t["quantization"][0]>0]
    if int8_outs:
        t = int8_outs[-1]
        scale, zp = t["quantization"]
        thresh_int8 = THRESHOLD/scale + zp
        thresh_rounded = round(thresh_int8)
        effective = (thresh_rounded - zp) * scale
        error = abs(effective - THRESHOLD)
        if error > 0.01:
            print(f"    [{name}] WARNING: threshold {THRESHOLD} → INT8 {thresh_int8:.1f} "
                  f"→ rounds to {thresh_rounded} → effective {effective:.4f} (error={error:.4f})")
        else:
            print(f"    [{name}] Threshold representability OK (effective={effective:.4f}, error={error:.5f})")

# ── Quantize one model ─────────────────────────────────────────────────────
def quantize_model(name, cal_data, contract):
    print(f"\n{'='*55}")
    print(f" Quantizing: {name.upper()}")
    print(f"{'='*55}")
    keras_path  = KERAS_DIR / f"{name}.keras"
    tflite_path = TFLITE_DIR / f"{name}_int8.tflite"
    if not keras_path.exists():
        raise FileNotFoundError(f"Missing {keras_path}. Run 02_train_models.py first.")
    model = tf.keras.models.load_model(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations           = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset  = make_representative_dataset(cal_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type    = tf.float32   # firmware accesses data.f[]
    converter.inference_output_type   = tf.float32

    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_bytes = len(tflite_bytes)
    size_kb    = size_bytes / 1024
    print(f"  Size: {size_bytes} bytes ({size_kb:.2f} KB)")

    if size_kb > MAX_MODEL_SIZE_KB:
        raise RuntimeError(f"SIZE CONSTRAINT VIOLATED: {name} {size_kb:.2f} KB > {MAX_MODEL_SIZE_KB} KB")
    print(f"  Size OK (limit: {MAX_MODEL_SIZE_KB} KB)")

    # Tensor contract
    print("  Verifying tensor contract...")
    verify_tensor_contract(tflite_path, name)

    # Accuracy evaluation
    X_test, y_soft, y_hard = load_test_data(contract)
    probs = run_tflite(tflite_path, X_test)
    print(f"  Post‑quant accuracy (n={len(X_test)}):")
    m_soft = evaluate(probs, y_soft, THRESHOLD, label=f"vs soft‑labels @{THRESHOLD}")
    m_hard = evaluate(probs, y_hard, THRESHOLD, label=f"vs hard‑rules  @{THRESHOLD}")

    passed = m_hard["acc"] >= ACCURACY_GATE
    if passed:
        print(f"  [PASS] Hard‑rule accuracy {m_hard['acc']*100:.2f}% ≥ {ACCURACY_GATE*100:.0f}%")
    else:
        print(f"  [FAIL] Hard‑rule accuracy {m_hard['acc']*100:.2f}% < {ACCURACY_GATE*100:.0f}%")

    # Sample predictions
    _tflite_sample_predictions(tflite_path, name)

    return {
        "name":            name,
        "tflite_path":     str(tflite_path),
        "size_bytes":      int(size_bytes),
        "size_kb":         float(round(size_kb, 3)),
        "size_ok":         bool(size_kb <= MAX_MODEL_SIZE_KB),
        "acc_soft_labels": float(round(float(m_soft["acc"]), 4)),
        "acc_hard_rules":  float(round(float(m_hard["acc"]), 4)),
        "f1_hard_rules":   float(round(float(m_hard["f1"]),  4)),
        "recall":          float(round(float(m_hard["rec"]),  4)),
        "false_negatives": int(m_hard["fn"]),
        "acc_gate_passed": bool(passed),
    }

# ── Sample predictions from TFLite ─────────────────────────────────────────
def _tflite_sample_predictions(tflite_path, name):
    SAMPLES = [
        ([35/45, 60/100, 15/100], "soil=15 (firmware override)"),
        ([25/45, 70/100, 25/100], "soil=25 → irrigate"),
        ([38/45, 35/100, 50/100], "hot+dry → irrigate"),
        ([25/45, 70/100, 50/100], "moderate → no"),
        ([20/45, 80/100, 80/100], "cool+wet → no"),
        ([30/45, 50/100, 30/100], "boundary soil=30"),
    ]
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    print(f"\n  Sample predictions [{name}]:")
    for feat, desc in SAMPLES:
        x = np.array([feat], dtype=np.float32)
        interp.set_tensor(in_idx, x)
        interp.invoke()
        prob = float(interp.get_tensor(out_idx)[0][0])
        soil_raw = feat[2] * 100.0
        if soil_raw < 20.0:
            dec = "OVERRIDE (firmware)"
        elif prob > THRESHOLD:
            dec = "IRRIGATE"
        else:
            dec = "no irrigate"
        print(f"    {desc:<30} prob={prob:.4f} → {dec}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    contract = verify_normalization_contract()
    cal_data = load_calibration_data()

    results = []
    for name in ["lite", "full"]:
        try:
            r = quantize_model(name, cal_data, contract)
            results.append(r)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"\n[FATAL] {e}")
            raise

    # Summary
    print(f"\n{'='*65}")
    print(" QUANTIZATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<10} {'Size(KB)':>9} {'SoftAcc':>8} {'HardAcc':>8} {'F1':>7} {'Recall':>7} {'FN':>5} {'Gate':>6}")
    print("  " + "-"*63)
    all_pass = True
    for r in results:
        status = "PASS" if r["acc_gate_passed"] else "FAIL"
        if not r["acc_gate_passed"]:
            all_pass = False
        print(f"  {r['name']:<10} {r['size_kb']:>9.3f} "
              f"{r['acc_soft_labels']*100:>7.2f}% "
              f"{r['acc_hard_rules']*100:>7.2f}% "
              f"{r['f1_hard_rules']:>7.4f} "
              f"{r['recall']:>7.3f} "
              f"{r['false_negatives']:>5} "
              f"{status:>6}")

    # Ensure JSON-safe types
    def json_safe(obj):
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    results_safe = json_safe(results)

    out_path = TFLITE_DIR / "quantization_results.json"
    with open(out_path, "w") as f:
        json.dump(results_safe, f, indent=2)
    print(f"\nResults → {out_path}")

    if not all_pass:
        raise SystemExit("\n[FATAL] One or more models failed the accuracy gate. "
                         "Do NOT proceed to header conversion.")
    print("\n[OK] All models passed accuracy gate. Proceed to 04_convert_to_header.py")

if __name__ == "__main__":
    main()