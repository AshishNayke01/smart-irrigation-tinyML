"""
04_convert_to_header.py  (CORRECTED)
=====================================
Converts .tflite → C/C++ header for Arduino / ESP32‑S3.
FIX: multiline comment properly enclosed.
"""

from pathlib import Path
import json

TFLITE_DIR = Path("models/tflite")
HEADER_DIR = Path("models/headers")
DATA_DIR   = Path("data")
HEADER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["lite", "full"]
BYTES_PER_LINE = 12

def load_normalization_contract():
    path = DATA_DIR / "normalization_contract.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "NORM_TEMP": 45.0,
        "NORM_HUM": 100.0,
        "NORM_SOIL": 100.0,
        "feature_order": ["temp_norm", "hum_norm", "soil_norm"],
        "DECISION_THRESHOLD": 0.45,
        "SOIL_HARD_OVERRIDE": 20.0,
    }

def tflite_to_header(name, contract):
    tflite_path = TFLITE_DIR / f"{name}_int8.tflite"
    header_path = HEADER_DIR / f"{name}_model.h"

    if not tflite_path.exists():
        raise FileNotFoundError(f"Missing {tflite_path}. Run 03_quantize_convert.py first.")

    with open(tflite_path, "rb") as f:
        data = f.read()

    n_bytes  = len(data)
    n_kb     = n_bytes / 1024
    var_name = f"g_{name}_model_data"
    guard    = f"IRRIGATION_{name.upper()}_MODEL_H"

    # Format hex body
    hex_lines = []
    for i in range(0, n_bytes, BYTES_PER_LINE):
        chunk = data[i:i+BYTES_PER_LINE]
        hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
        hex_lines.append(f"  {hex_str},")
    if hex_lines:
        hex_lines[-1] = hex_lines[-1].rstrip(",")
    hex_body = "\n".join(hex_lines)

    feat = contract.get("feature_order", ["temp_norm","hum_norm","soil_norm"])
    # Proper /* ... */ block
    norm_comment = (
        f"/*\n"
        f" * Normalisation (MUST match firmware NORM_* constants):\n"
        f" *   {feat[0]} = temp     / {contract.get('NORM_TEMP', 45.0)}\n"
        f" *   {feat[1]} = humidity / {contract.get('NORM_HUM', 100.0)}\n"
        f" *   {feat[2]} = soil     / {contract.get('NORM_SOIL', 100.0)}\n"
        f" * Feature order: [{', '.join(feat)}]\n"
        f" * Decision threshold : {contract.get('DECISION_THRESHOLD', 0.45)}\n"
        f" * Hard override      : soil < {contract.get('SOIL_HARD_OVERRIDE', 20.0)}  (firmware bypasses model)\n"
        f" * Input  : float32 [1, 3]  (data.f[] access)\n"
        f" * Output : float32 [1, 1]  (data.f[0] access)\n"
        f" * Internal ops : INT8\n"
        f" */"
    )

    header_content = f"""\
// AUTO-GENERATED — DO NOT EDIT MANUALLY
// Regenerate with: python 04_convert_to_header.py
//
// Model  : {name.capitalize()} irrigation model (INT8 quantized, float32 I/O)
// Source : {tflite_path}
// Size   : {n_bytes} bytes ({n_kb:.2f} KB)
//
{norm_comment}
//
// Usage:
//   #include "{name}_model.h"
//   const tflite::Model* model = tflite::GetModel({var_name});

#ifndef {guard}
#define {guard}

#include <pgmspace.h>

// PROGMEM forces placement in Flash, NOT SRAM.
// alignas(16) is required by Xtensa LX7 SIMD kernels in TFLite Micro.

alignas(16) const unsigned char {var_name}[] PROGMEM = {{
{hex_body}
}};

const unsigned int {var_name}_len = {n_bytes}U;

#endif  // {guard}
"""

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header_content)

    print(f"  [{name}] {n_bytes} bytes ({n_kb:.2f} KB) → {header_path}")
    return {"name": name, "size_bytes": n_bytes, "size_kb": n_kb, "header": str(header_path)}

def main():
    contract = load_normalization_contract()
    print(f"Normalisation contract: "
          f"NORM_TEMP={contract['NORM_TEMP']}  "
          f"NORM_HUM={contract['NORM_HUM']}  "
          f"NORM_SOIL={contract['NORM_SOIL']}")

    results = []
    for name in MODEL_NAMES:
        try:
            r = tflite_to_header(name, contract)
            results.append(r)
        except FileNotFoundError as e:
            print(f"\n[FATAL] {e}")
            raise SystemExit(1)

    print(f"\nHeaders written to: {HEADER_DIR}/")
    print("Copy these files into your Arduino sketch folder.")
    print("Include in .ino:")
    for r in results:
        print(f'  #include "{r["name"]}_model.h"')

if __name__ == "__main__":
    main()