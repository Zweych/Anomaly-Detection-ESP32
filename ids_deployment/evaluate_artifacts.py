
import json
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "deployment_artifacts" / "Primary"

FLOAT32_MODEL = ARTIFACT_DIR / "model_esp32_float32.keras"
INT8_MODEL = ARTIFACT_DIR / "model_esp32_int8.tflite"
SCALER_FILE = ARTIFACT_DIR / "esp32_scaler.npz"
REPLAY_FILE = ARTIFACT_DIR / "esp32_replay.csv"
METADATA_FILE = ARTIFACT_DIR / "esp32_metadata.json"

OUTPUT_DIR = ARTIFACT_DIR / "standalone_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WARMUP_RUNS = 100
BENCHMARK_RUNS = 1000


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print("=" * 60)
print("STANDALONE ARTIFACT EVALUATION")
print("=" * 60)

for path in [
    FLOAT32_MODEL,
    INT8_MODEL,
    SCALER_FILE,
    REPLAY_FILE,
    METADATA_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

THRESHOLD = float(metadata["threshold"])
FEATURES = metadata["features"]

replay = pd.read_csv(REPLAY_FILE)

scaler_data = np.load(SCALER_FILE)

scaler_mean = scaler_data["mean_"].astype(np.float32)
scaler_scale = scaler_data["scale_"].astype(np.float32)

X_raw = replay[FEATURES].to_numpy(dtype=np.float32)
y_true = replay["true_label"].to_numpy(dtype=np.int32)

X_scaled = (
    (X_raw - scaler_mean) / scaler_scale
).astype(np.float32)

print(f"Samples       : {len(X_scaled)}")
print(f"Features      : {len(FEATURES)}")
print(f"Threshold     : {THRESHOLD}")
print(f"TensorFlow    : {tf.__version__}")
print(f"Platform      : {platform.platform()}")


# ============================================================
# HELPER: MODEL SIZE
# ============================================================

def get_model_size(path):
    size_bytes = Path(path).stat().st_size

    return {
        "bytes": size_bytes,
        "KB": size_bytes / 1024,
        "MB": size_bytes / (1024 ** 2),
    }


# ============================================================
# HELPER: LATENCY BENCHMARK
# ============================================================

def benchmark_inference(
    inference_function,
    X,
    warmup=WARMUP_RUNS,
    runs=BENCHMARK_RUNS,
):
    sample = X[0:1]

    # Warm-up
    for _ in range(warmup):
        inference_function(sample)

    # Timed inference
    latency_results = []

    for _ in range(runs):
        start = time.perf_counter_ns()

        inference_function(sample)

        end = time.perf_counter_ns()

        latency_us = (end - start) / 1000.0

        latency_results.append(latency_us)

    latency = np.array(latency_results)

    return {
        "mean_us": float(np.mean(latency)),
        "median_us": float(np.median(latency)),
        "p95_us": float(np.percentile(latency, 95)),
        "min_us": float(np.min(latency)),
        "max_us": float(np.max(latency)),
        "std_us": float(np.std(latency)),
        "runs": runs,
        "warmup_runs": warmup,
    }


# ============================================================
# HELPER: CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "attack_recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "far": float(
            fp / (fp + tn)
            if (fp + tn) > 0 else 0.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 1. FLOAT32 MODEL
# ============================================================

print("\n" + "=" * 60)
print("FLOAT32 MODEL")
print("=" * 60)

model_float32 = tf.keras.models.load_model(
    FLOAT32_MODEL,
    compile=False,
)

def float32_inference(x):
    # Direct model invocation avoids model.predict()
    # overhead associated with its prediction pipeline.
    logits = model_float32(
        tf.convert_to_tensor(x),
        training=False,
    )

    return logits.numpy().reshape(-1)

float32_latency = benchmark_inference(
    float32_inference,
    X_scaled,
)

float32_logits = np.concatenate([
    float32_inference(X_scaled[i:i+1])
    for i in range(len(X_scaled))
])

float32_probability = 1.0 / (
    1.0 + np.exp(-np.clip(float32_logits, -80, 80))
)

float32_prediction = (
    float32_probability >= THRESHOLD
).astype(np.int32)

float32_metrics = calculate_metrics(
    y_true,
    float32_prediction,
)

float32_size = get_model_size(FLOAT32_MODEL)

print("Model size:")
print(f"  {float32_size['bytes']:,} bytes")
print(f"  {float32_size['KB']:.2f} KB")

print("Inference latency:")
for key, value in float32_latency.items():
    print(f"  {key}: {value}")


# ============================================================
# 2. INT8 TFLITE MODEL
# ============================================================

print("\n" + "=" * 60)
print("INT8 TFLITE MODEL")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=str(INT8_MODEL),
    num_threads=1,
)

interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_detail["quantization"]
output_scale, output_zero_point = output_detail["quantization"]

if input_scale <= 0 or output_scale <= 0:
    raise RuntimeError("Invalid INT8 quantization parameters.")


def int8_inference(x):
    # Float32 standardized input to INT8
    quantized_input = np.round(
        x / input_scale + input_zero_point
    )

    quantized_input = np.clip(
        quantized_input,
        -128,
        127,
    ).astype(np.int8)

    interpreter.set_tensor(
        input_detail["index"],
        quantized_input,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_detail["index"]
    )

    # Dequantize output logits
    logits = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

    return logits.reshape(-1)


int8_latency = benchmark_inference(
    int8_inference,
    X_scaled,
)

int8_logits = np.concatenate([
    int8_inference(X_scaled[i:i+1])
    for i in range(len(X_scaled))
])

int8_probability = 1.0 / (
    1.0 + np.exp(-np.clip(int8_logits, -80, 80))
)

int8_prediction = (
    int8_probability >= THRESHOLD
).astype(np.int32)

int8_metrics = calculate_metrics(
    y_true,
    int8_prediction,
)

int8_size = get_model_size(INT8_MODEL)

print("Model size:")
print(f"  {int8_size['bytes']:,} bytes")
print(f"  {int8_size['KB']:.2f} KB")

print("Inference latency:")
for key, value in int8_latency.items():
    print(f"  {key}: {value}")


# ============================================================
# 3. RESULT SUMMARY
# ============================================================

results = {
    "environment": {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "num_threads_int8": 1,
        "benchmark_runs": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "batch_size": 1,
    },
    "float32": {
        "model_size": float32_size,
        "latency": float32_latency,
        "metrics": float32_metrics,
    },
    "int8": {
        "model_size": int8_size,
        "latency": int8_latency,
        "metrics": int8_metrics,
    },
    "int8_quantization": {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    },
}

json_path = OUTPUT_DIR / "benchmark_results.json"

with open(json_path, "w") as f:
    json.dump(results, f, indent=4)


# ============================================================
# 4. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "true_label": y_true,
    "float32_prediction": float32_prediction,
    "int8_prediction": int8_prediction,
    "float32_probability": float32_probability,
    "int8_probability": int8_probability,
})

predictions.to_csv(
    OUTPUT_DIR / "benchmark_predictions.csv",
    index=False,
)


# ============================================================
# 5. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "Model": "PC Float32",
        "Accuracy (%)": float32_metrics["accuracy"] * 100,
        "Macro F1 (%)": float32_metrics["macro_f1"] * 100,
        "Attack Recall (%)": float32_metrics["attack_recall"] * 100,
        "FAR (%)": float32_metrics["far"] * 100,
        "Mean Latency (us)": float32_latency["mean_us"],
        "Median Latency (us)": float32_latency["median_us"],
        "P95 Latency (us)": float32_latency["p95_us"],
        "Model Size (bytes)": float32_size["bytes"],
    },
    {
        "Model": "PC INT8",
        "Accuracy (%)": int8_metrics["accuracy"] * 100,
        "Macro F1 (%)": int8_metrics["macro_f1"] * 100,
        "Attack Recall (%)": int8_metrics["attack_recall"] * 100,
        "FAR (%)": int8_metrics["far"] * 100,
        "Mean Latency (us)": int8_latency["mean_us"],
        "Median Latency (us)": int8_latency["median_us"],
        "P95 Latency (us)": int8_latency["p95_us"],
        "Model Size (bytes)": int8_size["bytes"],
    },
])

summary.to_csv(
    OUTPUT_DIR / "benchmark_summary.csv",
    index=False,
)

print("\n" + "=" * 60)
print("FINAL BENCHMARK SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\nResults saved:")
print(json_path)
print(OUTPUT_DIR / "benchmark_summary.csv")
print(OUTPUT_DIR / "benchmark_predictions.csv")

print("\nEvaluation completed.")
