import json
import numpy as np
import os
import sys

def check_file(path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)
    else:
        print(f"[OK] Found {path}")

def main(tensor_dir):
    print("="*60)
    print(" Checking Tensor Directory:", tensor_dir)
    print("="*60)

    # -----------------------------
    # 1. Load meta
    # -----------------------------
    meta_path = os.path.join(tensor_dir, "meta.json")
    check_file(meta_path)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    print("\n[meta.json loaded]")
    print(json.dumps(meta, indent=2))

    # Shapes
    X_shape = tuple(meta["shapes"]["X_shape"])
    y_shape = tuple(meta["shapes"]["y_future_shape"])
    static_shape = tuple(meta["shapes"]["node_static_shape"])

    print("\nExpected X shape:", X_shape)
    print("Expected y shape:", y_shape)
    print("Expected static shape:", static_shape)

    # -----------------------------
    # 2. Load memmaps
    # -----------------------------
    X_path = os.path.join(tensor_dir, "X.dat")
    y_path = os.path.join(tensor_dir, "y_future.dat")
    static_path = os.path.join(tensor_dir, "node_static.npy")

    check_file(X_path)
    check_file(y_path)
    check_file(static_path)

    X = np.memmap(X_path, dtype=np.float32, mode="r", shape=X_shape)
    y = np.memmap(y_path, dtype=np.uint8, mode="r", shape=y_shape)
    node_static = np.load(static_path)

    # -----------------------------
    # 3. Print basic checks
    # -----------------------------
    print("\n=== BASIC CHECKS ===")
    print("X: dtype=", X.dtype, " shape=", X.shape)
    print("y: dtype=", y.dtype, " shape=", y.shape)
    print("static: dtype=", node_static.dtype, " shape=", node_static.shape)

    # -----------------------------
    # 4. Compute and print statistics
    # -----------------------------
    print("\n=== X Stats ===")
    print("X min:", float(X.min()))
    print("X max:", float(X.max()))
    print("X mean:", float(X.mean()))
    print("X NaN count:", int(np.isnan(X).sum()))

    print("\n=== y Stats ===")
    print("Total positives in y:", int(y.sum()))
    print("Positive %:", float(y.sum() / y.size) * 100)

    print("\n=== Static Feature Stats ===")
    print("Static min:", float(node_static.min()))
    print("Static max:", float(node_static.max()))
    print("Static mean:", float(node_static.mean()))
    print("Static NaN count:", int(np.isnan(node_static).sum()))

    # -----------------------------
    # 5. Sanity: time alignment check
    # -----------------------------
    print("\n=== SANITY CHECK ===")
    if X_shape[0] != y_shape[0]:
        print("[WARNING] Node count mismatch between X and y!")
    else:
        print("[OK] Node count matches")

    if X_shape[1] <= y_shape[1]:
        print("[WARNING] X has fewer time steps than y!? Check horizon.")
    else:
        print("[OK] X time dimension is larger than y.")

    print("\nAll checks complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("   python check_tensors.py <tensor_dir>")
        sys.exit(1)

    main(sys.argv[1])
