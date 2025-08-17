# src/merge_and_finetune.py
import os
import json
from common import (
    set_seed, load_data, build_full_model, build_split_model,
    merge_conv_weights_from_splits, wait_for_files,
    plot_histories, save_history_json, INPUT_SHAPE,
    CONV1_FILTERS, CONV2_FILTERS, DENSE1_SIZE, DENSE2_SIZE
)

def main():
    num_splits = int(os.getenv("NUM_SPLITS", "2"))
    epochs_merged = int(os.getenv("EPOCHS_MERGED", "20"))
    epochs_baseline = int(os.getenv("EPOCHS_BASELINE", "20"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    shared_dir = os.getenv("SHARED_DIR", "/workspace/shared")
    dataset = os.getenv("DATASET", "CIFAR10")
    seed = int(os.getenv("SEED", "123"))

    set_seed(seed)

    weights_dir = os.path.join(shared_dir, "weights")
    plots_dir = os.path.join(shared_dir, "plots")
    histories_dir = os.path.join(shared_dir, "histories")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(histories_dir, exist_ok=True)

    split_weight_paths = [os.path.join(weights_dir, f"split_{i}.weights.h5") for i in range(num_splits)]
    print("[merger] Waiting for split weights:", split_weight_paths)
    wait_for_files(split_weight_paths, timeout_s=86400, poll_s=5.0)
    print("[merger] All split weights found.")

    (X_train, y_train), (X_test, y_test), num_classes = load_data(dataset)

    # Build full model and merge
    full_model_merged = build_full_model(
        input_size=INPUT_SHAPE,
        num_classes=num_classes,
        conv1_filters=CONV1_FILTERS,
        conv2_filters=CONV2_FILTERS,
        dense1_size=DENSE1_SIZE,
        dense2_size=DENSE2_SIZE,
    )

    full_model_merged = merge_conv_weights_from_splits(
        full_model_merged,
        split_weight_paths=split_weight_paths,
        num_splits=num_splits,
        input_height_split=INPUT_SHAPE // num_splits,
        input_width=INPUT_SHAPE,
        num_classes=num_classes,
    )

    # Evaluate before fine-tune (optional)
    pre_loss, pre_acc = full_model_merged.evaluate(X_test, y_test, verbose=0)
    print(f"[merger] Before fine-tune - Test acc: {pre_acc:.4f}")

    hist_merged = full_model_merged.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs_merged,
        batch_size=batch_size,
        verbose=1
    )

    # Baseline full model trained from scratch
    set_seed(seed + 1)  # different init
    full_model_baseline = build_full_model(
        input_size=INPUT_SHAPE,
        num_classes=num_classes,
        conv1_filters=CONV1_FILTERS,
        conv2_filters=CONV2_FILTERS,
        dense1_size=DENSE1_SIZE,
        dense2_size=DENSE2_SIZE,
    )
    hist_baseline = full_model_baseline.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs_baseline,
        batch_size=batch_size,
        verbose=1
    )

    # Eval
    m_loss, m_acc = full_model_merged.evaluate(X_test, y_test, verbose=0)
    b_loss, b_acc = full_model_baseline.evaluate(X_test, y_test, verbose=0)
    print(f"[merger] After fine-tune - Merged Test acc: {m_acc:.4f}, Baseline Test acc: {b_acc:.4f}")

    # Save histories and metrics
    save_history_json(hist_merged, os.path.join(histories_dir, "merged.json"))
    save_history_json(hist_baseline, os.path.join(histories_dir, "baseline.json"))

    metrics = {
        "pre_finetune_acc": float(pre_acc),
        "merged_test_acc": float(m_acc),
        "baseline_test_acc": float(b_acc),
        "epochs_merged": epochs_merged,
        "epochs_baseline": epochs_baseline,
    }
    with open(os.path.join(shared_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    plot_histories(
        acc_pairs=[
            (hist_merged.history["accuracy"], hist_merged.history["val_accuracy"]),
            (hist_baseline.history["accuracy"], hist_baseline.history["val_accuracy"]),
        ],
        labels=["Merged", "Baseline"],
        out_path=os.path.join(plots_dir, "comparison_accuracy.png"),
        title="Merged vs Baseline - Accuracy",
    )

if __name__ == "__main__":
    main()