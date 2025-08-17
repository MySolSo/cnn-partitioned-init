# src/train_split.py
import os
from common import (
    set_seed, load_data, split_data_horizontal, build_split_model,
    save_history_json, INPUT_SHAPE
)

def main():
    split_index = int(os.getenv("SPLIT_INDEX", "0"))
    num_splits = int(os.getenv("NUM_SPLITS", "2"))
    epochs = int(os.getenv("EPOCHS", "15"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    shared_dir = os.getenv("SHARED_DIR", "/workspace/shared")
    dataset = os.getenv("DATASET", "CIFAR10")
    seed = int(os.getenv("SEED", "42"))

    os.makedirs(os.path.join(shared_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(shared_dir, "histories"), exist_ok=True)

    set_seed(seed)
    (X_train, y_train), (X_test, y_test), num_classes = load_data(dataset)

    # Split along height
    X_train_split = split_data_horizontal(X_train, num_splits, split_index)
    X_test_split  = split_data_horizontal(X_test,  num_splits, split_index)

    model = build_split_model(
        input_height=INPUT_SHAPE // num_splits,
        input_width=INPUT_SHAPE,
        num_splits=num_splits,
        num_classes=num_classes,
    )

    history = model.fit(
        X_train_split, y_train,
        validation_data=(X_test_split, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save artifacts
    weight_path = os.path.join(shared_dir, "weights", f"split_{split_index}.weights.h5")
    hist_path = os.path.join(shared_dir, "histories", f"split_{split_index}.json")
    model.save_weights(weight_path)
    save_history_json(history, hist_path)
    print(f"[split{split_index}] Saved weights to {weight_path}")
    print(f"[split{split_index}] Saved history to {hist_path}")

if __name__ == "__main__":
    main()