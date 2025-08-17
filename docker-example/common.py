# src/common.py
import os
import time
import json
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Defaults aligned with CIFAR-10
INPUT_SHAPE = 32
NUM_CLASSES = 10
CONV1_FILTERS = 64
CONV2_FILTERS = 32
DENSE1_SIZE = 64
DENSE2_SIZE = 64

def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data(dataset_name: str = "CIFAR10"):
    if dataset_name.upper() == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
        num_classes = 10
    elif dataset_name.upper() == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    return (X_train, y_train), (X_test, y_test), num_classes

def split_data_horizontal(X: np.ndarray, num_splits: int, split_index: int) -> np.ndarray:
    # Split along height (axis=1)
    H = X.shape[1]
    assert H % num_splits == 0, "Image height must be divisible by num_splits"
    h_chunk = H // num_splits
    start = split_index * h_chunk
    end = (split_index + 1) * h_chunk
    return X[:, start:end, :, :]

def build_split_model(
    input_height: int,
    input_width: int,
    num_splits: int,
    num_classes: int,
    conv1_filters: int = CONV1_FILTERS,
    conv2_filters: int = CONV2_FILTERS,
    dense1_size: int = DENSE1_SIZE,
    dense2_size: int = DENSE2_SIZE,
):
    # Reduce channels for split model
    conv1_f = conv1_filters // num_splits
    conv2_f = conv2_filters // num_splits
    dense2_s = dense2_size // num_splits

    model = Sequential(name=f"split_model_{num_splits}x")
    model.add(layers.Input(shape=(input_height, input_width, 3), name="input"))

    model.add(layers.Conv2D(conv1_f, (4, 4), padding="same", name="conv1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1"))
    model.add(layers.Activation("relu", name="relu1"))
    model.add(layers.Dropout(0.25, name="drop1"))

    model.add(layers.Conv2D(conv2_f, (3, 3), padding="same", name="conv2"))
    model.add(layers.Activation("relu", name="relu2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2"))
    model.add(layers.Dropout(0.25, name="drop2"))

    model.add(layers.Flatten(name="flat"))
    model.add(layers.Dense(dense1_size, name="dense1"))
    model.add(layers.Activation("relu", name="relu3"))
    model.add(layers.Dropout(0.25, name="drop3"))
    model.add(layers.Dense(dense2_s, name="dense2"))
    model.add(layers.Activation("tanh", name="tanh"))
    model.add(layers.Dense(num_classes, activation="softmax", name="logits"))

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_full_model(
    input_size: int,
    num_classes: int,
    conv1_filters: int = CONV1_FILTERS,
    conv2_filters: int = CONV2_FILTERS,
    dense1_size: int = DENSE1_SIZE,
    dense2_size: int = DENSE2_SIZE,
):
    model = Sequential(name="full_model")
    model.add(layers.Input(shape=(input_size, input_size, 3), name="input"))

    model.add(layers.Conv2D(conv1_filters, (4, 4), padding="same", name="conv1"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool1"))
    model.add(layers.Activation("relu", name="relu1"))
    model.add(layers.Dropout(0.25, name="drop1"))

    model.add(layers.Conv2D(conv2_filters, (3, 3), padding="same", name="conv2"))
    model.add(layers.Activation("relu", name="relu2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2"))
    model.add(layers.Dropout(0.25, name="drop2"))

    model.add(layers.Flatten(name="flat"))
    model.add(layers.Dense(dense1_size, name="dense1"))
    model.add(layers.Activation("relu", name="relu3"))
    model.add(layers.Dropout(0.25, name="drop3"))
    model.add(layers.Dense(dense2_size, name="dense2"))
    model.add(layers.Activation("tanh", name="tanh"))
    model.add(layers.Dense(num_classes, activation="softmax", name="logits"))

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def save_history_json(history, path: str):
    data = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def plot_histories(acc_pairs: List[Tuple[List[float], List[float]]], labels: List[str], out_path: str, title: str):
    plt.figure(figsize=(7,5))
    for (train_acc, val_acc), label in zip(acc_pairs, labels):
        plt.plot(train_acc, label=f"{label} - train")
        plt.plot(val_acc, label=f"{label} - val", linestyle="--")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def wait_for_files(paths: List[str], timeout_s: int = 86400, poll_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if all(os.path.exists(p) for p in paths):
            return
        time.sleep(poll_s)
    missing = [p for p in paths if not os.path.exists(p)]
    raise TimeoutError(f"Timed out waiting for files: {missing}")

def merge_conv_weights_from_splits(
    full_model,
    split_weight_paths: List[str],
    num_splits: int,
    input_height_split: int,
    input_width: int,
    num_classes: int,
):
    """
    Merge only convolutional layers:
    - conv1: concatenate split kernels along out_channels.
    - conv2: place split kernels in block-diagonal fashion:
      first split fills [0:half_in, 0:half_out], second split fills [half_in:, half_out:], rest zeros.
    Dense layers remain as initialized.
    """
    # Instantiate split models to load weights and extract conv kernels
    split_models = []
    for _ in range(num_splits):
        sm = build_split_model(
            input_height=input_height_split,
            input_width=input_width,
            num_splits=num_splits,
            num_classes=num_classes,
        )
        split_models.append(sm)

    for sm, path in zip(split_models, split_weight_paths):
        sm.load_weights(path)

    # Retrieve full conv weights
    W1_full, b1_full = full_model.get_layer("conv1").get_weights()
    W2_full, b2_full = full_model.get_layer("conv2").get_weights()

    # Shapes
    # conv1: (k1, k1, in_ch=3, out_ch=CONV1_FILTERS)
    # conv2: (k2, k2, in_ch=CONV1_FILTERS, out_ch=CONV2_FILTERS)
    k1_out = W1_full.shape[-1]
    k2_in = W2_full.shape[-2]
    k2_out = W2_full.shape[-1]

    # Arrays to fill
    W1_new = np.zeros_like(W1_full)
    b1_new = np.zeros_like(b1_full)
    W2_new = np.zeros_like(W2_full)
    b2_new = np.zeros_like(b2_full)

    # Each split conv1 out_ch
    half_out1 = k1_out // num_splits
    half_in2 = k2_in // num_splits
    half_out2 = k2_out // num_splits

    for idx, sm in enumerate(split_models):
        W1_s, b1_s = sm.get_layer("conv1").get_weights()
        W2_s, b2_s = sm.get_layer("conv2").get_weights()

        # conv1: stack along out_channels
        out_slice1 = slice(idx * half_out1, (idx + 1) * half_out1)
        W1_new[:, :, :, out_slice1] = W1_s
        b1_new[out_slice1] = b1_s

        # conv2: block-diagonal placement
        in_slice2 = slice(idx * half_in2, (idx + 1) * half_in2)
        out_slice2 = slice(idx * half_out2, (idx + 1) * half_out2)
        W2_new[:, :, in_slice2, out_slice2] = W2_s
        b2_new[out_slice2] = b2_s

    # Set merged conv weights back to the full model
    full_model.get_layer("conv1").set_weights([W1_new, b1_new])
    full_model.get_layer("conv2").set_weights([W2_new, b2_new])

    return full_model