import os
import random
import numpy as np
import pandas
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
import keras
from keras import layers


def load_data(validation_size=0.1):
    train_csv = os.path.join("image_data", "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} not found")
    
    train_df = pandas.read_csv(train_csv)
    labels = train_df.pop("label").values.astype("int64")
    # Normalize pixel data to range (0, 1)
    images = train_df.values.astype("float32") / 255.0

    # Reshape data from 1 dimension to 4 (-1, 28, 28, 1), where -1 is batch size
    # left as a placeholder
    images = images.reshape(-1, 28, 28, 1)
    print(f"Dataset shape : {images.shape}, labels shape: {labels.shape}")

    return train_test_split(
        images,
        labels,
        test_size=validation_size,
        stratify=labels,
        random_state=42
    )

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)), # Input layer

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax") # Output, prediction layer, with probabilities
    ])

    return model

def main():
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Set randomness seed. This allows for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, X_val, y_train, y_val = load_data()
    model = build_model()

    learning_rate = 0.00259302 # Originally 0.001
    batch_size = 256 # Originally 64

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        # Test for ReduceLROnPlateau callback
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.0005, # Allow the model to make final small improvements
            patience=7,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val), # Alternative to using validation_split=0.1
        batch_size=batch_size,
        epochs=30,
        callbacks=callbacks
    )

    # Final prediction report
    y_val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print(classification_report(y_val, y_val_pred, digits=4))

    # Save model
    model_path = os.path.join(MODEL_DIR, "digit_nn.keras")
    model.save(model_path)

    # Graph loss and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.ylim(0, 0.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(MODEL_DIR, "loss_curve.png")
    plt.savefig(fig_path)


if __name__ == "__main__":
    main()