import argparse
import random
import numpy as np

import optuna
import tensorflow as tf
import keras

from train_model import load_data, build_model

def trial_func(trial: optuna.Trial):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.005, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    X_train, X_val, y_train, y_val = load_data()
    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.001,
            patience=4,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=30,
        callbacks=callbacks,
        verbose=0
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    return best_val_accuracy

def main(trials: int):
    storage = "sqlite:///hpo_results.db"
    study = optuna.create_study(
        storage=storage,
        study_name="digit_recognizer_hpo",
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(trial_func, n_trials=trials, show_progress_bar=True)

    print(f"Best trial: {study.best_trial.number}")
    print("Best hyper parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"Best accuracy: {study.best_trial.value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for digit recognizer nn.")
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of trials to be executed by optuna. default=20"
    )
    args = parser.parse_args()

    main(args.trials)