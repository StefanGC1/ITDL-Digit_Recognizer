import os
import pandas
import numpy as np

import tensorflow as tf
import keras

def main():
    model_path = os.path.join('models', 'digit_nn.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found.")
    model = keras.models.load_model(model_path)

    test_csv = os.path.join('image_data', 'test.csv')
    if not os.path.exists(test_csv):
        raise FileNotFoundError("Test data not found.")

    test_df = pandas.read_csv(test_csv)
    X_test = test_df.values.astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Make predictions and extract number with highest confidence
    print("Making predictions...")
    predictions = model.predict(X_test, batch_size=256)
    labels = np.argmax(predictions, axis=1)

    submission_df = pandas.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels})
    submission_path = os.path.join('submission', 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print("Submission created")


if __name__ == "__main__":
    main()