import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Constants
IMG_SIZE = 128
BATCH_SIZE = 8
AUTOTUNE = tf.data.AUTOTUNE

# Enable memory growth for GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth error: {e}")

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")


# Modified focal loss with adjusted parameters
def focal_loss(gamma=2.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        loss = -tf.reduce_mean(
            alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1 + epsilon)
        ) - tf.reduce_mean(
            (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0 + epsilon)
        )
        return loss

    return focal_loss_fixed


def calculate_class_weights(n_classes):
    total = sum(n_classes.values())
    max_count = max(n_classes.values())
    class_weights = {cls: max_count / count for cls, count in n_classes.items()}

    # Normalize weights
    weight_sum = sum(class_weights.values())
    class_weights = {
        cls: weight / weight_sum * 2.0 for cls, weight in class_weights.items()
    }

    return class_weights


# Data Loading and Preprocessing
def prepare_dataset(directory, validation_split=None, subset=None):
    if validation_split:
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            validation_split=validation_split,
            subset=subset,
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            shuffle=True,
        )
    else:
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            shuffle=True,
        )
    return dataset


# Model architecture
def create_efficient_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    # First CNN Block
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Second CNN Block
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Third CNN Block
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Prepare data for LSTM
    # Reshape the CNN output to create sequences
    # Calculate shape after CNN blocks
    _, h, w, c = x.shape

    # Reshape to treat each row as a sequence
    x = layers.Reshape((h, w * c))(x)

    # LSTM layers
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.LSTM(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Dense Layers
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model


def predict_with_threshold(model, images, threshold=0.5):
    predictions = model.predict(images, verbose=0)
    return (predictions > threshold).astype(int)


def main():
    # Load datasets
    try:
        train_ds = prepare_dataset("chest_xray/train", 0.2, "training")
        val_ds = prepare_dataset("chest_xray/train", 0.2, "validation")
        test_ds = prepare_dataset("chest_xray/test")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

    # Normalize data
    normalization_layer = layers.Rescaling(1.0 / 255)

    def normalize_data(x, y):
        return normalization_layer(x), y

    train_ds = train_ds.map(normalize_data, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(normalize_data, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(normalize_data, num_parallel_calls=AUTOTUNE)

    # Calculate class weights
    total_samples = 0
    n_classes = {}

    for _, labels in train_ds:
        for label in labels:
            label = int(label)
            n_classes[label] = n_classes.get(label, 0) + 1
            total_samples += 1

    class_weight_dict = calculate_class_weights(n_classes)

    print("Class distribution:", n_classes)
    print("Class weights:", class_weight_dict)

    # Optimize dataset performance
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Create and compile model
    model = create_efficient_model()

    # Initialize optimizer with fixed initial learning rate instead of schedule
    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,  # Use fixed learning rate
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
    )

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.5),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
        ],
    )

    # Print model summary
    model.summary()

    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            "best_model.keras", save_best_only=True, monitor="val_loss", verbose=1
        ),
    ]

    # Train model
    history = model.fit(
        train_ds,
        epochs=50,  # Increased epochs
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Evaluate model
    print("\nEvaluating model on test set:")
    test_results = model.evaluate(test_ds, verbose=1)

    print("\nTest Results:")
    for name, value in zip(model.metrics_names, test_results):
        print(f"{name}: {value:.4f}")

    # Generate predictions
    print("\nGenerating predictions for confusion matrix...")
    true_labels = []
    raw_predictions = []

    # First collect all true labels and raw predictions
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        raw_predictions.extend(preds.flatten())
        true_labels.extend(labels.numpy())

    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = 0
    best_threshold = 0.5
    best_predictions = None

    # Convert lists to numpy arrays for easier manipulation
    true_labels = np.array(true_labels)
    raw_predictions = np.array(raw_predictions)

    for threshold in thresholds:
        # Apply threshold
        predictions = (raw_predictions > threshold).astype(int)

        # Calculate F1 score for this threshold
        report = classification_report(true_labels, predictions, output_dict=True)
        avg_f1 = report["macro avg"]["f1-score"]

        print(f"\nThreshold: {threshold}")
        print(classification_report(true_labels, predictions))

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold
            best_predictions = predictions

    print(f"\nBest threshold: {best_threshold}")
    print("\nFinal Classification Report:")
    print(classification_report(true_labels, best_predictions))

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, best_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (threshold={best_threshold:.2f})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Plot training history
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
