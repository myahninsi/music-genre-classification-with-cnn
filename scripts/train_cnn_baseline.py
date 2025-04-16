import argparse
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from scripts.image_data_loader import load_data_generators
from scripts.utils import ensure_dirs, save_training_outputs

def run(
    dataset_path: str,
    model_output_path: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    epochs: int,
    model_path_artifact: str,
    metrics_artifact: str
) -> str:
    """
    Trains a baseline Convolutional Neural Network (CNN) on spectrogram images
    and saves both the trained model and its training metrics as artifacts.

    This function will:
    - Loads training and validation image data from the dataset path.
    - Builds a baseline CNN model for image classification.
    - Compiles the model with Adam optimizer and categorical crossentropy loss.
    - Trains the model using early stopping and model checkpoint callbacks.
    - Saves the best model and training history to model ouput path. 

    Parameters:
        dataset_path (str): Path to the dataset directory.
        model_output_path (str): Path to save the trained model.
        img_height (int): Height of the input images.
        img_width (int): Width of the input images.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        model_path_artifact (str): Path to save a text file containing the model path (for pipeline tracking).
        metrics_artifact (str): Path to save training metrics (accuracy/loss) in JSON format.

    Returns:
        model_output_path (str): Path where the trained model is saved.
    """
    ensure_dirs()

    # Load data
    train_generator, val_generator, _, class_weights_dict = load_data_generators(
        dataset_path=dataset_path,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        augment=True
    )
    num_classes = train_generator.num_classes

    # Define the model
    model = Sequential([
        Input(shape=(img_height, img_width, 3)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_output_path, monitor='val_loss', save_best_only=True, mode='min')
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )


    # Save training outputs
    save_training_outputs(
        model_path=model_output_path,
        history=history,
        model_path_artifact=model_path_artifact,
        metrics_artifact=metrics_artifact
    )

    return model_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(**config)