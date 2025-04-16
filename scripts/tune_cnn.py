import argparse
import yaml
import tensorflow as tf
import keras_tuner as kt
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
    Performs hyperparameter tuning on a CNN model using Keras Tuner, then retrains
    the best model and saves both the model and metrics. 

    This function will:
        - Load training and validation image data from the dataset path.
        - Define a CNN architecture with tunable hyperparameters using Keras Tuner.
        - Perform random search tuning over a predefined search space.
        - Retrieve the best hyperparameters based on validation accuracy.
        - Retrain the best model from scratch using those optimal hyperparameters.
        - Save the final trained model to disk.
        - Save the training history and model path as reusable artifacts.
        - Export the best hyperparameter configuration to a text file for reference.

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
        model_output_path (str): Path where the tuned model is saved.
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

    def build_model(hp):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)))

        for i in range(hp.Int('num_conv_layers', 2, 4)):
            model.add(tf.keras.layers.Conv2D(
                filters=hp.Int(f'filters_{i}', 32, 128, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', [3, 5]),
                activation='relu'
            ))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(
            units=hp.Int('dense_units', 64, 256, step=64),
            activation=hp.Choice('dense_activation', ['relu', 'tanh'])
        ))
        model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='cnn_tuning'
    )

    tuner.search(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        class_weight=class_weights_dict
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    history = best_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[ModelCheckpoint(model_output_path, save_best_only=True)],
        class_weight=class_weights_dict
    )

    
    save_training_outputs(
        model_path=model_output_path,
        history=history,
        model_path_artifact=model_path_artifact,
        metrics_artifact=metrics_artifact
    )

    with open("tuner_results/best_hyperparameters.txt", "w") as f:
        f.write(str(best_hp.values))

    return model_output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(**config)