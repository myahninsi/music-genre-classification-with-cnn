import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.image_data_loader import load_data_generators

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 200
BATCH_SIZE = 16
EPOCHS_TUNING = 30  # Reduced for faster tuning
EPOCHS_FINAL = 15
DATASET_PATH = 'Dataset/spectrograms_split'
MODEL_SAVE_PATH = 'models/best_model.keras'

# Load data
train_generator, val_generator, test_generator, class_weights_dict = load_data_generators(
    dataset_path=DATASET_PATH,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE,
    augment=True
)

NUM_CLASSES = train_generator.num_classes

# Define model building function
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

    for i in range(hp.Int('num_conv_layers', 2, 3)):  # Limit to 2–3 conv layers
        model.add(tf.keras.layers.Conv2D(
            filters=hp.Int(f'filters_{i}', 32, 96, step=32),  # Reduce upper limit
            kernel_size=hp.Choice(f'kernel_size_{i}', [3]),  # Stick to 3x3 only
            activation='relu'
        ))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense_units', 64, 128, step=64),  # Simpler dense layer
        activation='relu'
    ))

    model.add(tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1)))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Define the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Reduce number of trials to ease CPU load
    executions_per_trial=1,
    directory='tuner_results',
    project_name='music_genre_cnn_tuning_light'
)

# Start the search
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_TUNING,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    class_weight=class_weights_dict
)

# Retrieve best model
best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hp)

# Retrain best model
history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINAL,
    callbacks=[ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)],
    class_weight=class_weights_dict
)

# Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save tuning summary
with open("tuner_results/best_hyperparameters.txt", "w") as f:
    f.write(str(best_hp.values))
