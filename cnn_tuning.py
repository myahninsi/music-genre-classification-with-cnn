import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from image_data_loader import load_data_generators
import keras_tuner as kt

# Constants
IMG_HEIGHT = 300
IMG_WIDTH = 400
BATCH_SIZE = 32
EPOCHS_TUNING = 50
EPOCHS_FINAL = 20 # Number of epochs for final training after tuning
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

    # Tunable convolution layers
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
    max_trials=15,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='music_genre_cnn_tuning_with_layers'
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
