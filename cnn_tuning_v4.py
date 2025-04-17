import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from image_data_loader import load_data_generators
import os

# Constants
IMG_HEIGHT = 300
IMG_WIDTH = 400
BATCH_SIZE = 8  # reduced for memory safety
EPOCHS_TUNING = 10
EPOCHS_FINAL = 8
DATASET_PATH = 'Dataset/spectrograms_split'
MODEL_SAVE_PATH = 'models/finetuned_model.keras'
BASE_MODEL_PATH = 'models/cnn_baseline.keras'
GCS_MODEL_PATH = 'gs://cnn-music-bucket/models/finetuned_model.keras'
GCS_HPARAM_PATH = 'gs://cnn-music-bucket/tuner_results/best_hyperparameters_finetune.txt'

# Load data
train_generator, val_generator, test_generator, class_weights_dict = load_data_generators(
    dataset_path=DATASET_PATH,
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH,
    batch_size=BATCH_SIZE,
    augment=True
)

NUM_CLASSES = train_generator.num_classes

# Define tunable model builder
def build_finetune_model(hp):
    base_model = load_model(BASE_MODEL_PATH)

    # Force the model to be 'built' by running it on dummy data
    dummy_input = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_output = base_model(dummy_input)  # now it's 'called'

    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Add dropout and dense layers on top
    x = Dropout(hp.Float('dropout', 0.2, 0.4, step=0.1))(base_output)
    x = Dense(
        hp.Int('dense_units', 64, 128, step=32),
        activation='relu'
    )(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=dummy_input, outputs=output)

    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up tuner
tuner = kt.RandomSearch(
    build_finetune_model,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='finetune_from_baseline'
)

# Run tuning
tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_TUNING,
    callbacks=[EarlyStopping(patience=3, min_delta=0.01, restore_best_weights=True)],
    class_weight=class_weights_dict
)

# Build & retrain best model
best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hp)

history = best_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINAL,
    callbacks=[ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)],
    class_weight=class_weights_dict
)

# Evaluate on test data
test_loss, test_acc = best_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save best hyperparameters
with open("tuner_results/best_hyperparameters_finetune.txt", "w") as f:
    f.write(str(best_hp.values))

# Upload model + hyperparameters to GCS
os.system(f"gsutil cp {MODEL_SAVE_PATH} {GCS_MODEL_PATH}")
os.system(f"gsutil cp tuner_results/best_hyperparameters_finetune.txt {GCS_HPARAM_PATH}")
