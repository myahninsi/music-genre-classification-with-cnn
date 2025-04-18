from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight

def load_data_generators(
    DATASET_PATH='Dataset/spectrograms_split',
    IMG_HEIGHT=300,
    IMG_WIDTH=400,
    BATCH_SIZE=32,
    AUGMENT=True):

    """
    Loads training, validation, and test generators from specified dataset directory.

    Parameters:
        DATASET_PATH (str): Path to the dataset directory containing 'train', 'val', and 'test' folders.
        IMG_HEIGHT (int): Height to resize images to.
        IMG_WIDTH (int): Width to resize images to.
        BATCH_SIZE (int): Number of images per batch.
        AUGMENT (bool): Whether to apply augmentation to training images.

    Returns:
        train_generator, val_generator, test_generator, class_weights_dict
    """

    # Generate training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,         # slight rotation as music spectrograms tolerate this
        width_shift_range=0.1,     # horizontal shift
        height_shift_range=0.1,    # vertical shift
        zoom_range=0.1,            # zoom in/out
        horizontal_flip=False,     # don't flip spectrograms
        fill_mode='nearest'        # fill empty pixels
    )

    # Generate validation and test without augmentation and only rescaling to [0, 1]
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Load validation data
    val_generator = test_val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Load test data
    test_generator = test_val_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Get the class labels from the training generator
    labels = train_generator.classes

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))

    return train_generator, val_generator, test_generator, class_weights_dict