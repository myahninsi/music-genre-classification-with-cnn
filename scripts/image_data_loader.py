import numpy as np
import os
from tensorflow.keras.preprocessing.image import DirectoryIterator
from typing import Tuple, Dict

from sklearn.utils.class_weight import compute_class_weight

def load_data_generators(
    dataset_path: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    augment: bool
) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, Dict[int, float]]:

    """
    Loads training, validation, and test generators from specified dataset directory.

    Parameters:
        dataset_path (str): Path to the dataset directory containing 'train', 'val', and 'test' folders.
        img_height (int): Height to resize images to.
        img_width (int): Width to resize images to.
        batch_size (int): Number of images per batch.
        augment (bool): Whether to apply augmentation to training images.

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, Dict[int, float]]:
            - train_generator: Generator for training data.
            - val_generator: Generator for validation data.
            - test_generator: Generator for test data.
            - class_weights_dict: Dictionary of class weights for balancing classes.
    """

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = test_val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Compute class weights for balancing
    labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_dict = dict(enumerate(class_weights))

    return train_generator, val_generator, test_generator, class_weights_dict