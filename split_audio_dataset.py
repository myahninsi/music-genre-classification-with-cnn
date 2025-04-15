import os
import shutil
import random

def split_audio_dataset(original_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits a folder of genre-labeled audio files into train (70%), validation(15%), and test sets(15%).

    Parameters:
        original_dir (str): Path to the original dataset directory containing genre folders. Assumes each genre folder contains audio files.
        output_dir (str): Path to the output directory where the split datasets will be saved.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        None
    """
    random.seed(seed)
    genres = os.listdir(original_dir)

    for genre in genres:
        genre_path = os.path.join(original_dir, genre)
        files = os.listdir(genre_path)

        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        for split, split_files in splits.items():
            split_dir = os.path.join(output_dir, split, genre)
            os.makedirs(split_dir, exist_ok=True)

            for file in split_files:
                src = os.path.join(genre_path, file)
                dst = os.path.join(split_dir, file)
                shutil.copy(src, dst)

if __name__ == "__main__":
    original_path = "Dataset/genres_original"
    output_path = "Dataset/genres_split"

    split_audio_dataset(original_path, output_path)
    print("Audio dataset split into train, val, and test sets.")