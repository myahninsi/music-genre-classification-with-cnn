import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import ffmpeg
from pathlib import Path

def process_spectrogram_split(audio_root, image_root, split, img_size=(128, 128)):
    """
    Converts audio files in a specified split of a dataset into Mel spectrogram images, and saves them to a specified directory.

    Parameters:
        audio_root (str): Path to the root directory of the audio dataset. 
        image_root (str): Path to the root directory where the spectrogram images will be saved.
        split (str): The split of the dataset to process (e.g., 'train', 'val', 'test').
        img_size (tuple): Size of the output images in pixels (width, height).

    Returns:
        None
    """
    audio_root = Path(audio_root)
    image_root = Path(image_root)
    split_dir = audio_root / split

    for genre_dir in split_dir.iterdir():
        if genre_dir.is_dir():
            for audio_file in genre_dir.glob("*.wav"):
                try:
                    # Load audio and convert to Mel spectrogram
                    y, sr = librosa.load(audio_file, duration=30)
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    # Prepare output path
                    rel_path = audio_file.relative_to(audio_root)
                    out_path = image_root / rel_path.with_suffix('.png')
                    os.makedirs(out_path.parent, exist_ok=True)

                    # Plot and save spectrogram
                    # plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
                    # librosa.display.specshow(S_dB, sr=sr, cmap='magma')
                    # plt.axis('off')
                    # plt.tight_layout(pad=0)
                    # plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

                    plt.figure(figsize=(4, 3), dpi=100)
                    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    print(f"Saved: {out_path}")
                except Exception as e:
                    print(f"Error with {audio_file}: {e}")


if __name__ == "__main__":
    process_spectrogram_split('Dataset/genres_split', 'Dataset/spectrograms_split', split='train')
    process_spectrogram_split('Dataset/genres_split', 'Dataset/spectrograms_split', split='val')
    process_spectrogram_split('Dataset/genres_split', 'Dataset/spectrograms_split', split='test')
    print("Spectrogram images generated for train, val, and test splits.")