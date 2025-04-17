from PIL import Image
import numpy as np
import os

def load_flat_dataset(path, image_size=(128, 128), as_gray=True):
    X, y = [], []
    for genre in os.listdir(path):
        genre_path = os.path.join(path, genre)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(genre_path, file)
                try:
                    img = Image.open(img_path)
                    if as_gray:
                        img = img.convert('L')
                    img = img.resize(image_size)
                    img_array = np.asarray(img).flatten()
                    X.append(img_array)
                    y.append(genre)
                except:
                    print(f"Error loading: {img_path}")
    return np.array(X), np.array(y)
