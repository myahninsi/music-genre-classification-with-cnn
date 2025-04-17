# music-genre-classification-with-cnn
This repository is for the final project of AML 3104 - Neural Networks and Deep Learning, that uses Convolutional Neural Networks (CNN) for classifying music genres from audio files. The project includes feature extraction from audio, data augementation, and improving classification accuracy. The project leverages Keras for model training and Streamlit for web application deployment. The model performance is further optimized through hyperparameter tuning.

## Repository Structure

```
Dataset/                # Directory containing the dataset
models/                 # Directory for saving models
.gitignore              # Gitignore file to exclude unnecessary files from Git tracking
LICENSE                 # License file for the repository
README.md               # This file
app.py                  # Streamlit app for the web interface
cnn_tuning.py           # Script for hyperparameter tuning of the CNN model
generate_spectogram_images.py # Script to generate spectrogram images from audio files
image_data_loader.py    # Data loading and processing script
main_v7.ipynb           # Jupyter notebook for training, evaluation, and saving models
requirements.txt        # Dependencies for the project
split_audio_dataset.py  # Script to split audio dataset into training and validation sets
```

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/myahninsi/music-genre-classification-with-cnn
   ```

2. Navigate into the project directory:

   ```
   cd music-genre-classification-with-cnn
   ```

3. Install the required Python dependencies:

   ```
   pip install -r requirements.txt
   ```

## Dataset

This project uses a dataset of music audio files that are processed into spectrogram images. The dataset is expected to be placed in the `Dataset/` directory. If the dataset is not included, you can generate spectrograms from the audio files using the `generate_spectogram_images.py` script.

## Model Training

1. The main model training and evaluation process is done in the `main_v7.ipynb` Jupyter notebook. This notebook contains the full pipeline for training, validating, and saving the CNN model.

2. The baseline CNN model can be found in `cnn_baseline.keras`. To improve the model's performance, hyperparameter tuning is performed using the `cnn_tuning.py` script.

## Web Application

The Streamlit web app is located in `app.py`. To run the application:

1. Ensure all dependencies are installed by running `pip install -r requirements.txt`.
2. Launch the Streamlit app by running:

   ```
   streamlit run app.py
   ```

## Scripts

- `generate_spectogram_images.py`: Converts audio files into spectrogram images.
- `image_data_loader.py`: Handles the loading and preprocessing of image data for model training.
- `cnn_tuning.py`: Handles the hyperparameter tuning process for the CNN model.
- `split_audio_dataset.py`: Splits the audio dataset into training and validation sets.

## Hyperparameter Tuning

Hyperparameter tuning is performed using the `cnn_tuning.py` script. This script applies Random Search for tuning parameters such as dropout rates, the number of units in dense layers, and optimizers. The tuning process helps to optimize the model's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

