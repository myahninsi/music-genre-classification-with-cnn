import argparse
import yaml
import tensorflow as tf

from scripts.image_data_loader import load_data_generators
from scripts.utils import ensure_dirs, load_training_outputs, save_evaluation_outputs

def run(
    dataset_path: str,
    model_path_artifact: str,
    evaluation_output_path: str,
    img_height: int,
    img_width: int,
    batch_size: int
) -> tuple[float, float]:
    """
        Loads a trained CNN model and evaluates it on the test data. 

        This function will 
            - Load a saved model path from the artifact file. 
            - Initalize a test data generator from the dataset path.
            - Evaluate the test accuracy and test loss. 
            - Save the evaluation metrics (accuracy and loss) to a JSON file. 

        Parameters: 
            dataset_path (str): Path to the dataset directory. Assume it contains `test` subfolder.
            model_path_artifact (str): Path to the artifact file containing the model path.
            evaluation_output_path (str): Path to save the evaluation metrics.
            img_height (int): Height of the input images.
            img_width (int): Width of the input images.
            batch_size (int): Batch size for evaluation.

        Returns: 
            tuple[float, float]: A tuple containing (test accuracy, test loss). 
    """

    # Make sure all directories exist
    ensure_dirs()

    # Load trained model path
    model_path, _ = load_training_outputs(
        model_path_artifact=model_path_artifact,
        metrics_artifact="" # we don't need these metrics for evaluation
    )

    model = tf.keras.models.load_model(model_path)
    print("Model loaded from:", model_path)

    # Load test data
    _, _, test_generator, _ = load_data_generators(
        dataset_path=dataset_path,
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
        augment=False
    )

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save evaluation metrics
    save_evaluation_outputs(
        output_path=evaluation_output_path,
        loss=test_loss,
        accuracy=test_acc
    )

    # Return the test accuracy and loss
    return test_acc, test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run(**config)