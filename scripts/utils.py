import os
import json


def ensure_dirs():
    """Ensure that all common output directories exist."""
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)


def save_training_outputs(model_path, history, model_path_artifact, metrics_artifact):
    """Save model path and training history to standard artifact files."""
    with open(model_path_artifact, "w") as f:
        f.write(str(model_path))

    with open(metrics_artifact, "w") as f:
        json.dump(history.history, f, indent=4)

    print("Model path saved to:", model_path_artifact)
    print("Training history saved to:", metrics_artifact)


def load_training_outputs(model_path_artifact, metrics_artifact):
    """Load model path and training history from artifacts."""
    with open(model_path_artifact, "r") as f:
        model_path = f.read().strip()

    with open(metrics_artifact, "r") as f:
        import json
        history = json.load(f)

    print("Loaded model path and training history.")
    return model_path, history


def save_evaluation_outputs(output_path, loss, accuracy):
    """Save test loss and accuracy as a JSON artifact."""
    data = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy)
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Evaluation metrics saved to {output_path}")
