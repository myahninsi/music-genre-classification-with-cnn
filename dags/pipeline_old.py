import subprocess

"""
Runs the full image classification pipeline step by step:
1. Train baseline CNN
2. Evaluate baseline CNN
3. Tune CNN with Keras Tuner and get the best model
4. Evaluate the final tuned CNN

Each step uses its own configuration YAML file for parameters.
"""

def run_step(description: str, command: list):
    """
    Executes a subprocess command for a pipeline step.

    Parameters:
        description (str): Human-readable label for the step.
        command (list): CLI command as a list of arguments.
    """
    print(f"\nRunning step: {description}")
    try:
        subprocess.run(command, check=True)
        print(f"Completed: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while executing: {description}\n{e}\n")
        exit(1)

def main():
    """
    Defines and runs all pipeline steps sequentially using predefined scripts and config files.
    """
    steps = [
        {
            "desc": "Training Baseline CNN",
            "cmd": ["python", "scripts/train_baseline.py", "--config", "config/train_baseline.yaml"]
        },
        {
            "desc": "Evaluating Baseline CNN",
            "cmd": ["python", "scripts/evaluate_cnn.py", "--config", "config/evaluate_cnn_baseline.yaml"]
        },
        {
            "desc": "Tuning CNN with Keras Tuner",
            "cmd": ["python", "scripts/tune_cnn.py", "--config", "config/tune_cnn.yaml"]
        },
        {
            "desc": "Evaluating Tuned CNN",
            "cmd": ["python", "scripts/evaluate_cnn.py", "--config", "config/evaluate_cnn_tuned.yaml"]
        }
    ]

    for step in steps:
        run_step(step["desc"], step["cmd"])

if __name__ == "__main__":
    main()