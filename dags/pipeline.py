from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

def run_script(script_name: str, config_file: str):
    cmd = ["python", f"scripts/{script_name}", "--config", f"config/{config_file}"]
    subprocess.run(cmd, check=True)

with DAG(
    dag_id="cnn_image_pipeline",
    default_args=default_args,
    description="End-to-end image classification pipeline with CNN",
    schedule_interval=None,
    catchup=False,
    tags=["cnn", "tuning", "baseline"]
) as dag:

    train_baseline = PythonOperator(
        task_id="train_baseline",
        python_callable=run_script,
        op_args=["train_baseline.py", "train_baseline.yaml"]
    )

    evaluate_baseline = PythonOperator(
        task_id="evaluate_baseline",
        python_callable=run_script,
        op_args=["evaluate_cnn.py", "evaluate_cnn_baseline.yaml"]
    )

    tune_cnn = PythonOperator(
        task_id="tune_cnn",
        python_callable=run_script,
        op_args=["tune_cnn.py", "tune_cnn.yaml"]
    )

    evaluate_tuned = PythonOperator(
        task_id="evaluate_tuned",
        python_callable=run_script,
        op_args=["evaluate_cnn.py", "evaluate_cnn_tuned.yaml"]
    )

    train_baseline >> evaluate_baseline >> tune_cnn >> evaluate_tuned
