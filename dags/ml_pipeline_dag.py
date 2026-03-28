from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(BASE_DIR, "src"))

from ml_pipeline.data import generate_data, load_data
from ml_pipeline.model import train_model, evaluate_model, promote_model

DATA_PATH = os.path.join(BASE_DIR, "data/breast_cancer.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/cancer_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models/metrics.json")
METADATA_PATH = os.path.join(BASE_DIR, "models/metadata.json")

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="three step pipeline: train> evaluate> promote",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    #get data
    generate_task = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
        op_kwargs={"output_path": DATA_PATH},
    )
    
    #train
    def train_wrapper(data_path: str, model_path: str):
        df=load_data(data_path)
        return train_model(df, model_path)
        
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_wrapper,
        op_kwargs={
            "data_path": DATA_PATH, 
            "model_path": MODEL_PATH},
    )

    #evaluate
    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={
            "model_path": MODEL_PATH,
            "test_data_path": DATA_PATH,
            "metrics_path": METRICS_PATH,
            "metadata_path": METADATA_PATH,
            "model_version": "{{ logical_date.strftime('%Y%m%d_%H%M%S') }}"
        },
    )
    
    #promote
    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
        op_kwargs={
            "model_path": MODEL_PATH,
            "metrics_path": METRICS_PATH,
            "metadata_path": METADATA_PATH,
            "bucket_name": "lab-3-bucket-mlops",
            "model_version": "{{ logical_date.strftime('%Y%m%d_%H%M%S') }}",
            "threshold": 0.94
        },
    )
    
    generate_task >> train_task >> evaluate_task >> promote_task