from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import boto3
import json
import os

SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/128311524242/cancer-inference-queue"


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data/breast_cancer.csv")

default_args = {"owner": "airflow", "retries": 1}

def dispatch_to_sqs():
    """Reads test data and sends one message per record to SQS."""
    

    sqs = boto3.client('sqs', region_name='us-east-1')
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run the training DAG first!")
        
    df = pd.read_csv(DATA_PATH)
    
    test_df = df.head(50) 
    
    messages_sent = 0
    for index, row in test_df.iterrows():
        features = row.iloc[:-1].tolist() 
        
        payload = {
            "record_id": f"sample_{index:03d}",
            "features": features
        }
        
        response = sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(payload)
        )
        
        print(f"Sent {payload['record_id']} with MessageId: {response['MessageId']}")
        messages_sent += 1
        
    print(f"Successfully dispatched {messages_sent} messages to SQS.")

with DAG(
    dag_id="sqs_inference_dispatch",
    default_args=default_args,
    description="Reads dataset and dispatches inference jobs to SQS",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    dispatch_task = PythonOperator(
        task_id="dispatch_messages",
        python_callable=dispatch_to_sqs,
    )