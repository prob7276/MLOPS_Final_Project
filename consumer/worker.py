import boto3
import json
import joblib
import os
from datetime import datetime, timezone


SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/128311524242/cancer-inference-queue"
REGION="us-east-1"
S3_BUCKET_NAME="final-project-bucket-prob7276"
sqs=boto3.client('sqs',region_name=REGION)
s3=boto3.client('s3',region_name=REGION)

def load_model():
    print("downloading...")
    local_model_path="local_model.pkl"
    
    s3.download_file(S3_BUCKET_NAME, "models/20260503_232121/model.pkl", local_model_path)
    
    model=joblib.load(local_model_path)
    print("success")
    return model
    
def process_messages(model):
    print("SQS processing meassages")
    while True:
        response=sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=5
        )
        
        if 'Messages' not in response:
            print("Queue is empty...")
            break
        
        message=response['Messages'][0]
        receipt_handle=message['ReceiptHandle']
    
        payload=json.loads(message['Body'])
        record_id=payload['record_id']
        features =payload['features']
    
        print(f"\nProcessing {record_id}...")
    
    
        prediction=model.predict([features])[0]
    
        result={
            "record_id": record_id,
            "prediction": int(prediction),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    
        s3_key=f"predictions/{record_id}.json"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(result)
        )
        print(f"saved to s3://final-project-bucket-prob7276/{s3_key}")
    
        sqs.delete_message(
            QueueUrl=SQS_QUEUE_URL,
            ReceiptHandle=receipt_handle
        )
        print("message DELETED!!!")
    
if __name__=="__main__":
    loaded_model=load_model()
    process_messages(loaded_model)