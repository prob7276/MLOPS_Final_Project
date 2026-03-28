import os
import joblib
import pandas as pd
import shutil
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.exceptions import AirflowException

def train_model(df: pd.DataFrame, model_path: str = "models/iris_model.pkl") -> float:
    """Train a logistic regression classifier and save it."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    return acc


def evaluate_model(model_path: str, test_data_path: str, metrics_path: str, metadata_path: str, model_version: str): 
    """Load a model, evaluate it, and save metrics to a JSON file."""
    print(f"[ml_pipeline.model] Evaluating version: {model_version}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Load the model and data
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    acc = accuracy_score(y, model.predict(X))
    
    metadata = {
        "model_version": model_version,
        "dataset": "breast_cancer",
        "model_type": "logistic_regression",
        "accuracy": float(acc)
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[ml_pipeline.model] Metadata saved to {metadata_path}")
    return acc
    
    
def promote_model(
    model_path: str, 
    metrics_path: str,
    metadata_path: str, 
    bucket_name: str, 
    model_version: str, 
    threshold: float = 0.94
):
    """check model quality and promote"""
    
    #load data
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    accuracy=metadata['accuracy']
    
    #quality check
    if accuracy < threshold:
        raise AirflowException(
            f"Model accuracy {accuracy:.4f} is below threshold {threshold}")
            
    print(f"[ml_pipeline.model] Promoting version {model_version}")
    s3 = S3Hook(aws_conn_id='aws_default')
    
    artifacts = {
        "model.pkl": model_path,
        "metrics.json": metrics_path,
        "metadata.json": metadata_path
    }
    
    for filename, local_path in artifacts.items():
        s3_key=f"models/{model_version}/{filename}"
        print(f"Uploading {filename} to s3://{bucket_name}/{s3_key}")
        
        s3.load_file(
            filename=local_path,
            key=s3_key,
            bucket_name=bucket_name,
            replace=True)
    
    return True