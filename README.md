MLOPS Final Project: HW 5

Version 1.0

Jacob Probst, University of St. Thomas

<br>
---Description---

This repository demonstrates Apache Airflow Orchestration, S3 storage, SQS message queuing, and Kubernetes scaling.

<br>
---System Requirements---

AWS Linux Environment

AWS Cloud9

Kubernetes/EKS cluster

Docker

Apache Airflow

<br>
---IAM---

The ServiceAccount needs IAM policy for:

s3:PutObject

sqs:ReceiveMessage

***NOTE: This project was completed in an AWS Lab environment. Confirm permissions for AWS CLI are correct.***

<br>
---Configuration---

Before running, update the following in Airflow Variables or .env:

- BUCKET_NAME: final-project-bucket-prob7276
- 
- SQS_QUEUE_URL: <insert_generated_url_here>

- AWS_DEFAULT_REGION: us-east-1

Docker Code:

cd consumer

docker build -t inference-worker .

docker tag inference-worker:latest [AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest

docker push [AWS_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/inference-worker:latest

Kubernetes Code:

kubectl apply -f deployment.yaml

<br>
---Instructions for System Execution---

Run the Training DAG in the Airflow UI at "<IP-address>:8080"

Run the Queue Population DAG in the Airflow UI

Code to Monitor Kubernetes Workers:

kubectl logs -l app=inference-worker -f --prefix

<br>
---Allow for Scaling Via Kubernetes---

Increase Number of Workers Code:

kubectl scale deployment inference-worker-deployment --replicas=5

<br>
---Cleanup---

Cleanup Code:

kubectl delete -f deployment.yaml

aws s3 rb s3://final-project-bucket-prob7276 --force

Delete the SQS queue and EKS cluster via the AWS Console.
