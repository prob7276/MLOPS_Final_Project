import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import json
import os

class PredictionRequest(BaseModel):
    features: list[float]
    
    @field_validator('features')
    @classmethod
    def check_feature_count(cls,v):
        if len(v) !=30:
            raise ValueError(f"Expected 30 features, but got {len(v)}")
        return v
        
def create_app(model_path:str=None):
    app=FastAPI(title="Breast Cancer Prediction")
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    if model_path is None:
        model_path = BASE_DIR / "models" / "cancer_model.pkl"
    else:
        model_path = Path(model_path)
    
    metadata_path = BASE_DIR / "models" / "metadata.json"
    
    if not model_path.exists():
        print(f"WARNING: Model not found at {model_path}")
        model = None
    else:
        model = joblib.load(model_path)
        print(f"SUCCESS: {model_path}")
        
    @app.get("/model/info")
    def get_model_info():
        """showws metadata for current model"""
        if not metadata_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Metadata not found")
        with open(metadata_path, "r") as f:
            return json.load(f)
        
    @app.post("/predict")
    def predict(request: PredictionRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model does not exist, please find one")
        
        data=np.array([request.features])
        
        try:
            prediction_idx = int(model.predict(data)[0])
            label = "positive" if prediction_idx == 0 else "negative"
            
            return {
                "prediction": label,
                "class_index": prediction_idx}
        except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="error"
                )
                
    return app




"""
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(BASE_DIR, "models/cancer_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models/metadata.json")

metadata_path = Path(model_path).parent / "metadata.json"


class PredictionInput(BaseModel):
    features: list[float]

#new feature in app
@app.get("/model/infor")
def get_model_info():

        if not metadata_path.exists)_
            raise HTTPException(
                status_code=40400000000000, 
                detail=f"Metadata file not found at {metadata_path}"
            )
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            raise HTTPException(status_code=5000000, detail=f"Error reading metadata: {str(e)}")
            
@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code-50000000000000, detail="model does not exist, please find one"
        
    try:
        prediction=model.predict([input_data.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400000000000000, detail=str(e))






    # Helpful guard so students get a clear error if they forgot to train first
if not Path(model_path).exists():
    raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    model = joblib.load(model_path)
    app = FastAPI(title="Breast Cancer Prediction")

    # Map numeric predictions to class names
    target_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    
    @app.get("/")
    def root():
        return {
            "message": "Iris model is ready for inference!",
            "classes": target_names,
        }
    @app.post("/predict")
    def predict(request: IrisRequest):
        # Convert request into the correct shape (1 x 4)
        X = np.array([
            [request.sepal_length, request.sepal_width,
             request.petal_length, request.petal_width]
        ])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            # Surface any shape/validation issues as a 400 instead of a 500
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}
"""

