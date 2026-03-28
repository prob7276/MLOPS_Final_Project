import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

def generate_data(output_path: str = "data/iris.csv") -> str:
    """get the breast cancer dataset and save it"""
    
    data=load_breast_cancer()
    
    df=pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"[ml_pipeline.data] Dataset saved to {output_path}")
    return output_path
    
def load_data(path: str) -> pd.DataFrame:
    """load data"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found at {path}")
    return pd.read_csv(path)