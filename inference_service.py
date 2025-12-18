import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# === Create FastAPI app ===
app = FastAPI(
    title="Manifold Projection Service",
    description="Topological projection of high-dimensional network signatures into latent manifold spaces for structural analysis.",
    version="1.0.0"
)

# === Define Input Model ===
class FeatureVectors(BaseModel):
    DpiPolicy: Optional[List[str]] = []
    contentType: Optional[List[str]] = []
    IpProtocol: Optional[List[str]] = []
    appName: Optional[List[str]] = []
    bytesFromClient: float = 0.0
    bytesFromServer: float = 0.0
    sessions_count: int = 0
    transationDuration: float = 0.0

# === Define Output Model ===
class ProjectionOutput(BaseModel):
    latent_assignments: List[str] # Renamed from cluster_id to reflect research output

# === Load saved models ===
@app.on_event("startup")
def load_assets():
    global som_weights, centroid_feature_map, policy_to_bit, content_to_bit, proto_to_bit, scaler, medians, numeric_cols
    
    try:
        som_model = joblib.load("models2/som_model.joblib")
        som_weights = som_model._weights
    except (FileNotFoundError, AttributeError):
        try:
            som_weights = np.load("som_weights1.npy")
        except FileNotFoundError:
            raise RuntimeError("SOM artifacts missing.")

    centroid_feature_map = joblib.load('models2/centroid_feature_map.joblib')
    policy_to_bit = joblib.load('models2/policy_to_bit.joblib')
    content_to_bit = joblib.load('models2/content_to_bit.joblib')
    proto_to_bit = joblib.load('models2/proto_to_bit.joblib')
    scaler = joblib.load('models2/scaler.joblib')
    medians = joblib.load('models2/medians.joblib')
    numeric_cols = joblib.load('models2/numeric_cols.joblib')

def project_to_manifold(signal_features, som_weights, mapping, som_dims=(10, 10)):
    expected_features = som_weights.shape[2]
    # Internal dimension handling
    if len(signal_features) != expected_features:
        if len(signal_features) > expected_features:
            signal_features = signal_features[:expected_features]
        else:
            temp = np.zeros(expected_features)
            temp[:len(signal_features)] = signal_features
            signal_features = temp
    
    features_reshaped = signal_features.reshape(1, 1, -1)
    distances = np.linalg.norm(som_weights - features_reshaped, axis=2)
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
    node_id = bmu_index[0] * som_dims[1] + bmu_index[1]
    
    assignments = mapping.get(node_id, [])
    return assignments if assignments else ['LATENT_CLASS_A', 'LATENT_CLASS_B']

def preprocess_signal_data(signal: FeatureVectors) -> np.ndarray:
    data_dict = {}
    # Bitmap Encoding
    data_dict['DpiPolicy'] = sum(1 << policy_to_bit[p] for p in signal.DpiPolicy if p in policy_to_bit) if signal.DpiPolicy else 0
    data_dict['contentType'] = sum(1 << content_to_bit[c] for c in signal.contentType if c in content_to_bit) if signal.contentType else 0
    data_dict['IpProtocol'] = sum(1 << proto_to_bit[p] for p in signal.IpProtocol if p in proto_to_bit) if signal.IpProtocol else 0
    data_dict['app_count'] = len(signal.appName) if signal.appName else 0
    
    # Feature Imputation
    data_dict['bytesFromClient'] = max(signal.bytesFromClient, medians.get('bytesFromClient', 1))
    data_dict['bytesFromServer'] = max(signal.bytesFromServer, medians.get('bytesFromServer', 1))
    data_dict['sessions_count'] = signal.sessions_count
    data_dict['transationDuration'] = max(signal.transationDuration, medians.get('transationDuration', 1))
    
    df = pd.DataFrame([data_dict])
    for col in numeric_cols:
        if col not in df.columns: df[col] = 0
    df = df[numeric_cols]
    return scaler.transform(df)[0]

@app.post("/classify", response_model=ProjectionOutput)
def classify_signal(signal: FeatureVectors):
    try:
        is_empty = all([not signal.DpiPolicy, not signal.contentType, signal.bytesFromClient == 0])
        if is_empty:
            return {"latent_assignments": ['REGIME_UNDETERMINED']}
        
        processed_features = preprocess_signal_data(signal)
        results = project_to_manifold(processed_features, som_weights, centroid_feature_map)
        return {"latent_assignments": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.get("/")
def read_root(): # Fixed Syntax
    return {
        "message": "Manifold Projection Service Active", 
        "usage": "POST network signal signatures to /classify for topological mapping."
    }

if __name__ == "__main__":
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8005, reload=True)
