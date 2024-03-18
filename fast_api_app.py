# main.py
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pathlib
import pandas as pd
import numpy as np
import pickle
from src.pydantic_class import pydantic_class_defination
from src.features.build_features import feature_build, feature_drop

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent    
data_path = home_dir.as_posix() + '/data/raw'    
test_data_path = data_path + '/test.csv'
app = FastAPI()

pydantic_class_defination.pydantic_class(test_data_path)

from src.pydantic_class.pydantic_class import PredictionInput

# Load the pre-trained RandomForest model
model_path = "models/model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.get('/')
def home():
    return 'Working fine'
    
@app.post('/predict')
def predict(input_data: PredictionInput):
    
    features = {'vendor_id': input_data.vendor_id,
                'pickup_datetime': input_data.pickup_datetime,
                'passenger_count': input_data.passenger_count,
                'pickup_longitude': input_data.pickup_longitude,
                'pickup_latitude': input_data.pickup_latitude,
                'dropoff_longitude': input_data.dropoff_longitude,
                'dropoff_latitude': input_data.dropoff_latitude,
                'store_and_fwd_flag': input_data.store_and_fwd_flag}
    
    features = pd.DataFrame(features, index = [0])
    features = feature_build(features)
    feature_drop_list = ['pickup_datetime']
    features = feature_drop(features, feature_drop_list)
    
    predict = model.predict(features)[0].item()

    return {'prediction': predict}

if __name__ == '__main__':
    
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8080)
    
    