from pydantic import BaseModel

class PredictionInput(BaseModel):
	vendor_id: int
	passenger_count: int
	pickup_longitude: float
	pickup_latitude: float
	dropoff_longitude: float
	dropoff_latitude: float
	store_and_fwd_flag: int
	distance_haversine: float
	distance_dummy_manhattan: float
	direction: float
