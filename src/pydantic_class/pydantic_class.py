from pydantic import BaseModel

class PredictionInput(BaseModel):
	vendor_id: int
	pickup_datetime: object
	passenger_count: int
	pickup_longitude: float
	pickup_latitude: float
	dropoff_longitude: float
	dropoff_latitude: float
	store_and_fwd_flag: object
