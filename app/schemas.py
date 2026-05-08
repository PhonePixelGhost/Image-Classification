from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    score: float
    label_id: int
    inference_time_ms: float

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
