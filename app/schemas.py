from pydantic import BaseModel, HttpUrl
from typing import List
from enum import Enum

# Error response schema send to the user in case of an error
class APIErrorSchema(BaseModel):
    type: str
    status: int
    detail: str

# Prediction response schema
class PredictionResponse(BaseModel):
    status: str
    db_id: str
    original_image: HttpUrl
    prediction_image: HttpUrl
    task: str
    modalities_used: List[str]


class TaskType(str, Enum):
    metastasis = "metastasis"
    acv = "acv"
    alzheimer = "alzheimer"
