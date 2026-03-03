from pydantic import BaseModel, HttpUrl
from typing import List
from enum import Enum

# Error response schema send to the user in case of an error
class APIErrorSchema(BaseModel):
    type: str
    status: int
    detail: str

# Prediction response schema for segmentation tasks (ACV, Metastasis)
class PredictionResponse(BaseModel):
    status: str
    db_id: str
    original_image: HttpUrl
    prediction_image: HttpUrl
    task: str
    modalities_used: List[str]

# Classification response schema for Alzheimer
class AlzheimerPredictionResponse(BaseModel):
    status: str
    db_id: str
    original_image: HttpUrl
    task: str
    prediction: int  # 0 or 1
    probability: float  # Probability of positive class
    threshold: float  # Threshold used for classification
    modalities_used: List[str]


class TaskType(str, Enum):
    metastasis = "metastasis"
    acv = "acv"
    #alzheimer = "alzheimer"
