from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
from enum import Enum

class APIErrorSchema(BaseModel):
    type: str
    status: int
    detail: str

class PredictionResponse(BaseModel):
    status: str
    db_id: str
    paciente_id: str
    doctor_id: str
    # Diccionario para mapear modalidad -> URL en MinIO
    original_images: Dict[str, HttpUrl]
    prediction_image: HttpUrl
    task: str
    modalities_used: List[str]

class AlzheimerPredictionResponse(BaseModel):
    status: str
    db_id: str
    paciente_id: str
    doctor_id: str
    original_image: HttpUrl
    task: str
    prediction: int 
    probability: float 
    threshold: float 
    modalities_used: List[str]

class TaskType(str, Enum):
    metastasis = "metastasis"
    acv = "acv"
    alzheimer = "alzheimer"