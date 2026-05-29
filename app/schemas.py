from pydantic import BaseModel, HttpUrl
from typing import List, Dict
from enum import Enum
from datetime import datetime


class APIErrorSchema(BaseModel):
    type: str
    status: int
    detail: str


class PredictionResponse(BaseModel):
    status: str
    db_id: str
    paciente_id: str
    doctor_id: str
    original_images: Dict[str, HttpUrl]
    prediction_image: HttpUrl
    visualization_image: HttpUrl | None = None  # URL del JPG
    task: str
    modalities_used: List[str]


class TaskType(str, Enum):
    metastasis = "metastasis"
    acv = "acv"
    alzheimer = "alzheimer"


class HistoryRecord(BaseModel):
    id: str
    doctor_id: str
    paciente_id: str
    task_type: str
    created_at: datetime
    original_images: Dict[str, str]
    prediction_image: str | None = None
    visualization_image: str | None = None  # Para JPG en el historial
    status: str


class PaginationMeta(BaseModel):
    total_records: int
    current_page: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedHistoryResponse(BaseModel):
    data: List[HistoryRecord]
    meta: PaginationMeta
