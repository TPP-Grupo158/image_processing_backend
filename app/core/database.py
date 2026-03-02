from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime
from app.schemas import TaskType

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
collection = db["predictions"]


def save_prediction_metadata(doctor_id: str, task_type: TaskType, original_url: str, prediction_url: str):
    record = {
        "doctor_id": doctor_id,
        "task_type": task_type.value,  # 'metastasis' o 'acv'
        "created_at": datetime.utcnow(),
        "original_image": original_url,
        "prediction_image": prediction_url,
        "status": "completed"
    }
    result = collection.insert_one(record)
    return str(result.inserted_id)
