from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime
from typing import Dict, Optional

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
collection = db["predictions"]

def save_prediction_metadata(
    doctor_id: str, 
    paciente_id: str, 
    task_type: str, 
    input_images: Dict[str, str], 
    prediction_url: Optional[str],
    status: str = "completed"
):
    """
    Registra el evento de predicción en MongoDB Atlas.
    Mapea dinámicamente las imágenes originales al formato original_image_{modalidad}.
    """
    record = {
        "doctor_id": doctor_id,
        "paciente_id": paciente_id,
        "task_type": task_type,
        "created_at": datetime.utcnow(),
        "prediction_image": prediction_url,
        "status": status
    }
    
    # Inyectamos dinámicamente las URLs de las imágenes de entrada
    for modality, url in input_images.items():
        record[f"original_image_{modality}"] = url
        
    result = collection.insert_one(record)
    return str(result.inserted_id)