from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime
from typing import Dict, Optional, Any
import math

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
collection = db["predictions"]

# ==========================================
# CREACIÓN DE ÍNDICES EN MONGO:
# ==========================================
# Al crear índices, las búsquedas por paciente o médico van  pasar de 
# complejidad O(N) a O(log N), optimizando la base de datos drásticamente.
collection.create_index("paciente_id")
collection.create_index("doctor_id")
collection.create_index([("created_at", -1)])


def save_prediction_metadata(
    doctor_id: str, 
    paciente_id: str, 
    task_type: str, 
    input_images: Dict[str, str], 
    prediction_url: Optional[str],
    status: str = "completed"
):
    """Registra el evento de predicción en MongoDB Atlas."""
    record = {
        "doctor_id": doctor_id,
        "paciente_id": paciente_id,
        "task_type": task_type,
        "created_at": datetime.utcnow(),
        "prediction_image": prediction_url,
        "status": status
    }
    
    # Inyectamos dinámicamente las URLs de entrada
    for modality, url in input_images.items():
        record[f"original_image_{modality}"] = url
        
    result = collection.insert_one(record)
    return str(result.inserted_id)


def get_paginated_history(filter_query: Dict[str, Any], page: int = 1, limit: int = 10) -> Dict[str, Any]:
    """
    Recupera el historial paginado basado en un filtro dinámico.
    Devuelve los documentos procesados y los metadatos de paginación.
    """
    # 1. Calculamos el desplazamiento (Offset)
    skip = (page - 1) * limit
    
    # 2. Obtenemos el conteo total de documentos que coinciden con el filtro
    total_records = collection.count_documents(filter_query)
    total_pages = math.ceil(total_records / limit) if limit > 0 else 1
    
    # 3. Ejecutamos la consulta con ordenamiento y paginación
    # Ordenamos por 'created_at' de forma descendente (-1) para ver lo más nuevo primero
    cursor = collection.find(filter_query).sort("created_at", -1).skip(skip).limit(limit)
    
    records = []
    for doc in cursor:
        # Transformamos el _id de ObjectId de Mongo a string estándar
        doc_id = str(doc.pop("_id"))
        
        # Reconstruimos el diccionario de imágenes originales que guardamos de forma aplanada
        original_images = {}
        keys_to_remove = []
        
        for key, value in doc.items():
            if key.startswith("original_image_"):
                modality = key.replace("original_image_", "")
                original_images[modality] = value
                keys_to_remove.append(key)
        
        # Limpiamos las llaves aplanadas del documento original
        for key in keys_to_remove:
            doc.pop(key)
            
        # Ensamblamos el registro final
        doc["id"] = doc_id
        doc["original_images"] = original_images
        records.append(doc)
        
    # 4. Ensamblamos la respuesta completa
    return {
        "data": records,
        "meta": {
            "total_records": total_records,
            "current_page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }