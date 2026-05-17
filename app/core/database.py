from pymongo import MongoClient
from app.core.config import settings
from datetime import datetime
from typing import Dict, Optional, Any
import math
import os

client = MongoClient(settings.MONGO_URI)
db = client[settings.DB_NAME]
collection = db["predictions"]

# --- PROTECCIÓN DEL ENTORNO DE TESTING ---
# Parseo booleano explícito. Evita que el string "False" detenga los tests.
# Durante la ejecución de la suite de pruebas, este archivo es importado ANTES
# de que las fixtures de simulación (mongomock) estén completamente montadas.
# Al omitir la creación de índices, evitamos que el test_runner colapse por timeout.
testing_mode = os.getenv("TESTING_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

if not testing_mode:
    # Optimización O(log N): Acelera las búsquedas en historiales
    collection.create_index("paciente_id")
    collection.create_index("doctor_id")
    collection.create_index([("created_at", -1)])


def save_prediction_metadata(
    doctor_id: str,
    paciente_id: str,
    task_type: str,
    input_images: Dict[str, str],
    prediction_url: Optional[str],
    visualization_url: Optional[str] = None,
    status: str = "completed",
):
    """
    Registra el evento de predicción en MongoDB Atlas.
    Se aplana dinámicamente el diccionario de 'input_images' para facilitar consultas indexadas.
    """
    record = {
        "doctor_id": doctor_id,
        "paciente_id": paciente_id,
        "task_type": task_type,
        "created_at": datetime.utcnow(),
        "prediction_image": prediction_url,
        "visualization_image": visualization_url,
        "status": status,
    }

    # Inyectamos dinámicamente las URLs de entrada ("original_image_t1", "original_image_flair", etc.)
    for modality, url in input_images.items():
        record[f"original_image_{modality}"] = url

    result = collection.insert_one(record)
    return str(result.inserted_id)


def get_paginated_history(filter_query: Dict[str, Any], page: int = 1, limit: int = 10) -> Dict[str, Any]:
    """
    Recupera el historial paginado.
    Aplica lógica inversa para reconstruir el diccionario de imágenes originales.
    """
    # 1. Calculamos el desplazamiento (Offset) para la paginación
    skip = (page - 1) * limit
    total_records = collection.count_documents(filter_query)
    total_pages = math.ceil(total_records / limit) if limit > 0 else 1

    # 2. Búsqueda con ordenamiento por fecha (más nuevos primero)
    cursor = collection.find(filter_query).sort("created_at", -1).skip(skip).limit(limit)

    records = []
    for doc in cursor:
        # Casteo de ObjectId de Mongo a String para compatibilidad con JSON/Pydantic
        doc_id = str(doc.pop("_id"))
        original_images = {}
        keys_to_remove = []

        # 3. Des-Aplanamiento: Agrupamos de nuevo las modalidades en un diccionario
        for key, value in doc.items():
            if key.startswith("original_image_"):
                modality = key.replace("original_image_", "")
                original_images[modality] = value
                keys_to_remove.append(key)

        for key in keys_to_remove:
            doc.pop(key)

        doc["id"] = doc_id
        doc["original_images"] = original_images
        records.append(doc)

    return {
        "data": records,
        "meta": {
            "total_records": total_records,
            "current_page": page,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
    }
