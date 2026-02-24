from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
from typing import Optional, List
from app.core.inference import load_models, run_inference
from app.core.storage import upload_file
from app.core.database import save_prediction_metadata
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import *

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("Apagando servicio...")

app = FastAPI(title="Medical AI API", lifespan=lifespan)
register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/{task_type}")
async def predict_endpoint(
    task_type: str,
    doctor_id: str = Form(...),
    # Definimos los 4 tipos de secuencias posibles
    file_t1: UploadFile = File(...),         # T1 es base para ambos
    file_t1ce: Optional[UploadFile] = File(
        None),  # T1 con Contraste (Solo Mets)
    file_t2: Optional[UploadFile] = File(None),   # T2 (Solo Mets)
    file_flair: Optional[UploadFile] = File(None)  # FLAIR (Solo Mets)
):
    """
    Endpoint inteligente:
    - Para ACV: Solo requiere file_t1.
    - Para Metástasis: Requiere T1, T1CE, T2 y FLAIR.
    """
    if task_type not in ["metastasis", "acv"]:
        raise ValidationError(detail="Task must be 'metastasis' or 'acv'")

    # --- VALIDACIÓN DE ARCHIVOS REQUERIDOS ---
    if task_type == "metastasis":
        if not (file_t1ce and file_t2 and file_flair):
            raise UnprocessableEntityError(
                detail="Para Metástasis se requieren 4 secuencias: T1, T1CE, T2 y FLAIR."
            )
        files_map = {
            "t1": file_t1, "t1ce": file_t1ce, "t2": file_t2, "flair": file_flair
        }
    else:
        # Para ACV solo usamos T1
        files_map = {"t1": file_t1}

    # Crear carpeta temporal única
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths = {}

    try:
        # 1. Guardamos todos los archivos recibidos en disco
        for key, file_obj in files_map.items():
            file_path = f"{temp_dir}/{key}.nii.gz"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            saved_paths[key] = file_path

        # 2. Definir ruta de salida
        output_filename = f"{temp_dir}/prediction.nii.gz"

        # 3. Corremos Inferencia (Pasamos el diccionario de rutas)
        # La función run_inference ahora sabe cómo juntarlos
        run_inference(saved_paths, output_filename, task_type)

        # 4. Subir a MinIO
        # Subimos solo el T1 como "original" para visualización rápida
        # (O podemos subir los 4 si queremos, acá subimos el principal)
        s3_path_in = f"{doctor_id}/{task_type}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{task_type}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

        # 5. Guardamos Metadata
        db_id = save_prediction_metadata(doctor_id, task_type, url_in, url_out)

        return {
            "status": "success",
            "db_id": db_id,
            "original_image": url_in,
            "prediction_image": url_out,
            "task": task_type,
            "modalities_used": list(saved_paths.keys())
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise InternalError(detail=str(e))

    finally:
        # Limpiar disco
        shutil.rmtree(temp_dir, ignore_errors=True)
