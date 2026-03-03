from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
from typing import Optional
from app.core.inference import load_models, run_inference, run_inference_alzheimer
from app.core.storage import upload_file
from app.core.database import save_prediction_metadata
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import InternalError, UnprocessableEntityError
from app.schemas import PredictionResponse, AlzheimerPredictionResponse, APIErrorSchema, TaskType

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


@app.post(
    "/predict/acv",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        404: {"model": APIErrorSchema},
        422: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
    },
)
async def predict_endpoint(
    task_type: TaskType,
    doctor_id: str = Form(...),
    # Definimos los 4 tipos de secuencias posibles
    file_t1: UploadFile = File(...),         # T1 es base para todos
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
    #if task_type not in ["metastasis", "acv", "alzheimer"]:
    #    raise ValidationError(detail="Task must be 'metastasis', 'acv' or 'alzheimer'")

    # --- VALIDACIÓN DE ARCHIVOS REQUERIDOS ---
    if task_type == TaskType.metastasis:
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
        
        # ACV y Metástasis: devuelven segmentación
        # 2. Definir ruta de salida
        output_filename = f"{temp_dir}/prediction.nii.gz"

        # 3. Corremos Inferencia (Pasamos el diccionario de rutas)
        run_inference(saved_paths, output_filename, task_type)

        # 4. Subir a MinIO
        # Subimos solo el T1 como "original" para visualización rápida
        s3_path_in = f"{doctor_id}/{task_type}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{task_type}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

        # 5. Guardamos Metadata
        db_id = save_prediction_metadata(doctor_id, task_type.value, url_in, url_out)

        return PredictionResponse(
            status="success",
            db_id=db_id,
            original_image=url_in,
            prediction_image=url_out,
            task=task_type,
            modalities_used=list(saved_paths.keys())
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise InternalError(detail=str(e))

    finally:
        # Limpiar disco
        shutil.rmtree(temp_dir, ignore_errors=True)



@app.post(
    "/predict/alzheimer",
    response_model=AlzheimerPredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        404: {"model": APIErrorSchema},
        422: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
    },
)
async def predict_alzheimer(
    doctor_id: str = Form(...),
    file_t1: UploadFile = File(...),
):
    """Endpoint específico para Alzheimer. Solo requiere T1.
    """
    # Crear carpeta temporal única
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Guardar archivo T1
        file_path = f"{temp_dir}/t1.nii.gz"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_t1.file, buffer)
        
        saved_paths = {"t1": file_path}

        # Correr inferencia
        result = run_inference_alzheimer(saved_paths)
        
        # Subir solo la imagen original
        s3_path_in = f"{doctor_id}/alzheimer/{job_id}/input_t1.nii.gz"
        url_in = upload_file(saved_paths["t1"], s3_path_in)
        
        # Guardar metadata
        db_id = save_prediction_metadata(doctor_id, "alzheimer", url_in, None)
        
        return AlzheimerPredictionResponse(
            status="success",
            db_id=db_id,
            original_image=url_in,
            task="alzheimer",
            prediction=result["prediction"],
            probability=result["probability"],
            threshold=result["threshold"],
            modalities_used=list(saved_paths.keys())
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise InternalError(detail=str(e))

    finally:
        # Limpiar disco
        shutil.rmtree(temp_dir, ignore_errors=True)