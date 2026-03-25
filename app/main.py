from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import shutil
import os
import uuid
from app.core.inference import load_models, run_inference_alzheimer, run_inference_acv, run_inference_metastasis
from app.core.database import save_prediction_metadata
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import InternalError
from app.schemas import PredictionResponse, AlzheimerPredictionResponse, APIErrorSchema, TaskType
from app.core.storage import upload_file, initialize_storage 

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando carga de modelos AI...")
    load_models()
    print("Inicializando conexiones de almacenamiento...")
    initialize_storage()  # Se ejecuta cuando se levante el contenedor, 
    # para asegurar que el bucket exista antes de cualquier operacion de upload
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
    "/predict/metastasis",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        404: {"model": APIErrorSchema},
        422: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
    },
)
async def predict_metastasis(
    doctor_id: str = Form(...),
    # Definimos los 4 tipos de secuencias posibles
    file_t1: UploadFile = File(...),         # T1 es base para todos
    file_t1ce: UploadFile = File(...),  # T1 con Contraste
    file_t2: UploadFile = File(...),   # T2 
    file_flair: UploadFile = File(...)  # FLAIR
):
    """
    - Para Metástasis: Requiere T1, T1CE, T2 y FLAIR.
    """

    # --- VALIDACIÓN DE ARCHIVOS REQUERIDOS ---
    files_map = {
            "t1": file_t1, "t1ce": file_t1ce, "t2": file_t2, "flair": file_flair
        }

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
        run_inference_metastasis(saved_paths, output_filename, TaskType.metastasis)

        # 4. Subir a MinIO
        # Subimos solo el T1 como "original" para visualización rápida
        s3_path_in = f"{doctor_id}/{TaskType.metastasis.value}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{TaskType.metastasis.value}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

        # 5. Guardamos Metadata
        db_id = save_prediction_metadata(doctor_id, TaskType.metastasis.value, url_in, url_out)

        return PredictionResponse(
            status="success",
            db_id=db_id,
            original_image=url_in,
            prediction_image=url_out,
            task=TaskType.metastasis,
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
    "/predict/acv",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        404: {"model": APIErrorSchema},
        422: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
    },
)
async def predict_acv(
    doctor_id: str = Form(...),
    file_t1: UploadFile = File(...)
):
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
        
        # ACV: devuelve segmentación
        # 2. Definir ruta de salida
        output_filename = f"{temp_dir}/prediction.nii.gz"

        # 3. Corremos Inferencia (Pasamos el diccionario de rutas)
        run_inference_acv(saved_paths, output_filename, TaskType.acv)

        # 4. Subir a MinIO
        # Subimos solo el T1 como "original" para visualización rápida
        s3_path_in = f"{doctor_id}/{TaskType.acv.value}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{TaskType.acv.value}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

        # 5. Guardamos Metadata
        db_id = save_prediction_metadata(doctor_id, TaskType.acv.value, url_in, url_out)

        return PredictionResponse(
            status="success",
            db_id=db_id,
            original_image=url_in,
            prediction_image=url_out,
            task=TaskType.acv,
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
        db_id = save_prediction_metadata(doctor_id, TaskType.alzheimer.value, url_in, None)
        
        return AlzheimerPredictionResponse(
            status="success",
            db_id=db_id,
            original_image=url_in,
            task=TaskType.alzheimer.value,
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