from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import shutil
import os
import uuid

# Importaciones de tu proyecto
from app.core.inference import load_models, run_inference_alzheimer, run_inference_acv, run_inference_metastasis
from app.core.database import save_prediction_metadata
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import InternalError
from app.schemas import PredictionResponse, AlzheimerPredictionResponse, APIErrorSchema, TaskType
from app.core.storage import upload_file, initialize_storage 

# ==========================================
# 1. VARIABLE GLOBAL: Para El SEMÁFORO
# ==========================================
# Esta variable controla que solo pase 1 petición a la vez.
is_processing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando carga de modelos AI...")
    load_models()
    print("Inicializando conexiones de almacenamiento...")
    initialize_storage()  
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

# ==========================================
# 2. ENDPOINT DE STATUS (Para el Frontend)
# ==========================================
@app.get("/status")
async def get_system_status():
    """
    El frontend debe consumir este endpoint cada 2 o 3 segundos.
    Retorna el estado actual del servidor para mostrar en la UI.
    """
    if is_processing:
        return {
            "status": "busy", 
            "color": "red", 
            "message": "El equipo está procesando otra imagen. Por favor, aguarde."
        }
    return {
        "status": "free", 
        "color": "green", 
        "message": "Sistema listo para recibir imágenes."
    }


# ==========================================
# 3. ENDPOINTS DE INFERENCIA
# ==========================================

@app.post(
    "/predict/metastasis",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
        503: {"description": "Servidor Ocupado"}
    },
)
async def predict_metastasis(
    doctor_id: str = Form(...),
    file_t1: UploadFile = File(...),         
    file_t1ce: UploadFile = File(...),  
    file_t2: UploadFile = File(...),   
    file_flair: UploadFile = File(...)  
):
    global is_processing
    
    # 3.A VERIFICA SEMÁFORO: Si está rojo, rechazar la petición inmediatamente
    if is_processing:
        raise HTTPException(
            status_code=503, 
            detail="El servidor está procesando otra petición. Intente en unos momentos."
        )

    # Crear carpeta temporal única
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = {}

    try:
        # 3.B PONER SEMÁFORO EN ROJO
        is_processing = True

        # Guardar archivos recibidos en disco (Esto es rápido, no bloquea casi nada)
        files_map = {"t1": file_t1, "t1ce": file_t1ce, "t2": file_t2, "flair": file_flair}
        for key, file_obj in files_map.items():
            file_path = f"{temp_dir}/{key}.nii.gz"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            saved_paths[key] = file_path
        
        output_filename = f"{temp_dir}/prediction.nii.gz"

        # 3.C EJECUTAR INFERENCIA EN THREADPOOL (Se hace el trabajo pesado en otro hilo, no bloquea el event loop de FastAPI)
        # Esto libera a FastAPI para que siga respondiendo a los GET /status
        await run_in_threadpool(
            run_inference_metastasis, 
            saved_paths, 
            output_filename, 
            TaskType.metastasis
        )

        # Subir a MinIO
        s3_path_in = f"{doctor_id}/{TaskType.metastasis.value}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{TaskType.metastasis.value}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

        # Guardar Metadata
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
        # 3.D LIMPIEZA CRÍTICA: Pase lo que pase, liberamos el servidor y borramos archivos
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/predict/acv",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
        503: {"description": "Servidor Ocupado"}
    },
)
async def predict_acv(
    doctor_id: str = Form(...),
    file_t1: UploadFile = File(...)
):
    global is_processing
    
    if is_processing:
        raise HTTPException(
            status_code=503, 
            detail="El servidor está procesando otra petición. Intente en unos momentos."
        )

    files_map = {"t1": file_t1}
    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = {}

    try:
        is_processing = True

        for key, file_obj in files_map.items():
            file_path = f"{temp_dir}/{key}.nii.gz"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            saved_paths[key] = file_path
        
        output_filename = f"{temp_dir}/prediction.nii.gz"

        # DELEGAR A THREADPOOL
        await run_in_threadpool(
            run_inference_acv, 
            saved_paths, 
            output_filename, 
            TaskType.acv
        )

        s3_path_in = f"{doctor_id}/{TaskType.acv.value}/{job_id}/input_t1.nii.gz"
        s3_path_out = f"{doctor_id}/{TaskType.acv.value}/{job_id}/prediction.nii.gz"

        url_in = upload_file(saved_paths["t1"], s3_path_in)
        url_out = upload_file(output_filename, s3_path_out)

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
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/predict/alzheimer",
    response_model=AlzheimerPredictionResponse,
    responses={
        400: {"model": APIErrorSchema},
        500: {"model": APIErrorSchema},
        503: {"description": "Servidor Ocupado"}
    },
)
async def predict_alzheimer(
    doctor_id: str = Form(...),
    file_t1: UploadFile = File(...),
):
    global is_processing
    
    if is_processing:
        raise HTTPException(
            status_code=503, 
            detail="El servidor está procesando otra petición. Intente en unos momentos."
        )

    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        is_processing = True

        file_path = f"{temp_dir}/t1.nii.gz"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_t1.file, buffer)
        
        saved_paths = {"t1": file_path}

        # DELEGAR A THREADPOOL (Atrapamos el valor de retorno dict)
        result = await run_in_threadpool(run_inference_alzheimer, saved_paths)
        
        s3_path_in = f"{doctor_id}/alzheimer/{job_id}/input_t1.nii.gz"
        url_in = upload_file(saved_paths["t1"], s3_path_in)
        
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
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)