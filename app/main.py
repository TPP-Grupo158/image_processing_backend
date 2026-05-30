from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import shutil
import os
import uuid
import json

# Importaciones del proyecto
from app.core.inference import (
    load_models,
    run_inference_alzheimer,
    run_inference_acv,
    run_inference_metastasis,
    create_best_slice_visualization,
)
from app.core.database import save_prediction_metadata, get_paginated_history
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import InternalError
from app.schemas import PredictionResponse, APIErrorSchema, TaskType, PaginatedHistoryResponse
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
        return {"status": "busy", "color": "red", "message": "El equipo está procesando otra imagen. Por favor, aguarde."}
    return {"status": "free", "color": "green", "message": "Sistema listo para recibir imágenes."}


# ==========================================
# 3. ENDPOINTS DE INFERENCIA
# ==========================================


@app.post("/predict/metastasis", response_model=PredictionResponse)
async def predict_metastasis(
    doctor_id: str = Form(...),
    paciente_id: str = Form(...),
    t1_pre: UploadFile = File(...),
    t1_gd: UploadFile = File(...),
    flair: UploadFile = File(...),
    bravo: UploadFile = File(...),
):
    global is_processing
    if is_processing:
        raise HTTPException(status_code=503, detail="Servidor ocupado")

    study_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{study_id}"
    os.makedirs(temp_dir, exist_ok=True)

    saved_paths = {}
    input_urls = {}

    try:
        is_processing = True

        # 1. Guardar localmente para procesamiento
        files_map = {"t1_pre": t1_pre, "t1_gd": t1_gd, "flair": flair, "bravo": bravo}
        for key, file_obj in files_map.items():
            file_path = os.path.join(temp_dir, f"{key}.nii.gz")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            saved_paths[key] = file_path

        output_local = os.path.join(temp_dir, "prediction.nii.gz")

        # 2. Inferencia
        await run_in_threadpool(run_inference_metastasis, saved_paths, output_local, TaskType.metastasis)

        # Generar JPG del mejor corte. Elegimos t1_gd o bravo como representativo estructural
        rep_img_path = saved_paths.get("t1_gd", saved_paths.get("bravo", saved_paths["t1_pre"]))
        jpg_local = os.path.join(temp_dir, "visualization.jpg")
        await run_in_threadpool(
            create_best_slice_visualization, rep_img_path, output_local, paciente_id, jpg_local, TaskType.metastasis
        )

        # Subida a MinIO
        for key, local_path in saved_paths.items():
            s3_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/{key}.nii.gz"
            input_urls[key] = upload_file(local_path, s3_path)

        s3_prediction_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/prediction.nii.gz"
        prediction_url = upload_file(output_local, s3_prediction_path)

        # Subir JPG a MinIO
        s3_jpg_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/visualization.jpg"
        visualization_url = upload_file(jpg_local, s3_jpg_path)

        db_id = save_prediction_metadata(
            doctor_id=doctor_id,
            paciente_id=paciente_id,
            task_type=TaskType.metastasis.value,
            input_images=input_urls,
            prediction_url=prediction_url,
            visualization_url=visualization_url,  # Se pasa a la BD
        )

        return PredictionResponse(
            status="success",
            db_id=db_id,
            paciente_id=paciente_id,
            doctor_id=doctor_id,
            original_images=input_urls,
            prediction_image=prediction_url,
            visualization_image=visualization_url,
            task=TaskType.metastasis.value,
            modalities_used=list(saved_paths.keys()),
        )

    except Exception as e:
        raise InternalError(detail=str(e))
    finally:
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/predict/acv", response_model=PredictionResponse)
async def predict_acv(doctor_id: str = Form(...), paciente_id: str = Form(...), file_t1: UploadFile = File(...)):
    global is_processing
    if is_processing:
        raise HTTPException(status_code=503, detail="Servidor ocupado")

    study_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{study_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        is_processing = True
        local_t1 = os.path.join(temp_dir, "file_t1.nii.gz")
        with open(local_t1, "wb") as buffer:
            shutil.copyfileobj(file_t1.file, buffer)

        output_local = os.path.join(temp_dir, "prediction.nii.gz")
        await run_in_threadpool(run_inference_acv, {"t1": local_t1}, output_local, TaskType.acv)

        # Generar JPG
        jpg_local = os.path.join(temp_dir, "visualization.jpg")
        await run_in_threadpool(
            create_best_slice_visualization, local_t1, output_local, paciente_id, jpg_local, TaskType.acv
        )

        # Subida a MinIO
        s3_t1_path = f"{paciente_id}/{TaskType.acv.value}/{study_id}/file_t1.nii.gz"
        s3_pred_path = f"{paciente_id}/{TaskType.acv.value}/{study_id}/prediction.nii.gz"
        s3_jpg_path = f"{paciente_id}/{TaskType.acv.value}/{study_id}/visualization.jpg"

        url_t1 = upload_file(local_t1, s3_t1_path)
        url_pred = upload_file(output_local, s3_pred_path)
        url_jpg = upload_file(jpg_local, s3_jpg_path)

        db_id = save_prediction_metadata(
            doctor_id=doctor_id,
            paciente_id=paciente_id,
            task_type=TaskType.acv.value,
            input_images={"t1": url_t1},
            prediction_url=url_pred,
            visualization_url=url_jpg,
        )

        return PredictionResponse(
            status="success",
            db_id=db_id,
            paciente_id=paciente_id,
            doctor_id=doctor_id,
            original_images={"t1": url_t1},
            prediction_image=url_pred,
            visualization_image=url_jpg,
            task=TaskType.acv.value,
            modalities_used=["t1"],
        )

    except Exception as e:
        raise InternalError(detail=str(e))
    finally:
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/predict/alzheimer",
    response_model=PredictionResponse,
    responses={400: {"model": APIErrorSchema}, 500: {"model": APIErrorSchema}, 503: {"description": "Servidor Ocupado"}},
)
async def predict_alzheimer(doctor_id: str = Form(...), paciente_id: str = Form(...), file_t1: UploadFile = File(...)):
    global is_processing
    if is_processing:
        raise HTTPException(status_code=503, detail="El servidor está procesando otra petición.")

    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        is_processing = True
        # Guardar archivo T1
        file_path = f"{temp_dir}/t1.nii.gz"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_t1.file, buffer)
        
        saved_paths = {"t1": file_path}
        result = await run_in_threadpool(run_inference_alzheimer, saved_paths)
        result = {"prediction": result["prediction"]==1, "probability": result["probability"]}

        # Subir solo la imagen original
        s3_path_in = f"{doctor_id}/{TaskType.alzheimer.value}/{job_id}/input_t1.nii.gz"
        url_in = upload_file(saved_paths["t1"], s3_path_in)
        
        # Guardar JSON en archivo temporal y subirlo a MinIO
        json_path = f"{temp_dir}/result.json"
        with open(json_path, "w") as file:
            json.dump(result, file, indent=2)
        
        s3_json_path = f"{doctor_id}/{TaskType.alzheimer.value}/{job_id}/result.json"
        url_json = upload_file(json_path, s3_json_path)
        
        # Guardar metadata
        db_id = save_prediction_metadata(doctor_id, paciente_id, TaskType.alzheimer.value, {"t1": url_in}, url_json)
        
        return PredictionResponse(
            status="success",
            db_id=db_id,
            paciente_id=paciente_id,
            doctor_id=doctor_id,
            original_images={"t1": url_in},
            prediction_image=url_json,
            task=TaskType.alzheimer.value,
            modalities_used=list(saved_paths.keys())
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise InternalError(detail=str(e))
    finally:
        is_processing = False
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/history/patient/{paciente_id}", response_model=PaginatedHistoryResponse)
async def get_patient_history(paciente_id: str, page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    try:
        result = get_paginated_history({"paciente_id": paciente_id}, page, limit)
        return result
    except Exception as e:
        raise InternalError(detail=str(e))


@app.get("/history/doctor/{doctor_id}", response_model=PaginatedHistoryResponse)
async def get_doctor_history(doctor_id: str, page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=100)):
    try:
        result = get_paginated_history({"doctor_id": doctor_id}, page, limit)
        return result
    except Exception as e:
        raise InternalError(detail=str(e))
