from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import shutil
import os
import uuid

from app.core.inference import load_models, run_inference_alzheimer, run_inference_acv, run_inference_metastasis, create_best_slice_visualization
from app.core.database import save_prediction_metadata, get_paginated_history
from app.errors.handlers import register_exception_handlers
from app.errors.http_errors import InternalError
from app.schemas import PredictionResponse, AlzheimerPredictionResponse, APIErrorSchema, TaskType, PaginatedHistoryResponse
from app.core.storage import upload_file, initialize_storage

is_processing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Condición para evitar cargar modelos si estamos corriendo tests
    if not os.getenv("TESTING_MODE"):
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
# FUNCIONES DE VALIDACIÓN
# ==========================================
def validate_medical_image(file: UploadFile):
    """Verifica que el archivo sea un volumen NIfTI válido."""
    if not file.filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail=f"Extensión inválida en {file.filename}. Se requiere .nii o .nii.gz")


def validate_identifiers(*args):
    """Verifica que los IDs de paciente y médico no estén vacíos."""
    for arg_name, arg_val in args:
        if not arg_val or not arg_val.strip():
            raise HTTPException(status_code=400, detail=f"El campo '{arg_name}' es obligatorio y no puede estar vacío.")


# ==========================================
# ENDPOINTS
# ==========================================


@app.get("/status")
async def get_system_status():
    if is_processing:
        return {"status": "busy", "color": "red", "message": "El equipo está procesando."}
    return {"status": "free", "color": "green", "message": "Sistema listo."}


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

    # VALIDACIONES
    validate_identifiers(("doctor_id", doctor_id), ("paciente_id", paciente_id))
    files_map = {"t1_pre": t1_pre, "t1_gd": t1_gd, "flair": flair, "bravo": bravo}
    for file_obj in files_map.values():
        validate_medical_image(file_obj)

    study_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{study_id}"
    os.makedirs(temp_dir, exist_ok=True)
    saved_paths = {}
    input_urls = {}

    try:
        is_processing = True

        for key, file_obj in files_map.items():
            file_path = os.path.join(temp_dir, f"{key}.nii.gz")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            saved_paths[key] = file_path

        output_local = os.path.join(temp_dir, "prediction.nii.gz")
        await run_in_threadpool(run_inference_metastasis, saved_paths, output_local, TaskType.metastasis)

        rep_img_path = saved_paths.get("t1_gd", saved_paths.get("bravo", saved_paths["t1_pre"]))
        jpg_local = os.path.join(temp_dir, "visualization.jpg")
        await run_in_threadpool(create_best_slice_visualization, rep_img_path, output_local, paciente_id, jpg_local, TaskType.metastasis)

        for key, local_path in saved_paths.items():
            s3_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/{key}.nii.gz"
            input_urls[key] = upload_file(local_path, s3_path)

        s3_prediction_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/prediction.nii.gz"
        prediction_url = upload_file(output_local, s3_prediction_path)

        s3_jpg_path = f"{paciente_id}/{TaskType.metastasis.value}/{study_id}/visualization.jpg"
        visualization_url = upload_file(jpg_local, s3_jpg_path)

        db_id = save_prediction_metadata(
            doctor_id=doctor_id,
            paciente_id=paciente_id,
            task_type=TaskType.metastasis.value,
            input_images=input_urls,
            prediction_url=prediction_url,
            visualization_url=visualization_url,
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

    # VALIDACIONES
    validate_identifiers(("doctor_id", doctor_id), ("paciente_id", paciente_id))
    validate_medical_image(file_t1)

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

        jpg_local = os.path.join(temp_dir, "visualization.jpg")
        await run_in_threadpool(create_best_slice_visualization, local_t1, output_local, paciente_id, jpg_local, TaskType.acv)

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


@app.post("/predict/alzheimer", response_model=AlzheimerPredictionResponse)
async def predict_alzheimer(doctor_id: str = Form(...), file_t1: UploadFile = File(...)):
    global is_processing
    if is_processing:
        raise HTTPException(status_code=503, detail="Servidor ocupado")

    validate_identifiers(("doctor_id", doctor_id))
    validate_medical_image(file_t1)

    job_id = str(uuid.uuid4())
    temp_dir = f"/tmp/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        is_processing = True
        file_path = f"{temp_dir}/t1.nii.gz"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_t1.file, buffer)

        saved_paths = {"t1": file_path}
        result = await run_in_threadpool(run_inference_alzheimer, saved_paths)

        s3_path_in = f"{doctor_id}/alzheimer/{job_id}/input_t1.nii.gz"
        url_in = upload_file(saved_paths["t1"], s3_path_in)

        db_id = save_prediction_metadata(doctor_id, "paciente_anonimo", TaskType.alzheimer.value, {"t1": url_in}, None)

        return AlzheimerPredictionResponse(
            status="success",
            db_id=db_id,
            paciente_id="paciente_anonimo",
            doctor_id=doctor_id,
            original_image=url_in,
            task=TaskType.alzheimer.value,
            prediction=result["prediction"],
            probability=result["probability"],
            threshold=result["threshold"],
            modalities_used=list(saved_paths.keys()),
        )
    except Exception as e:
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
