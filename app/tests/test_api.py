import io
from datetime import datetime
from unittest.mock import patch


def get_dummy_file(filename="test.nii.gz"):
    """
    Fixture auxiliar: Crea un archivo binario en RAM.
    Evita tener que guardar archivos .nii.gz reales y pesados en el repositorio de GitHub.
    """
    return (filename, io.BytesIO(b"dummy_data"), "application/gzip")


def get_bad_file():
    """Simula una foto subida por error para probar la validación de extensiones."""
    return ("test.jpg", io.BytesIO(b"image_data"), "image/jpeg")


def test_system_status_free(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["status"] == "free"


def test_system_status_busy(client):
    """Verifica que el endpoint refleje el estado cuando la variable global cambia."""
    import app.main

    app.main.is_processing = True
    try:
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["status"] == "busy"
    finally:
        app.main.is_processing = False


def test_predict_busy(client):
    """Asegura que el servidor rechace peticiones con 503 si el modelo ya está inferiendo."""
    import app.main

    app.main.is_processing = True
    try:
        files = {"file_t1": get_dummy_file()}
        data = {"doctor_id": "Doc", "paciente_id": "Pac"}
        response = client.post("/predict/acv", data=data, files=files)
        assert response.status_code == 503
        assert "Servidor ocupado" in response.json()["detail"]
    finally:
        app.main.is_processing = False


def test_predict_acv_success(client, mock_mongo):
    """Test de integración del flujo exitoso. Comprueba guardado en base de datos mockeada."""
    files = {"file_t1": get_dummy_file("paciente_t1.nii.gz")}
    data = {"doctor_id": "DrSmith", "paciente_id": "Paciente123"}
    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["paciente_id"] == "Paciente123"
    assert "fake.nii.gz" in res_data["prediction_image"]
    assert mock_mongo.count_documents({}) == 1


def test_predict_acv_invalid_extension(client):
    """Comprueba el Fail-Fast de validación médica (HTTP 400)."""
    files = {"file_t1": get_bad_file()}
    data = {"doctor_id": "DrSmith", "paciente_id": "Paciente123"}
    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 400
    assert "Extensión inválida" in response.json()["detail"]


def test_predict_acv_missing_fields(client):
    files = {"file_t1": get_dummy_file()}
    data = {"doctor_id": "DrSmith", "paciente_id": "  "}
    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 400
    assert "no puede estar vacío" in response.json()["detail"]


def test_predict_acv_internal_error(client):
    """
    Test de gestión de errores: Forzamos un fallo inyectando una excepción en el motor del modelo
    para garantizar que el sistema no se caiga y devuelva un HTTP 500 controlado.
    """
    files = {"file_t1": get_dummy_file()}
    data = {"doctor_id": "DrSmith", "paciente_id": "Paciente123"}

    with patch("app.main.run_inference_acv", side_effect=Exception("Simulated Crash")):
        response = client.post("/predict/acv", data=data, files=files)
        assert response.status_code == 500
        assert response.json()["type"] == "InternalError"
        assert "Simulated Crash" in response.json()["detail"]


def test_predict_metastasis_success(client):
    """Comprueba el flujo de entrada multi-modal (4 secuencias obligatorias)."""
    files = {
        "t1_pre": get_dummy_file("t1.nii.gz"),
        "t1_gd": get_dummy_file("t1gd.nii.gz"),
        "flair": get_dummy_file("flair.nii.gz"),
        "bravo": get_dummy_file("bravo.nii.gz"),
    }
    data = {"doctor_id": "DrHouse", "paciente_id": "Mets001"}
    response = client.post("/predict/metastasis", data=data, files=files)

    assert response.status_code == 200
    assert response.json()["task"] == "metastasis"
    assert len(response.json()["modalities_used"]) == 4


def test_predict_alzheimer_success(client):
    files = {"file_t1": get_dummy_file()}
    data = {"doctor_id": "DrNeuro"}
    response = client.post("/predict/alzheimer", data=data, files=files)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["prediction"] == 1
    assert res_data["probability"] == 0.98


def test_history_endpoints(client, mock_mongo):
    # Generamos un timestamp falso único para todo el batch de prueba
    fake_time = datetime.utcnow()
    mock_mongo.insert_many(
        [
            {
                "doctor_id": "DrA",
                "paciente_id": "Pac1",
                "task_type": "acv",
                "status": "completed",
                "created_at": fake_time,
                "original_image_t1": "http://url/t1.nii.gz",
            },
            {
                "doctor_id": "DrA",
                "paciente_id": "Pac2",
                "task_type": "metastasis",
                "status": "completed",
                "created_at": fake_time,
                "original_image_t1_gd": "http://url/t1_gd.nii.gz",
                "original_image_bravo": "http://url/bravo.nii.gz",
            },
            {"doctor_id": "DrB", "paciente_id": "Pac1", "task_type": "alzheimer", "status": "completed", "created_at": fake_time},
        ]
    )

    res_pac = client.get("/history/patient/Pac1")
    assert res_pac.status_code == 200
    assert res_pac.json()["meta"]["total_records"] == 2

    # FIX ANTI-FLAKY: Como los 3 documentos se insertaron en el microsegundo exacto, MongoDB
    # no garantiza el orden al consultar. Usar any() asegura que busquemos la imagen 't1'
    # dinámicamente sin importar si el array llegó desordenado, previniendo fallos.
    assert any("t1" in record.get("original_images", {}) for record in res_pac.json()["data"])

    res_doc = client.get("/history/doctor/DrA")
    assert res_doc.status_code == 200
    assert res_doc.json()["meta"]["total_records"] == 2
