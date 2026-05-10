import io
import pytest


def get_dummy_file(filename="test.nii.gz"):
    """Crea un archivo binario en memoria simulando un NIfTI."""
    return (filename, io.BytesIO(b"dummy_data"), "application/gzip")


def get_bad_file():
    """Crea un archivo con extensión inválida."""
    return ("test.jpg", io.BytesIO(b"image_data"), "image/jpeg")


def test_system_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json()["status"] == "free"


def test_predict_acv_success(client, mock_mongo):
    files = {"file_t1": get_dummy_file("paciente_t1.nii.gz")}
    data = {"doctor_id": "DrSmith", "paciente_id": "Paciente123"}

    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["paciente_id"] == "Paciente123"
    assert "fake.nii.gz" in res_data["prediction_image"]

    # Verificar que MongoMock insertó la metadata
    assert mock_mongo.count_documents({}) == 1


def test_predict_acv_invalid_extension(client):
    files = {"file_t1": get_bad_file()}
    data = {"doctor_id": "DrSmith", "paciente_id": "Paciente123"}

    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 400
    assert "Extensión inválida" in response.json()["detail"]


def test_predict_acv_missing_fields(client):
    files = {"file_t1": get_dummy_file()}
    # Falta el paciente_id o está vacío
    data = {"doctor_id": "DrSmith", "paciente_id": "  "}

    response = client.post("/predict/acv", data=data, files=files)

    assert response.status_code == 400
    assert "no puede estar vacío" in response.json()["detail"]


def test_predict_metastasis_success(client):
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
    # Insertar registros ficticios en el MongoMock
    mock_mongo.insert_many(
        [
            {"doctor_id": "DrA", "paciente_id": "Pac1", "task_type": "acv", "status": "completed"},
            {"doctor_id": "DrA", "paciente_id": "Pac2", "task_type": "metastasis", "status": "completed"},
            {"doctor_id": "DrB", "paciente_id": "Pac1", "task_type": "alzheimer", "status": "completed"},
        ]
    )

    # Probar historial de Paciente
    res_pac = client.get("/history/patient/Pac1")
    assert res_pac.status_code == 200
    assert res_pac.json()["meta"]["total_records"] == 2

    # Probar historial de Doctor
    res_doc = client.get("/history/doctor/DrA")
    assert res_doc.status_code == 200
    assert res_doc.json()["meta"]["total_records"] == 2
