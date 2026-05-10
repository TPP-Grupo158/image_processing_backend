import pytest
import os
from unittest.mock import patch
import mongomock
from fastapi.testclient import TestClient

# Forzamos la variable de entorno para desactivar el lifespan real
os.environ["TESTING_MODE"] = "True"

from app.main import app


@pytest.fixture
def client():
    """Provee un cliente HTTP para testear FastAPI"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def reset_server_state():
    """Asegura que el semáforo global se resetee antes de cada test"""
    import app.main

    app.main.is_processing = False


@pytest.fixture(autouse=True)
def mock_mongo():
    """Moquea MongoDB Atlas usando MongoMock en memoria"""
    mock_client = mongomock.MongoClient()
    db = mock_client["medical_db"]
    mock_collection = db["predictions"]

    with patch("app.core.database.collection", mock_collection):
        yield mock_collection


@pytest.fixture(autouse=True)
def mock_storage():
    """Evita la subida a MinIO, devolviendo una URL fake"""
    with patch("app.main.upload_file", return_value="http://mock-minio:9000/medical-images/fake.nii.gz"):
        yield


@pytest.fixture(autouse=True)
def mock_inference():
    """Evita que PyTorch y la IA se ejecuten durante los tests"""

    def dummy_segmentation(paths, out, task):
        # Simula que se creó el archivo de salida
        with open(out, "wb") as f:
            f.write(b"dummy nifti mask")
        return out

    def dummy_visualization(orig, pred, pid, out, task):
        # Simula que matplotlib generó el JPG
        with open(out, "wb") as f:
            f.write(b"dummy jpg image")

    with (
        patch("app.main.run_inference_metastasis", side_effect=dummy_segmentation),
        patch("app.main.run_inference_acv", side_effect=dummy_segmentation),
        patch("app.main.run_inference_alzheimer", return_value={"prediction": 1, "probability": 0.98, "threshold": 0.5}),
        patch("app.main.create_best_slice_visualization", side_effect=dummy_visualization),
    ):
        yield
