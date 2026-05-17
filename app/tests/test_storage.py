import pytest
from unittest.mock import patch, MagicMock
from minio.error import S3Error
from app.core.storage import initialize_storage, upload_file


@patch("app.core.storage.client")
def test_initialize_storage_bucket_exists(mock_client):
    """Prueba cuando MinIO ya tiene el bucket creado"""
    mock_client.bucket_exists.return_value = True
    initialize_storage()
    mock_client.make_bucket.assert_not_called()


@patch("app.core.storage.client")
def test_initialize_storage_bucket_not_exists(mock_client):
    """Prueba cuando el sistema debe crear el bucket en el primer inicio"""
    mock_client.bucket_exists.return_value = False
    initialize_storage()
    mock_client.make_bucket.assert_called_once()


@patch("app.core.storage.client")
def test_initialize_storage_s3_error(mock_client):
    """Asegura que el servicio tire la excepción si MinIO está caído"""
    mock_client.bucket_exists.side_effect = S3Error(
        code="NoSuchBucket", message="S3 Down", resource="/", request_id="1", host_id="1", response=MagicMock()
    )
    with pytest.raises(S3Error):
        initialize_storage()


@patch("app.core.storage.client")
def test_upload_file(mock_client):
    """Verifica el formato del retorno de la subida a MinIO"""
    url = upload_file("local_path/file.nii.gz", "remote_path/file.nii.gz")

    mock_client.fput_object.assert_called_once()
    assert "remote_path/file.nii.gz" in url
    assert url.startswith("http://")
