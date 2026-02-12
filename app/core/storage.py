from minio import Minio
from app.core.config import settings
import os

# Cliente MinIO
client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False  # False porque estamos en local sin HTTPS
)


def upload_file(file_path: str, object_name: str) -> str:
    """
    Sube un archivo a MinIO y retorna la URL pública para el frontend.
    """
    try:
        # Subir el archivo
        client.fput_object(
            settings.MINIO_BUCKET,
            object_name,
            file_path,
            content_type="application/octet-stream"
        )

        # Construir URL manual para que funcione en el navegador
        # Formato: http://localhost:9000/medical-images/doctor_123/...
        url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
        return url

    except Exception as e:
        print(f"Error subiendo a MinIO: {e}")
        raise e
