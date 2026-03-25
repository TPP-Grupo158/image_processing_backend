from minio import Minio
from minio.error import S3Error
from app.core.config import settings

# Cliente MinIO
client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False  # False porque estamos en local sin HTTPS
)

def initialize_storage():
    """
    Verifica la existencia del bucket al iniciar la aplicación.
    Si no existe, lo crea automáticamente.
    """
    try:
        # Verifica si el bucket existe
        if not client.bucket_exists(settings.MINIO_BUCKET):
            client.make_bucket(settings.MINIO_BUCKET)
            print(f"✅ Infraestructura: Bucket '{settings.MINIO_BUCKET}' creado exitosamente en MinIO.")
        else:
            print(f"✅ Infraestructura: Bucket '{settings.MINIO_BUCKET}' verificado y listo.")
    except S3Error as e:
        print(f"❌ Error crítico de S3 comunicándose con MinIO: {e}")
        raise e
    except Exception as e:
        print(f"❌ Error inesperado en inicialización de storage: {e}")
        raise e

def upload_file(file_path: str, object_name: str) -> str:
    """
    Sube un archivo a MinIO y retorna la URL pública para el frontend.
    """
    try:
        # Subir el archivo (ya sabemos que el bucket existe por el lifespan)
        client.fput_object(
            settings.MINIO_BUCKET,
            object_name,
            file_path,
            content_type="application/octet-stream"
        )

        url = f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{object_name}"
        return url

    except Exception as e:
        print(f"Error subiendo a MinIO: {e}")
        raise e