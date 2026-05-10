import io
import pytest


def get_dummy_file(filename="test.nii.gz"):
    """Crea un archivo binario en memoria simulando un NIfTI."""
    return (filename, io.BytesIO(b"dummy_data"), "application/gzip")


def get_bad_file():
    """Crea un archivo con extensión inválida."""
    return ("test.jpg", io.BytesIO(b"image_data"), "image/jpeg")
