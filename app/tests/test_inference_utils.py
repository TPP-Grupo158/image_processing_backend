import numpy as np
from app.core.inference import robust_normalization, z_score_normalization


def test_z_score_normalization():
    # Creamos una matriz con ceros (máscara inactiva) y valores numéricos
    data = np.array([0.0, 2.0, 4.0, 6.0])

    out = z_score_normalization(data)

    assert out.shape == data.shape
    assert out[0] == 0.0  # El fondo (0) no debe alterarse

    # La media de [2, 4, 6] es 4. Tras Z-score, la media de los píxeles activos debe ser aprox 0
    mask = data > 0
    assert np.isclose(np.mean(out[mask]), 0.0, atol=1e-5)


def test_robust_normalization():
    # Simula valores atípicos (outliers) en imágenes médicas, ej: un píxel saturado a 1000
    data = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])

    out = robust_normalization(data)

    assert out.shape == data.shape
    assert out[0] == 0.0

    mask = data > 0
    # Media de elementos recortados ronda a 0 de forma robusta
    assert np.isclose(np.mean(out[mask]), 0.0, atol=1e-5)


def test_normalization_all_zeros():
    """El preprocesamiento no debe romperse si recibe un array vacío de información (ej. slide en negro)"""
    data = np.zeros((5, 5))
    out_z = z_score_normalization(data)
    out_r = robust_normalization(data)

    assert np.all(out_z == 0)
    assert np.all(out_r == 0)
