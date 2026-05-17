import torch
from app.models.architecture import DoubleConv, UNet3D


def test_double_conv_layer():
    """Prueba que el bloque fundamental Conv3d funciona y preserva las dimensiones"""
    in_channels = 1
    out_channels = 2
    layer = DoubleConv(in_channels, out_channels)

    # Creamos un batch de 1 imagen 3D de tamaño 8x8x8
    dummy_input = torch.randn(1, in_channels, 8, 8, 8)
    output = layer(dummy_input)

    assert output.shape == (1, out_channels, 8, 8, 8)


def test_unet3d_forward_pass():
    """
    Prueba la arquitectura completa de U-Net3D.
    Reducimos la profundidad de las features para que el test ejecute en milisegundos en CPU.
    """
    model = UNet3D(in_channels=4, out_channels=1, features=[8, 16])

    # Batch=1, Canales=4 (Simulando Metástasis), Profundidad=16, Alto=16, Ancho=16
    dummy_input = torch.randn(1, 4, 16, 16, 16)

    output = model(dummy_input)

    # La máscara de salida debe tener 1 canal y mantener la geometría de entrada original
    assert output.shape == (1, 1, 16, 16, 16)
