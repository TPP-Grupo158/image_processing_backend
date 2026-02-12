import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Bloque conv3d -> BN -> ReLU aplicado dos veces.
    Esta es la pieza fundamental de la U-Net.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    """
    Implementación estándar de 3D U-Net.
    Parametros:
      in_channels: 4 para Metástasis (Bravo, T1, T2, FLAIR), 1 para ACV (T1).
      out_channels: 1 (Máscara binaria).
      features: Profundidad de la red [32, 64, 128, 256].
    """

    def __init__(self, in_channels=4, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Parte Encoder (Bajada)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck (Parte más profunda)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Parte Decoder (Subida)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Capa Final (Conv 1x1)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Paso hacia abajo (Encoder)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Invertir las conexiones para usarlas al subir
        skip_connections = skip_connections[::-1]

        # Paso hacia arriba (Decoder)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip_connection = skip_connections[idx//2]

            # Concatenar con la skip connection correspondiente
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # DoubleConv

        return self.final_conv(x)
