import os
import torch
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from app.models.architecture import UNet3D

# Rutas de modelos (en app/models/)
MODEL_PATH_METS = "app/models/best_model_mets.pth"
MODEL_PATH_ACV = "app/models/best_model_acv.pth"

# Dispositivo (GPU si hay, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables globales para mantener los modelos en memoria
model_mets = None
model_acv = None


def load_models():
    """
    Carga los modelos en memoria al iniciar la API.
    Se llama desde el evento 'lifespan' en main.py.
    """
    global model_mets, model_acv
    print(f"Usando dispositivo: {device}")

    # --- 1. Cargar Modelo METÁSTASIS (4 Canales) ---
    try:
        # Instanciar arquitectura
        model_mets = UNet3D(in_channels=4, out_channels=1).to(device)

        # Cargar pesos
        if os.path.exists(MODEL_PATH_METS):
            state_dict = torch.load(MODEL_PATH_METS, map_location=device)
            model_mets.load_state_dict(state_dict)
            model_mets.eval()  # Modo evaluación (apaga Dropout/Batchnorm updates)
            print("✅ Modelo Metástasis cargado correctamente.")
        else:
            print(
                f"⚠️ Alerta: No se encontró {MODEL_PATH_METS}. La inferencia de Mets fallará.")

    except Exception as e:
        print(f"❌ Error cargando Modelo Metástasis: {e}")

    # --- 2. Cargar Modelo ACV (1 Canal - T1) ---
    try:
        # Instanciar arquitectura
        model_acv = UNet3D(in_channels=1, out_channels=1).to(device)

        # Cargar pesos
        if os.path.exists(MODEL_PATH_ACV):
            state_dict = torch.load(MODEL_PATH_ACV, map_location=device)
            model_acv.load_state_dict(state_dict)
            model_acv.eval()
            print("✅ Modelo ACV cargado correctamente.")
        else:
            print(
                f"⚠️ Alerta: No se encontró {MODEL_PATH_ACV}. La inferencia de ACV fallará.")

    except Exception as e:
        print(f"❌ Error cargando Modelo ACV: {e}")


def robust_normalization(img_data):
    """
    Normalización Específica para ACV (Atlas).
    Clip 1-99 percentil + Z-Score.
    """
    mask = img_data > 0
    if np.sum(mask) == 0:
        return img_data

    p01 = np.percentile(img_data[mask], 1)
    p99 = np.percentile(img_data[mask], 99)
    img_data = np.clip(img_data, p01, p99)

    mean = np.mean(img_data[mask])
    std = np.std(img_data[mask])
    return (img_data - mean) / (std + 1e-8)


def z_score_normalization(img_data):
    """
    Normalización Estándar para Metástasis.
    Solo Z-Score sobre la región del cerebro (mask > 0).
    """
    mask = img_data > 0
    if np.sum(mask) == 0:
        return img_data

    mean = np.mean(img_data[mask])
    std = np.std(img_data[mask])
    return (img_data - mean) / (std + 1e-8)


def preprocess_multichannel(paths_dict, task_type):
    """
    Maneja la carga y fusión de múltiples canales.
    """
    # 1. Cargar el canal principal (T1) para obtener affine/header
    img_t1 = nib.load(paths_dict["t1"])
    affine = img_t1.affine
    header = img_t1.header

    if task_type == "acv":
        # Caso Simple: Solo 1 canal
        data = img_t1.get_fdata()  # (H, W, D)
        data = robust_normalization(data)
        # Añadir canal dim: (C=1, H, W, D)
        data = np.expand_dims(data, axis=0)

    else:
        # Caso Metástasis: 4 Canales (T1, T1CE, T2, FLAIR)
        # ORDEN CRÍTICO: Debe ser el mismo usado en el entrenamiento.
        # Asumiremos el orden estándar de BraTS: [T1, T1CE, T2, FLAIR]
        # Si tu modelo se entrenó en otro orden, cambia esta lista.
        ordered_keys = ["t1", "t1ce", "t2", "flair"]
        channels_data = []

        for key in ordered_keys:
            if key not in paths_dict:
                raise ValueError(f"Falta la secuencia {key} para metástasis")

            img = nib.load(paths_dict[key])
            d = img.get_fdata()

            # Chequeo de seguridad: Dimensiones iguales
            if d.shape != img_t1.shape:
                raise ValueError(
                    f"Dimension mismatch: {key} tiene shape {d.shape}, pero T1 tiene {img_t1.shape}")

            # Normalizar individualmente (Z-Score por canal)
            d = z_score_normalization(d)
            channels_data.append(d)

        # Apilar canales: Resultado (4, H, W, D)
        data = np.stack(channels_data, axis=0)

    # 2. Convertir a Tensor PyTorch (Batch, Channel, D, H, W)
    # Numpy es (C, H, W, D) -> PyTorch espera (B, C, D, H, W) usualmente para 3D
    # Pero cuidado: nibabel carga (X, Y, Z).
    # MONAI Sliding Window espera (Batch, Channel, Spatial...)

    tensor = torch.from_numpy(data).float()
    tensor = tensor.unsqueeze(0)  # Añadir Batch -> (1, 4, H, W, D)

    return tensor, affine, header


def run_inference(saved_paths_dict, output_path, task_type):
    """
    Recibe un diccionario de rutas {"t1": path, "t2": path...}
    """
    model = model_mets if task_type == "metastasis" else model_acv

    if model is None:
        raise ValueError(f"Modelo para {task_type} no cargado.")

    # Llamamos al nuevo preprocesador multicanal
    tensor_img, affine, header = preprocess_multichannel(
        saved_paths_dict, task_type)
    tensor_img = tensor_img.to(device)

    print(f"Inferencia {task_type}. Tensor shape: {tensor_img.shape}")

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=tensor_img,
            roi_size=(64, 64, 64),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )

        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.5).float()

        # (1, 1, X, Y, Z) -> (X, Y, Z)
        pred_mask = pred_mask.cpu().numpy()[0, 0]

    result_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine, header)
    nib.save(result_img, output_path)

    return output_path
