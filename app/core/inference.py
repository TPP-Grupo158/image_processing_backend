import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Resized,
    ScaleIntensityd, ToTensord
)
from app.models.architecture import UNet3D
from monai.networks.nets import DenseNet121
from app.schemas import TaskType

# Rutas de modelos (en app/models/)
MODEL_PATH_METS = "app/models/best_model_mets.pth"
MODEL_PATH_ACV = "app/models/best_model_acv.pth"
MODEL_PATH_ALZHEIMER = "app/models/best_model_alzheimer.pt"

# Dispositivo (GPU si hay, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variables globales para mantener los modelos en memoria
model_mets = None
model_acv = None
model_alzheimer = None
threshold = None  # Threshold por defecto para Alzheimer

def load_models():
    """
    Carga los modelos en memoria al iniciar la API.
    Se llama desde el evento 'lifespan' en main.py.
    """
    global model_mets, model_acv, model_alzheimer, threshold
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

    # ---3. Cargar Modelo Alzheimer (1 Canal - T1)---
    try:
        model_alzheimer = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        if os.path.exists(MODEL_PATH_ALZHEIMER):
            checkpoint = torch.load(MODEL_PATH_ALZHEIMER, map_location=device, weights_only=False)
            model_alzheimer.load_state_dict(checkpoint['model_state'])
            threshold = checkpoint["threshold"]
            model_alzheimer.eval()
            print("✅ Modelo Alzheimer cargado correctamente.")
        else:
            print(
                f"⚠️ Alerta: No se encontró {MODEL_PATH_ALZHEIMER}. La inferencia de Alzheimer fallará.")
    
    except Exception as e:
        print(f"❌ Error cargando Modelo Alzheimer: {e}")


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


def preprocess_alzheimer(image_path):
    """
    Preprocesamiento específico para Alzheimer usando transformaciones MONAI.
    Retorna tensor listo para el modelo.
    """
    # Transformaciones idénticas al pipeline de Kaggle
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0)),
        Resized(keys=["image"], spatial_size=(176, 256, 256)),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])
    
    # Aplicar transformaciones
    data = transforms({"image": image_path})
    tensor_img = data["image"]
    
    # Asegurar batch dim: (C, H, W, D) -> (B, C, H, W, D)
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.unsqueeze(0)
    
    return tensor_img


def preprocess_multichannel(paths_dict, task_type: TaskType):
    """
    Maneja la carga y fusión de múltiples canales.
    """
    # 1. Cargar el canal principal (T1) para obtener affine/header
    img_t1 = nib.load(paths_dict["t1"])
    affine = img_t1.affine
    header = img_t1.header

    if task_type == TaskType.acv:
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


def run_inference_alzheimer(saved_paths_dict):
    """
    Ejecuta inferencia para Alzheimer (clasificación binaria).
    Aplica transformaciones MONAI específicas del modelo.
    Retorna: {"prediction": 0 o 1, "probability": probabilidad del caso positivo}
    """
    global model_alzheimer, threshold
    
    if model_alzheimer is None:
        raise ValueError("Modelo Alzheimer no cargado.")
    
    # Obtener ruta de la imagen T1
    image_path = saved_paths_dict["t1"]
    
    # Preprocesar con transformaciones MONAI
    tensor_img = preprocess_alzheimer(image_path)
    tensor_img = tensor_img.to(device)
    
    print(f"Inferencia Alzheimer. Tensor shape: {tensor_img.shape}")
    
    with torch.no_grad():
        output = model_alzheimer(tensor_img)
        # output shape: (batch, num_classes=2)
        probs = F.softmax(output, dim=1)[:, 1]  # Probabilidad de clase 1
        prob_value = probs.cpu().numpy()[0]
        
        # Usar threshold guardado durante training
        prediction = int(prob_value >= threshold)
    
    return {
        "prediction": prediction,
        "probability": float(prob_value),
        "threshold": float(threshold)
    }


def run_inference(saved_paths_dict, output_path, task_type: TaskType):
    """
    Recibe un diccionario de rutas {"t1": path, "t2": path...}
    
    Retorna:
    - Para ACV y Metástasis: ruta del archivo segmentado
    - Para Alzheimer: diccionario con clasificación {"prediction": 0 o 1, "probability": float}
    """
    
    if task_type == TaskType.alzheimer:
        return run_inference_alzheimer(saved_paths_dict)
    
    model = model_mets if task_type == TaskType.metastasis else model_acv

    if model is None:
        raise ValueError(f"Modelo para {task_type.value} no cargado.")

    # Llamamos al nuevo preprocesador multicanal
    tensor_img, affine, header = preprocess_multichannel(
        saved_paths_dict, task_type)
    tensor_img = tensor_img.to(device)

    print(f"Inferencia {task_type.value}. Tensor shape: {tensor_img.shape}")

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
