import os
import torch
import torch.nn.functional as F
import nibabel as nib
import numpy as np

# Configuración headless estricta para el microservicio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Resized, ScaleIntensityd, ToTensord
from app.models.architecture import UNet3D
from monai.networks.nets import DenseNet121
from app.schemas import TaskType

MODEL_PATH_METS = "app/models/best_model_mets.pth"
MODEL_PATH_ACV = "app/models/best_model_acv.pth"
MODEL_PATH_ALZHEIMER = "app/models/best_model_alzheimer.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_mets = None
model_acv = None
model_alzheimer = None
threshold = None


def load_models():
    global model_mets, model_acv, model_alzheimer, threshold
    print(f"Usando dispositivo: {device}")

    try:
        model_mets = UNet3D(in_channels=4, out_channels=1).to(device)
        if os.path.exists(MODEL_PATH_METS):
            state_dict = torch.load(MODEL_PATH_METS, map_location=device, weights_only=True)
            model_mets.load_state_dict(state_dict)
            model_mets.eval()
            print("✅ Modelo Metástasis cargado correctamente.")
        else:
            print(f"⚠️ Alerta: No se encontró {MODEL_PATH_METS}.")
    except Exception as e:
        print(f"❌ Error cargando Modelo Metástasis: {e}")

    try:
        model_acv = UNet3D(in_channels=1, out_channels=1).to(device)
        if os.path.exists(MODEL_PATH_ACV):
            state_dict = torch.load(MODEL_PATH_ACV, map_location=device, weights_only=True)
            model_acv.load_state_dict(state_dict)
            model_acv.eval()
            print("✅ Modelo ACV cargado correctamente.")
        else:
            print(f"⚠️ Alerta: No se encontró {MODEL_PATH_ACV}.")
    except Exception as e:
        print(f"❌ Error cargando Modelo ACV: {e}")

    try:
        model_alzheimer = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        if os.path.exists(MODEL_PATH_ALZHEIMER):
            checkpoint = torch.load(MODEL_PATH_ALZHEIMER, map_location=device, weights_only=False)
            model_alzheimer.load_state_dict(checkpoint["model_state"])
            threshold = checkpoint["threshold"]
            model_alzheimer.eval()
            print("✅ Modelo Alzheimer cargado correctamente.")
        else:
            print(f"⚠️ Alerta: No se encontró {MODEL_PATH_ALZHEIMER}.")
    except Exception as e:
        print(f"❌ Error cargando Modelo Alzheimer: {e}")


def robust_normalization(img_data):
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
    mask = img_data > 0
    if np.sum(mask) == 0:
        return img_data
    mean = np.mean(img_data[mask])
    std = np.std(img_data[mask])
    return (img_data - mean) / (std + 1e-8)


def preprocess_alzheimer(image_path):
    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0)),
            Resized(keys=["image"], spatial_size=(176, 256, 256)),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image"]),
        ]
    )
    data = transforms({"image": image_path})
    tensor_img = data["image"]
    if tensor_img.dim() == 4:
        tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


def preprocess_metastasis(paths_dict):
    img_ref = nib.load(paths_dict["t1_pre"])
    affine = img_ref.affine
    header = img_ref.header

    ordered_keys = ["t1_pre", "t1_gd", "flair", "bravo"]
    channels_data = []

    for key in ordered_keys:
        if key not in paths_dict:
            raise ValueError(f"Falta la secuencia obligatoria: {key}")
        img = nib.load(paths_dict[key])
        d = img.get_fdata()
        if d.shape != img_ref.shape:
            raise ValueError(f"Error de dimensiones: {key} {d.shape} != t1_pre {img_ref.shape}")
        d = z_score_normalization(d)
        channels_data.append(d)

    data = np.stack(channels_data, axis=0)
    tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor, affine, header


def preprocess_acv(paths_dict):
    img_t1 = nib.load(paths_dict["t1"])
    affine = img_t1.affine
    header = img_t1.header

    data = img_t1.get_fdata()
    data = robust_normalization(data)
    data = np.expand_dims(data, axis=0)

    tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor, affine, header


def _run_segmentation_inference(model, preprocess_fn, saved_paths_dict, output_path, task_type: TaskType):
    if model is None:
        raise ValueError(f"Modelo para {task_type.value} no cargado.")

    tensor_img, affine, header = preprocess_fn(saved_paths_dict)
    tensor_img = tensor_img.to(device)

    print(f"Inferencia {task_type.value}. Tensor shape: {tensor_img.shape}")

    with torch.no_grad():
        output = sliding_window_inference(inputs=tensor_img, roi_size=(64, 64, 64), sw_batch_size=4, predictor=model, overlap=0.5, mode="gaussian")
        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.5).float()
        pred_mask = pred_mask.cpu().numpy()[0, 0]

    result_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine, header)
    nib.save(result_img, output_path)

    return output_path


def run_inference_metastasis(saved_paths_dict, output_path, task_type: TaskType):
    return _run_segmentation_inference(model_mets, preprocess_metastasis, saved_paths_dict, output_path, task_type)


def run_inference_acv(saved_paths_dict, output_path, task_type: TaskType):
    return _run_segmentation_inference(model_acv, preprocess_acv, saved_paths_dict, output_path, task_type)


def run_inference_alzheimer(saved_paths_dict):
    global model_alzheimer, threshold
    if model_alzheimer is None:
        raise ValueError("Modelo Alzheimer no cargado.")

    image_path = saved_paths_dict["t1"]
    tensor_img = preprocess_alzheimer(image_path).to(device)

    with torch.no_grad():
        output = model_alzheimer(tensor_img)
        probs = F.softmax(output, dim=1)[:, 1]
        prob_value = probs.cpu().numpy()[0]
        prediction = int(prob_value >= threshold)

    return {"prediction": prediction, "probability": float(prob_value), "threshold": float(threshold)}


# ==========================================
# NUEVA LOGICA DE VISUALIZACION
# ==========================================
def create_best_slice_visualization(orig_path, pred_path, paciente_id, output_jpg_path, task_type: TaskType):
    """
    Carga el NIfTI original representativo y la predicción binaria, encuentra el mejor corte,
    y guarda un JPG renderizado analíticamente sin bloquear el Main Loop.
    """
    # 1. Carga de NIfTIs para minimizar ocupación RAM
    img_nii = nib.load(orig_path)
    pred_nii = nib.load(pred_path)

    vol = img_nii.get_fdata()
    pred = pred_nii.get_fdata()

    # Asegurarnos de que el volumen original esté en 3D
    if len(vol.shape) > 3:
        vol = vol[..., 0]

    # 2. Lógica de selección del mejor corte (eje Z típicamente indice 2)
    if np.sum(pred) > 0:
        # Sumamos píxeles a través de ejes 0 y 1 para ver densidad en cada plano Z
        z_sums = np.sum(pred, axis=(0, 1))
        best_slice_idx = np.argmax(z_sums)
        status = "Hallazgo Patológico Detectado"
    else:
        best_slice_idx = vol.shape[2] // 2
        status = "Sin Hallazgos Detectados (Corte Central)"

    # Rotar para alinear anatomicamente (nariz hacia arriba usualmente)
    slice_img = np.rot90(vol[:, :, best_slice_idx])
    slice_pred = np.rot90(pred[:, :, best_slice_idx])

    # 3. Graficación estructurada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Paciente: {paciente_id} | Tipo de Estudio: {task_type.value.upper()} | Corte Óptimo (Slice Z): {best_slice_idx} \nEstado: {status}\n",
        fontsize=14,
        fontweight="bold",
    )

    # Panel A: Original
    axes[0].imshow(slice_img, cmap="gray")
    axes[0].set_title("Corte Original Representativo", fontsize=14)
    axes[0].axis("off")

    # Panel B: Predicción en verde (Estilo de nuestro Grupo 158)
    axes[1].imshow(slice_img, cmap="gray")
    masked_pred = np.ma.masked_where(slice_pred == 0, slice_pred)
    axes[1].imshow(masked_pred, cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title("Predicción (Mascara Completa)", fontsize=14)
    axes[1].axis("off")

    # Panel C: Overlay elegante con contorno amarillo
    axes[2].imshow(slice_img, cmap="gray")
    axes[2].imshow(masked_pred, cmap="Greens", alpha=0.25)
    if np.sum(slice_pred) > 0:
        axes[2].contour(slice_pred, colors="yellow", linewidths=1.2, alpha=0.9)
    axes[2].set_title("Solapamiento Anatómico y Contorno", fontsize=14)
    axes[2].axis("off")

    # 4. Guardado seguro
    plt.tight_layout()
    plt.savefig(output_jpg_path, format="jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)  # Liberar de memoria RAM
