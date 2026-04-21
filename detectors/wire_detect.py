from pathlib import Path
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
import segmentation_models_pytorch as smp

from swig import trace_skeleton

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def load_model(weights_path: str | Path) -> torch.nn.Module:
    """
    Loads a segmentation model with the weights from the weights file.

    Args:
        weights_path: Path to the saved weights dictionary file.

    Returns:
        model (torch.nn.Module): ResNet-50 UNet in eval mode.
    """
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)
    state = torch.load(str(weights_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def remove_small_blobs(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Remove connected components whose area is below min_area pixels. Used for removing small specs of
    noise in the  mask.

    Args:
        mask:     Binary or grayscale uint8 mask to filter.
        min_area: Minimum pixel area for a component to be kept.

    Returns:
        cleaned (np.ndarray): uint8 mask with small blobs zeroed out.
    """
    binary = (mask > 0).astype("uint8")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):  # skip label 0 = background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def erase_components(
    mask: np.ndarray,
    annotations: ET.Element,
    padding: int = 2,
) -> np.ndarray:
    """
    Zero out bounding-box regions so component pixels. Must be done prior to removing small blobs and after running inference on the
    segmentation model

    Args:
        mask:        Binary wire mask to modify.
        annotations: Root XML element containing <bounding_box>.
        padding:     Extra pixels to expand each box on all sides.

    Returns:
        out (np.ndarray): Copy of mask with all component bounding-box regions set to 0.
    """
    out = mask.copy()
    height, width = mask.shape[:2]
    for box in annotations.iter("bounding_box"):
        x0 = max(0, int(box.get("xmin")) - padding)
        y0 = max(0, int(box.get("ymin")) - padding)
        x1 = min(width, int(box.get("xmax")) + padding)
        y1 = min(height, int(box.get("ymax")) + padding)
        out[y0:y1, x0:x1] = 0

    return out


def sliding_window_inference(
    img: np.ndarray,
    model: torch.nn.Module,
    patch: int = 256,
    stride: int = 180,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run segmentation inference over a full image using overlapping patches.

    Splits the image into overlapping patches, runs the model on each, then
    averages overlapping predictions back into a full-resolution binary mask.

    Args:
        img (list[uint8]):    RGB image as a uint8 numpy array.
        model(torch.nn):      Segmentation model in eval mode.
        patch (int):      Patch size in pixels (both height and width).
        stride (int):     Step size between patches.
        batch_size (int): Number of patches to process per forward pass.
        threshold (float):  Sigmoid probability above which a pixel is classed as drawing, rather than background paper.

    Returns:
        mask (np.ndarray): uint8 binary mask with drawn pixels set to 255.
    """
    infer_tf = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    h, w = img.shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    ys = list(range(0, h - patch, stride)) + [max(h - patch, 0)]
    xs = list(range(0, w - patch, stride)) + [max(w - patch, 0)]

    coords = [(y, x) for y in ys for x in xs]
    crops = [
        cv2.resize(img[y : y + patch, x : x + patch], (patch, patch)) for y, x in coords
    ]
    tensors = torch.stack([infer_tf(image=c)["image"] for c in crops])

    preds_all = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i : i + batch_size].to(DEVICE)
            preds_all.append(torch.sigmoid(model(batch)).squeeze(1).float().cpu())
    preds_all = torch.cat(preds_all).numpy()  # (N, H, W)

    for (y, x), pred in zip(coords, preds_all):
        pred_r = cv2.resize(pred, (patch, patch))
        y_end = min(y + patch, h)
        x_end = min(x + patch, w)
        ph = y_end - y
        pw = x_end - x
        accum[y:y_end, x:x_end] += pred_r[:ph, :pw]
        count[y:y_end, x:x_end] += 1

    prob_map = accum / np.maximum(count, 1)
    return ((prob_map > threshold) * 255).astype(np.uint8)


def detect_wires(
    image_path: str | Path,
    annotations: ET.Element,
    model_weights_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Run wire segmentation on an image and return the cleaned binary wire mask.

    Args:
        image_path (str | Path):          Path to the input schematic image.
        annotations (Et.Element):         Root XML element containing component bounding boxes.
        model_weights_path (str | Path):  Path to the segmentation model checkpoint.

    Returns:
        cleaned (np.ndarray):   Binary wire mask after blob removal.
        erased (np.ndarray):    Binary wire mask after component regions are zeroed out.
        skeleton (np.ndarray):  Bool skeleton array from skeletonize.
        polys list[list[tuple]]:     List of polylines, each a list of points tracing a wire.
    """
    model = load_model(model_weights_path)

    img = cv2.imread(
        str(image_path),
    )
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    # Image needs to converted to RGB as Open cv returns it in  BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Doing the actual inference
    mask = sliding_window_inference(img_rgb, model)

    # Removing bounding boxes
    erased = erase_components(mask, annotations)

    # Removing small patches
    cleaned = remove_small_blobs(erased, min_area=20)

    # Skeletonizing the image
    skeleton = skeletonize(cleaned > 0)
    skeletonized = skeleton.astype("uint8") * 255

    # This is where we use extract the wires in the image as poly lines
    polys = trace_skeleton.from_numpy(skeletonized.copy())

    return cleaned, erased, skeletonized, polys
