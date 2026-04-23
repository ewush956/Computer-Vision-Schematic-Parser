from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from schematics.schematic import Schematic
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
        weights_path: Path to the saved state dictionary (.pth file).

    Returns:
        model: UNet in eval mode.
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


def run_inference(model: torch.nn.Module, image_path: str | Path) -> np.ndarray:
    """
    Run wire segmentation inference on a single image.

    Args:
        model:      UNet model in eval mode returned by load_model.
        image_path: Path to the input schematic image.

    Returns:
        Binary uint8 mask with wire pixels set to 255 and background to 0.
    """
    infer_tf = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        tensor = infer_tf(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        pred = model(tensor)
    return (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255




def erase_components(
    mask: np.ndarray,
    schematic: Schematic,
    padding: int = 2,
) -> np.ndarray:
    """
    Zero out bounding-box regions on component pixels. Must be done prior to removing small blobs and after running inference on the
    segmentation model

    Args:
        mask:      Binary wire mask to modify.
        schematic: Schematic whose components define the regions to erase.
        padding:   Extra pixels to expand each box on all sides.

    Returns:
        out (np.ndarray): Copy of mask with all component bounding-box regions set to 0.
    """
    out = mask.copy()
    height, width = mask.shape[:2]
    for comp in schematic.components:
        x0 = max(0, comp.xmin - padding)
        y0 = max(0, comp.ymin - padding)
        x1 = min(width, comp.xmax  + padding)
        y1 = min(height, comp.ymax + padding)
        out[y0:y1, x0:x1] = 0

    return out


def detect_wires(
    image_path: str | Path,
    schematic : Schematic,
    model_weights_path: str | Path, output_dir : None | Path ) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    """
    Run wire segmentation on an image and return the binary wire mask, and polylines.

    Args:
        image_path (str | Path):          Path to the input schematic image.
        schematic (Schematic):            Schematic containing component bounding boxes to erase.
        model_weights_path (str | Path):  Path to the segmentation model weights file.
        output_dir:                       If provided, saves the wire mask and polyline overlay
                                          as PNG files to this directory.

    Returns:
        mask (np.ndarray):   Binary wire mask inferred from the model
        polys (list[list[tuple[int, int]]]) :     List of polylines, each a list of points tracing a wire.
    """
    model = load_model(model_weights_path)
    mask = run_inference(model, image_path=image_path)

    erased = erase_components(mask, schematic)

    polys = trace_skeleton.from_numpy(erased.copy())

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "wire_mask.png"), mask)
        img = cv2.imread(str(image_path))
        rng = np.random.default_rng()
        for poly in polys:
            if len(poly) < 2:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2) # CV expects this shape
            color = tuple(int(v) for v in rng.integers(40, 256, size=3))
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(str(output_dir / "polylines.png"), img)

    return mask, polys
