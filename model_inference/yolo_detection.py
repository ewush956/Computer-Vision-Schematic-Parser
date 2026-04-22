from pathlib import Path

from ultralytics import YOLO
import xml.etree.ElementTree as ET
from ultralytics.engine.results import Results

def detect_components(model_path: str,
           image_path: str,
           conf: float = 0.25,
           save_annotated: bool = False,
           save_directory: str = "./processing_steps"
           ) -> Results:
    """
    Runs YOLO inference on a schematic image and returns the raw results.
    Args:
        image_path (str): Path to the input hand-drawn schematic.
        model_path (str): Path to the trained YOLO weights.
        conf (float): Confidence threshold for detections.
        save_annotated (bool): If True, YOLO will save a visual plot of the detections.
        save_directory (str): Where to save the annotated plot
    Returns:
        Results: Raw YOLO results object.
    """
    model = YOLO(model_path)
    result = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
    )[0]

    if save_annotated:
        Path("output").mkdir(exist_ok=True)
    result.save(f"{save_directory}/yolo_output.png", line_width=1, labels=True, conf=False)
    return result