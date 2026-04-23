from copy import deepcopy

import cv2
from pathlib import Path
from schematics.schematic import Schematic
from model_inference.semantic_parser import SchematicTextClassifier
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import huggingface_hub.utils._validators as hf_validators



def run_ocr(crop_bgr: np.ndarray, processor: TrOCRProcessor, trocr_model: VisionEncoderDecoderModel) -> str:
    """
    Run TrOCR on a single BGR image crop and return the predicted text.

    Args:
        crop_bgr:    BGR image crop containing the text region.
        processor:   TrOCRProcessor for preprocessing the image.
        trocr_model: VisionEncoderDecoderModel for text generation.

    Returns:
        Predicted text string, or empty string if nothing was recognised.
    """
    
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_schematic_with_yolo(schematic: Schematic, model_dir : Path | str) -> Schematic:
    """
    Run OCR on all text components in the schematic and classify the results.

    Loads TrOCR once, then for each text component crops the region from the
    original image, preprocesses it (grayscale, upscale, binarize), runs OCR,
    and classifies the result. Components whose text passes classification have
    their text and text_type fields populated.

    Args:
        schematic: Schematic containing detected components and image path.
        model_dir: Path to the local TrOCR model directory.

    Returns:
        Deep copy of the schematic with text and text_type set on recognised components.
    """
    classifier = SchematicTextClassifier()
    img = cv2.imread(schematic.image_path)
    schematic_with_text = deepcopy(schematic)
    
    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
    trocr_model.eval()

    for component in schematic_with_text.components:
        if component.class_name != 'text':
            continue

        padding = 5
        ymin_p = max(0, component.ymin - padding)
        ymax_p = min(img.shape[0], component.ymax + padding)
        xmin_p = max(0, component.xmin - padding)
        xmax_p = min(img.shape[1], component.xmax + padding)

        crop = img[ymin_p:ymax_p, xmin_p:xmax_p]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binarized = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.mean(binarized)[0] < 127:
            binarized = cv2.bitwise_not(binarized)

        # convert back to BGR for TrOCR
        binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

        raw_text = run_ocr(binarized_bgr, processor, trocr_model)

        if raw_text:
            text_type = classifier.classify(raw_text)
            if text_type:
                print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()}")
                component.text = raw_text
                component.text_type = text_type
            else:
                print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")

        else:
            print(f"[ NO  ] OCR failed at predicted box {component.xmin},{component.ymin}")

    return schematic_with_text
        


        



