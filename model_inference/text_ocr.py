from copy import deepcopy

import cv2
from pathlib import Path
from schematics.schematic import Schematic
from model_inference.semantic_parser import SchematicTextClassifier
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import huggingface_hub.utils._validators as hf_validators



def run_ocr(crop_bgr, model_dir : Path) -> str:
    """Run TrOCR on a BGR crop and return the predicted string."""
    processor = TrOCRProcessor.from_pretrained(model_dir, local_files_only=True)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(model_dir, local_files_only=True)
    trocr_model.eval()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_schematic_with_yolo(schematic: Schematic, model_dir : Path | str) -> Schematic:
    classifier = SchematicTextClassifier()

    img = cv2.imread(schematic.image_path)

    schematic_with_text = deepcopy(schematic)

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

        # cnvert back to BGR for TrOCR
        binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

        raw_text = run_ocr(binarized_bgr, model_dir=Path(model_dir))
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
        


        



