import cv2
from pathlib import Path
from semantic_parser import SchematicTextClassifier
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import huggingface_hub.utils._validators as hf_validators
import os

#MODEL_DIR = Path(__file__).parent / 'trocr-schematic-final'
# processor = TrOCRProcessor.from_pretrained('./trocr-schematic-final')
# trocr_model = VisionEncoderDecoderModel.from_pretrained('./trocr-schematic-final')

script_dir = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(script_dir, "trocrSchematicFinal")

# MODEL_DIR = Path(script_dir) / "trocrSchematicFinal"
# model_uri = MODEL_DIR.as_uri()
# processor = TrOCRProcessor.from_pretrained(model_uri, local_files_only=True)
# trocr_model = VisionEncoderDecoderModel.from_pretrained(model_uri, local_files_only=True)

# processor = TrOCRProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
# trocr_model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR, local_files_only=True)

hf_validators.validate_repo_id = lambda repo_id: None

script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, "trocrSchematicFinal")

processor = TrOCRProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
trocr_model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR, local_files_only=True)






trocr_model.eval()



def run_ocr(crop_bgr) -> str:
    """Run TrOCR on a BGR crop and return the predicted string."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGB")
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_schematic_with_yolo(image_path, model_path):
    yolo_model = YOLO(model_path)
    classifier = SchematicTextClassifier()

    print(f"Loading Image: {image_path}")
    img = cv2.imread(image_path)

    print("-" * 50)
    print("STARTING YOLO INFERENCE")
    print("-" * 50)

    results = yolo_model(img, conf=0.5)

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = yolo_model.names[class_id]

        if class_name != 'text':
            continue

        coords = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        padding = 5
        ymin_p = max(0, ymin - padding)
        ymax_p = min(img.shape[0], ymax + padding)
        xmin_p = max(0, xmin - padding)
        xmax_p = min(img.shape[1], xmax + padding)

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

        raw_text = run_ocr(binarized_bgr)

        if raw_text:
            text_type = classifier.classify(raw_text)
            if text_type:
                print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()}")
            else:
                print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")
        else:
            print(f"[ NO  ] OCR failed at predicted box {xmin},{ymin}")

    print("-" * 50)


    output_path = './images_and_xml/yolo_raw_output.jpg'
    if not Path(output_path).exists():
        results[0].save(output_path)

IMAGE_FILE = 'images_and_xml/C159_D2_P1.jpg'
MODEL_FILE = './weights/best.pt'

process_schematic_with_yolo(IMAGE_FILE, MODEL_FILE)