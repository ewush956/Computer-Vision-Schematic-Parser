import cv2
from pathlib import Path
from semantic_parser import SchematicTextClassifier
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import huggingface_hub.utils._validators as hf_validators
import os

# ── Windows Path Compatibility Fix ───────────────────────────
# huggingface_hub v1.11+ incorrectly validates local Windows file paths
# as HuggingFace repository IDs. We monkey-patch the validator to a no-op
# so local paths are accepted without network validation.
# Note: Mac/Linux users do not need this fix as their paths pass validation.
hf_validators.validate_repo_id = lambda repo_id: None

# ── Model Loading ─────────────────────────────────────────────
# TrOCR (Transformer-based OCR) is a Vision Encoder-Decoder model from Microsoft.
# It uses a ViT (Vision Transformer) as the image encoder and a RoBERTa-based
# transformer as the text decoder. We load our fine-tuned weights from the
# local directory rather than downloading from HuggingFace Hub.
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(script_dir, "trocrSchematicFinal").replace("\\", "/")

# The processor handles two things:
#   1. Image preprocessing — resizes and normalises the crop for the ViT encoder
#   2. Tokenization — converts predicted token IDs back to text strings
processor = TrOCRProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

# The VisionEncoderDecoderModel is the full TrOCR architecture:
#   Encoder: ViT processes the image into feature embeddings
#   Decoder: RoBERTa autoregressively generates the output text token by token
trocr_model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR, local_files_only=True)

# eval() disables dropout and batch normalisation layers used during training
# — required for deterministic inference results
trocr_model.eval()


def run_ocr(crop_bgr) -> str:
    """
    Runs TrOCR inference on a single BGR image crop.

    Pipeline:
      BGR → RGB colour space conversion (ViT expects RGB)
      → PIL Image (required format for the TrOCR processor)
      → pixel_values tensor (normalised for ViT input)
      → autoregressive decoding (beam search generates token sequence)
      → decoded string (token IDs mapped back to text)
    """
    # Colour space conversion: OpenCV uses BGR, TrOCR expects RGB
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    # PIL Image is required by the TrOCR feature extractor
    pil_img = Image.fromarray(rgb).convert("RGB")

    # Feature extraction: resizes to 384x384 and normalises pixel values
    # to the distribution the ViT encoder was trained on
    pixel_values = processor(pil_img, return_tensors="pt").pixel_values

    # Autoregressive decoding: the decoder generates one token at a time
    # using beam search until it produces an end-of-sequence token
    generated_ids = trocr_model.generate(pixel_values)

    # Detokenization: converts token ID sequence back to a human-readable string
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def process_schematic_with_yolo(image_path, model_path):
    # ── Object Detection — YOLO Inference ────────────────────
    # YOLO (You Only Look Once) is a single-pass convolutional object detector.
    # It divides the image into a grid and predicts bounding boxes and class
    # probabilities for each cell simultaneously, making it very fast.
    yolo_model = YOLO(model_path)

    # Semantic classifier — categorises OCR output into schematic types
    # (reference, value, power_net, pin_label) using regex and hardcoded sets
    classifier = SchematicTextClassifier()

    print(f"Loading Image: {image_path}")
    img = cv2.imread(image_path)

    print("-" * 50)
    print("STARTING YOLO INFERENCE")
    print("-" * 50)

    # conf=0.5 is the confidence threshold — YOLO only returns detections
    # it is at least 50% confident about, filtering out weak predictions
    results = yolo_model(img, conf=0.5)

    # Iterate over every bounding box YOLO detected in the image
    for box in results[0].boxes:

        # YOLO returns integer class IDs — map back to human-readable class name
        class_id = int(box.cls[0].item())
        class_name = yolo_model.names[class_id]

        # We only run OCR on regions YOLO classified as 'text'
        # All other component types (resistor, capacitor etc.) are ignored here
        if class_name != 'text':
            continue

        # xyxy format: [xmin, ymin, xmax, ymax] in absolute pixel coordinates
        coords = box.xyxy[0].cpu().numpy()
        xmin, ymin, xmax, ymax = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

        # ── Bounding Box Padding ──────────────────────────────
        # Expand the crop by 5px in each direction to ensure we don't clip
        # characters at the edge of the bounding box. Clamped to image bounds
        # to prevent out-of-range array indexing.
        padding = 5
        ymin_p = max(0, ymin - padding)
        ymax_p = min(img.shape[0], ymax + padding)
        xmin_p = max(0, xmin - padding)
        xmax_p = min(img.shape[1], xmax + padding)

        # NumPy array slicing — crops the region of interest from the full image
        crop = img[ymin_p:ymax_p, xmin_p:xmax_p]
        if crop.size == 0:
            continue

        # ── Image Preprocessing ───────────────────────────────
        # Grayscale conversion: reduces 3-channel BGR to single luminance channel.
        # Removes colour information irrelevant to text recognition.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Bicubic upscaling 3x: enlarges the crop so individual character strokes
        # occupy more pixels. INTER_CUBIC uses a 4x4 pixel neighbourhood for
        # smooth interpolation, better than nearest-neighbour for text.
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Otsu's Binarization: automatically finds the optimal global threshold
        # that separates foreground (ink strokes) from background (paper).
        # Otsu's method minimises intra-class variance between the two pixel groups.
        # THRESH_BINARY + THRESH_OTSU means: pixels above threshold → 255 (white),
        # pixels below → 0 (black), with threshold chosen automatically.
        _, binarized = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inversion check: Otsu may produce white text on black background
        # if the image has more dark pixels than light. We check the mean
        # brightness — if below 127 the image is inverted and we flip it
        # so text is always black on white (expected by TrOCR).
        if cv2.mean(binarized)[0] < 127:
            binarized = cv2.bitwise_not(binarized)

        # Grayscale → BGR: TrOCR's ViT encoder expects a 3-channel RGB image.
        # cvtColor replicates the single channel across all 3 colour channels.
        binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

        # ── OCR + Semantic Classification ─────────────────────
        raw_text = run_ocr(binarized_bgr)

        if raw_text:
            # Pass OCR output to the semantic classifier which uses regex
            # patterns and hardcoded sets to determine the schematic role
            # of the detected text string
            text_type = classifier.classify(raw_text)
            if text_type:
                print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()}")
            else:
                print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")
        else:
            print(f"[ NO  ] OCR failed at predicted box {xmin},{ymin}")

    print("-" * 50)

    # Save YOLO's raw detection visualisation to disk (bounding boxes drawn).
    # Only saves if the file doesn't already exist to avoid overwriting results.
    output_path = './images_and_xml/yolo_raw_output.jpg'
    if not Path(output_path).exists():
        results[0].save(output_path)


# ── Entry Point ───────────────────────────────────────────────
IMAGE_FILE = 'images_and_xml/C-13_D1_P4.jpg'
MODEL_FILE = './weights/best.pt'

process_schematic_with_yolo(IMAGE_FILE, MODEL_FILE)