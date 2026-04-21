import cv2
import easyocr
import xml.etree.ElementTree as ET
from semantic_parser import SchematicTextClassifier
from ultralytics import YOLO
 
 
def process_schematic_with_yolo(image_path, model_path):
    print("Loading AI Models...")

    yolo_model = YOLO(model_path) 
    
    reader = easyocr.Reader(['en'], verbose=False)
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
        
        ocr_result = reader.readtext(
            binarized,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+-µΩkMunpfFVvRr~'
)
        
        if len(ocr_result) > 0:
            raw_text = ocr_result[0][1]
            confidence = ocr_result[0][2]
            text_type = classifier.classify(raw_text)
            
            if text_type:
                print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()} (Conf: {confidence:.2f})")
            else:
                print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")
        else:
            print(f"[ NO  ] OCR failed at predicted box {xmin},{ymin}")
 
    print("-" * 50)
    print("PIPELINE COMPLETE")
    
    from pathlib import Path
    output_path = './images_and_xml/yolo_raw_output.jpg'
    results[0].save(output_path)
 

IMAGE_FILE = 'images_and_xml/C-13_D1_P4.jpg'
MODEL_FILE = './weights/best.pt'
 
process_schematic_with_yolo(IMAGE_FILE, MODEL_FILE)