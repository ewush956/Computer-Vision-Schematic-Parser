import cv2
import easyocr
import xml.etree.ElementTree as ET
from model_inference.semantic_parser import SchematicTextClassifier
from schematics.schematic import Schematic
from ultralytics import YOLO
from ultralytics.engine.results import Results
from copy import deepcopy
def resolve_text_annotations(schematic: Schematic):
    resolved_schematic = deepcopy(schematic)
    reader = easyocr.Reader(['en'], verbose=False)
    classifier = SchematicTextClassifier()
    
    img = cv2.imread(schematic.image_path)    
    # schematic_with_ocr = schematic.copy()
    for comp in resolved_schematic.components:
        if comp.class_name == "text":
            continue
                
        padding = 5
        ymin_p = max(0, comp.ymin - padding)
        ymax_p = min(img.shape[0], comp.ymax + padding)
        xmin_p = max(0, comp.xmin - padding)
        xmax_p = min(img.shape[1], comp.xmax + padding)
 
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
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.+-µΩkMunpfFVvRr~')
        
        if len(ocr_result) > 0:
            raw_text = ocr_result[0][1]
            confidence = float(ocr_result[0][2])
            text_type = classifier.classify(raw_text)
            
            
            comp.text = raw_text
            comp.ocr_conf = confidence
            comp.text_type = text_type
        else:
            print(f"[ NO  ] OCR failed at predicted box {comp.xmin},{comp.ymin}")
 
    return  resolved_schematic

 

IMAGE_FILE = 'images_and_xml/C-13_D1_P4.jpg'
MODEL_FILE = './weights/best.pt'
 
# process_schematic_with_yolo(IMAGE_FILE, MODEL_FILE)








# def process_schematic_text(image_path, xml_path):
#     print("Loading Models...")

#     reader = easyocr.Reader(['en'], verbose=False)

#     classifier = SchematicTextClassifier()
    
#     print(f"Loading Image: {image_path}")
#     img = cv2.imread(image_path)
    
#     print(f"Parsing XML: {xml_path}")
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
    
#     print("-" * 50)
#     print("STARTING TEXT EXTRACTION LOOP")
#     print("-" * 50)
    
#     # Loop throughthe XML file
#     for obj in root.findall('object'):
#         name = obj.find('name').text
        
#         if name != 'text':
#             continue
            
#         # Extract Bounding Box
#         bndbox = obj.find('bndbox')
#         xmin = int(bndbox.find('xmin').text)
#         ymin = int(bndbox.find('ymin').text)
#         xmax = int(bndbox.find('xmax').text)
#         ymax = int(bndbox.find('ymax').text)
        
#         crop = img[ymin:ymax, xmin:xmax]
        
#         if crop.size == 0:
#             continue
            
#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
   
#         result = reader.readtext(scaled)

#         if len(result) > 0:
#             raw_text = result[0][1]
#             confidence = result[0][2]
            
#             text_type = classifier.classify(raw_text)
            
#             if text_type:
#                 print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()} (Conf: {confidence:.2f})")
#             else:
#                 print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")
#         else:
#             print(f"[ NO  ] OCR failed to read anything at coordinates {xmin},{ymin}")

#     print("-" * 50)
#     print("PIPELINE COMPLETE")


# IMAGE_FILE ='images_and_xml\C-13_D1_P4.jpg'
# XML_FILE = 'images_and_xml\C-13_D1_P4.xml'

# process_schematic_text(IMAGE_FILE, XML_FILE)