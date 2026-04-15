import cv2
import easyocr
import xml.etree.ElementTree as ET
from semantic_parser import SchematicTextClassifier


def process_schematic_text(image_path, xml_path):
    print("Loading AI Models...")
    reader = easyocr.Reader(['en'], verbose=False)
    classifier = SchematicTextClassifier()
    
    print(f"Loading Image: {image_path}")
    img = cv2.imread(image_path)
    
    print(f"Parsing XML: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    print("-" * 50)
    print("STARTING TEXT EXTRACTION LOOP")
    print("-" * 50)
    
    # Loop throughthe XML file
    for obj in root.findall('object'):
        name = obj.find('name').text
        
        if name != 'text':
            continue
            
        # Extract Bounding Box
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        crop = img[ymin:ymax, xmin:xmax]
        
        if crop.size == 0:
            continue
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
   
        result = reader.readtext(scaled)

        if len(result) > 0:
            raw_text = result[0][1]
            confidence = result[0][2]
            
            text_type = classifier.classify(raw_text)
            
            if text_type:
                print(f"[ YES ] Kept '{raw_text}' -> Classified as: {text_type.upper()} (Conf: {confidence:.2f})")
            else:
                print(f"[ NO  ] Trashed '{raw_text}' -> Noise / Math Equation")
        else:
            print(f"[ NO  ] OCR failed to read anything at coordinates {xmin},{ymin}")

    print("-" * 50)
    print("PIPELINE COMPLETE")


IMAGE_FILE ='images_and_xml\C-13_D1_P4.jpg'
XML_FILE = 'images_and_xml\C-13_D1_P4.xml'

process_schematic_text(IMAGE_FILE, XML_FILE)