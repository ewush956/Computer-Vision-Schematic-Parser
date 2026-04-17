from pathlib import Path

from ultralytics import YOLO
import xml.etree.ElementTree as ET
from ultralytics.engine.results import Results

def run_detection(model_path: str, image_path: str, conf: float, save_result: bool) ->Results:
    """
    Loads a YOLO model and performs object detection on a single image.
 
    Args:
        model_path (str): Path to the saved YOLO .pt model file.
        image_path (str): Path to the input image for inference.
        conf (float): Confidence threshold 
        save_result (bool): Whether to save the image with bounding boxes and labels annotated

    Returns:
        results (ultralytics.engine.results.Results): The first result object containing 
            detected boxes, names, and original image dimensions.
    """
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
        save=save_result,
        line_width=1,
        show_labels=True,
        show_conf=False,
    )
    

    return results[0]


def write_schematic_xml(result: Results, input_image_path:str, output_xml_path: str) -> None:
    """
    Converts YOLO detection results into a custom 'schematic' XML format.

    Args:
        result (ultralytics.engine.results.Results): The result object from YOLO.
        output_xml_path (str): The file path where the generated XML will be saved.
    """
    boxes = result.boxes
    h, w = result.orig_shape
    class_names = result.names
    root = ET.Element(
          "schematic",
          {
              "width": str(w),
              "height": str(h),
              "image_path": input_image_path

          },
      )
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])

        component = ET.SubElement(
            root,
            "component",
            {
                "id": str(i),
                "class": class_names[cls_id],
                "conf": str(conf)
            }
        )
        ET.SubElement(
            component,
            "bounding_box",
            {
                "xmin": str(int(round(x1))),
                "ymin": str(int(round(y1))),
                "xmax": str(int(round(x2))),
                "ymax": str(int(round(y2))),
            },
        )

        
    
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0) # for pretty printing on seperate line, Et does not print xml on seperate lines by default
    output_path = Path(output_xml_path)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    

def main():
    # this main is just for testing.
    result = run_detection(
        "./yolo-models/best.pt", "archive-2/drafter_2/images/C18_D1_P2.jpg", 0.5, False
    )
    write_schematic_xml(result, "archive-2/drafter_2/images/C18_D1_P2.jpg", "test.xml")


if __name__ == "__main__":
    main()
