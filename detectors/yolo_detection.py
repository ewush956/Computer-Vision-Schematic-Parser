from pathlib import Path

from ultralytics import YOLO
import xml.etree.ElementTree as ET
from ultralytics.engine.results import Results


def detect_and_export_to_xml(
    model_path: str,
    image_path: str,
    output_xml_path: str,
    conf: float = 0.25,
    save_annotated: bool = False,
) -> ET.ElementTree:
    """
    Performs component detection saves results to XML. This returns the xml schematic and also saves it as file.

    Args:
        model_path (str): Path to the trained YOLO weights.
        image_path (str): Path to the input hand-drawn schematic.
        output_xml_path (str): Path where the generated XML will be saved.
        conf (float): Confidence threshold for detections.
        save_annotated (bool): If True, YOLO will save a visual plot of the detections.

    Returns:
        ET.ElementTree: The generated XML tree object.
    """
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        verbose=False,
        save=save_annotated,
        line_width=1,
        show_labels=True,
        show_conf=False,
    )

    result = results[
        0
    ]  # Yolo handles the output weirdly, have to index into to it get the bounding boxes
    boxes = result.boxes
    h, w = result.orig_shape
    class_names = result.names
    root = ET.Element(
        "schematic",
        {"width": str(w), "height": str(h), "image_path": image_path},
    )
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])

        component = ET.SubElement(
            root,
            "component",
            {"id": str(i), "class": class_names[cls_id], "conf": str(conf)},
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
    ET.indent(
        tree, space="  ", level=0
    )  # for pretty printing on seperate line, Et does not print xml on seperate lines by default
    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return tree


def export_schematic_to_xml(
    result: Results, input_image_path: str, output_xml_path: str
) -> ET.ElementTree:
    """
    Serializes YOLO detection results into a standardized XML format. Saves the Xml in a file as well.

    Args:
        result (Results): The YOLO inference result object.
        input_image_path (str): Path to the source drawing (used as metadata in XML).
        output_xml_path (str): Destination file path for the generated xml file.

    Returns:
        ET.ElementTree: The formatted XML tree object.
    """

