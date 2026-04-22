import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from model_inference.text_ocr import process_schematic_with_yolo
from model_inference.wire_detect import detect_wires
from model_inference.yolo_detection import detect_components
# from model_inference.text_ocr import resolve_text_annotations
from schematics.schematic import Schematic, SchematicParser
from schematics.schematic_reconstructor import SchematicReconstructor, visualize_schematic
# from schematics.schematic_reconstructor import SchematicReconstructor, visualize_schematic

WIRE_MODEL_PATH = "models/unet_best.pth"
YOLO_MODEL_PATH = "models/yolo.pt"
OCR_MODEL_DIR = "./models/trocrSchematicFinal"
def draw_component_boxes(image_bgr: np.ndarray, schematic: Schematic) -> np.ndarray:
    output = image_bgr.copy()

    for comp in schematic.components:
        cv2.rectangle(output, (comp.xmin, comp.ymin), (comp.xmax, comp.ymax), (0, 180, 255), 2)
    return output




def draw_polylines(
    image_bgr: np.ndarray,
    polylines: list[list[tuple[int, int]]],
    random_colors: bool = True,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    output = image_bgr.copy()
    rng = np.random.default_rng(42)

    for polyline in polylines:
        if len(polyline) < 2:
            continue
       
        points = np.array(polyline, dtype=np.int32).reshape(-1, 1, 2)
        line_color = (
            tuple(int(v) for v in rng.integers(40, 256, size=3))
            if random_colors
            else color
        )
        cv2.polylines(
            output,
            [points],
            isClosed=False,
            color=line_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return output


def export_processing_steps(
    output_dir: Path,
    image_path: Path,
    schematic: Schematic,
    cleaned: np.ndarray,
    erased: np.ndarray,
    skeleton: np.ndarray,
    polylines: list[list[tuple[int, int]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    blank_canvas = np.full_like(image_bgr, 255)

    yolo_overlay = draw_component_boxes(image_bgr, schematic)
    polyline_overlay = draw_polylines(image_bgr, polylines, random_colors=True)

    final_canvas = draw_component_boxes(blank_canvas, schematic)
    final_canvas = draw_polylines(
        final_canvas,
        polylines,
        random_colors=False,
        color=(0, 0, 255),
        thickness=2,
    )

    cv2.imwrite(str(output_dir / "01_original.png"), image_bgr)
    cv2.imwrite(str(output_dir / "02_yolo_components.png"), yolo_overlay)
    cv2.imwrite(str(output_dir / "03_erased_components.png"), erased)
    cv2.imwrite(str(output_dir / "04_cleaned_wire_mask.png"), cleaned)
    cv2.imwrite(str(output_dir / "05_skeleton.png"), skeleton)
    cv2.imwrite(str(output_dir / "06_polylines_overlay.png"), polyline_overlay)
    cv2.imwrite(str(output_dir / "07_final_canvas.png"), final_canvas)



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Process hand-drawn circuit schematics."
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to input circuit image"
    )
    args = parser.parse_args(argv)

    img_path = args.image

    # Runs the yolo model and gives the component bounding boxes
    yolo_result = detect_components(YOLO_MODEL_PATH,img_path, save_annotated=True, save_directory= "./processing_steps")
    schematic = SchematicParser.from_yolo_to_schematic(yolo_result)
    schematic = process_schematic_with_yolo(schematic, OCR_MODEL_DIR)


    cleaned, erased, skeleton, polyLines = detect_wires(
         img_path,schematic, WIRE_MODEL_PATH
     )
    export_processing_steps(
        output_dir=Path("./processing_steps"),
        image_path=img_path,
        schematic=schematic,
        cleaned=cleaned,
        erased=erased,
        skeleton=skeleton,
        polylines=polyLines,
    )

    schematic_recon = SchematicReconstructor()
    schematic = schematic_recon.connect_components(schematic=schematic, polyLines=polyLines)
    visualize_schematic(schematic, "processing_steps/schematic_graph.png")

    SchematicParser.save_to_xml(schematic,"./processing_steps/out.xml")

    
if __name__ == "__main__":
    main()
