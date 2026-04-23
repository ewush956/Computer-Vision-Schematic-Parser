import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from model_inference.wire_detect import detect_wires
from model_inference.yolo_detection import detect_components
from schematics.schematic import SchematicParser, Schematic
from model_inference.text_ocr import process_schematic_with_yolo
from schematics.schematic_reconstructor import SchematicReconstructor
WIRE_MODEL_PATH = "models/unet_best.pth"
YOLO_MODEL_PATH = "models/yolo.pt"
OCR_MODEL_DIR = "./models/trocr-schematic-final"
OUTPUT_DIR = "./output"


def run_inference(
    image_path: str | Path,
    reconstructor: SchematicReconstructor | None = None,
    align: int | None = None,
) -> tuple[Schematic, int]:
    """Run YOLO + wire detection + filtering + alignment on ``image_path``.

    Returns the finalised ``Schematic`` plus the raw polyline count (useful
    for logging). The image's detected component XML is also written to
    ``output/<stem>.xml`` as a side-effect
    """

    reconstructor = reconstructor or SchematicReconstructor()
    image_path = Path(image_path)

    yolo_result = detect_components(
        model_path=YOLO_MODEL_PATH,
        image_path=str(image_path),
        save_annotated=True,
        save_directory=OUTPUT_DIR
    )

    schematic = SchematicParser.from_yolo_to_schematic(yolo_result)
    schematic = process_schematic_with_yolo(
        schematic, model_dir=Path(OCR_MODEL_DIR)
    )

    _cleaned, polylines = detect_wires(
        image_path, schematic, WIRE_MODEL_PATH, output_dir=OUTPUT_DIR
    )

    schematic = reconstructor.filter_by_confidence(schematic)
    schematic = reconstructor.link_text_to_components(schematic)
    schematic = reconstructor.connect_components(schematic, polylines)
    # Text components are retained so annotate_labels can render the OCR'd
    # handwriting at its original bbox. draw_components/draw_lines both skip
    # text, and connect_components already excludes it when matching wire
    # endpoints to components.
    #
    # Wire-based alignment runs first to collapse tilt-induced staircases
    # (transitively-connected components share an axis). Proximity-based
    # alignment then mops up small residual gaps for components that the
    # wire graph didn't link.
    schematic = reconstructor.align_components_by_wires(schematic)
    schematic = reconstructor.align_components(schematic, tolerance=align)

    SchematicParser.save_to_xml(schematic, f"{OUTPUT_DIR}/{image_path.stem}.xml")

    return schematic, len(polylines)


def render_schematic(
    schematic: Schematic,
    labels: bool = True,
    reconstructor: "SchematicReconstructor | None" = None,
) -> np.ndarray:
    """Render ``schematic`` to a fresh BGR canvas and return it.
    """
    reconstructor = reconstructor or SchematicReconstructor()
    canvas = reconstructor.render_canvas(schematic)
    reconstructor.draw_components(canvas, schematic)
    reconstructor.draw_lines(canvas, schematic)
    if labels:
        reconstructor.annotate_labels(canvas, schematic)
    return canvas



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Process hand-drawn circuit schematics."
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to input circuit image"
    )
    args = parser.parse_args(argv)

    img_path = args.image
    
    schematic , _line_count = run_inference(image_path=img_path)
    canvas = render_schematic(schematic)
    cv2.imwrite(f"{OUTPUT_DIR}/{img_path.stem}.png", canvas)

    
if __name__ == "__main__":
    main()
