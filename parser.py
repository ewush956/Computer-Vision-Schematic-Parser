import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from detectors.wire_detect import detect_wires
from detectors.yolo_detection import detect_and_export_to_xml
from schematics.schematic_reconstructor import SchematicReconstructor, visualize_schematic

WIRE_MODEL = "models/unet_best.pth"
YOLO_MODEL = "models/yolo.pt"


def draw_component_boxes(
    image_bgr: np.ndarray, component_xml: ET.ElementTree
) -> np.ndarray:
    output = image_bgr.copy()
    root = component_xml.getroot()

    for component in root.findall("component"):
        if component.get("class", "component") == "text":
            continue
        box = component.find("bounding_box")

        x0 = int(box.get("xmin"))
        y0 = int(box.get("ymin"))
        x1 = int(box.get("xmax"))
        y1 = int(box.get("ymax"))

        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 180, 255), 2)

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
    component_xml: ET.ElementTree,
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

    yolo_overlay = draw_component_boxes(image_bgr, component_xml)
    polyline_overlay = draw_polylines(image_bgr, polylines, random_colors=True)

    final_canvas = draw_component_boxes(blank_canvas, component_xml)
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

    component_xml = detect_and_export_to_xml(
        model_path=YOLO_MODEL,
        image_path=str(img_path),
        output_xml_path=f"output/{img_path.stem}.xml",
    )

    cleaned, erased, skeleton, polyLines = detect_wires(
        img_path, component_xml.getroot(), WIRE_MODEL
    )
    export_processing_steps(
        output_dir=Path("./output"),
        image_path=img_path,
        component_xml=component_xml,
        cleaned=cleaned,
        erased=erased,
        skeleton=skeleton,
        polylines=polyLines,
    )

    schematic_recon = SchematicReconstructor()
    schematic = schematic_recon.reconstruct(component_xml, polyLines, str(img_path))
    visualize_schematic(schematic)

if __name__ == "__main__":
    main()
