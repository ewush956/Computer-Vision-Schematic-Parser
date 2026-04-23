"""Batch-reconstruct every image in ``demo_diagrams/`` into side-by-side PNGs.

For each input image two outputs are produced under ``output/demo/``:

    <stem>_with_labels.png     original | reconstruction (labels on)
    <stem>_no_labels.png       original | reconstruction (labels off)

Inference is run once per image; the schematic is rendered twice (labelled
and unlabelled), so this is roughly the cost of a single labelled batch.

Usage
-----
    python demo.py                         # processes demo_diagrams/*
    python demo.py --input other_dir       # any directory of .png/.jpg/.jpeg
    python demo.py --output output/other   # custom output directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from schematics.schematic_reconstructor import SchematicReconstructor

from parser import   render_schematic, run_inference
DIVIDER_PX = 6           # width of the separator strip between the two panels
DIVIDER_COLOR = (220, 220, 220)  # BGR light-gray
HEADER_PX = 48           # height of the title banner above each panel
HEADER_BG = (245, 245, 245)
HEADER_FG = (40, 40, 40)


def _load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _match_heights(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize ``right`` to match ``left``'s height, preserving aspect ratio.

    Inputs usually share dimensions (the reconstructor canvas is sized to the
    schematic's ``width × height``, which in turn comes from YOLO's original
    shape), but padding on either image could still drift — this keeps the
    stack safe.
    """
    h_left = left.shape[0]
    h_right = right.shape[0]
    if h_left == h_right:
        return left, right
    scale = h_left / h_right
    new_w = max(1, int(round(right.shape[1] * scale)))
    resized = cv2.resize(right, (new_w, h_left), interpolation=cv2.INTER_AREA)
    return left, resized


def _banner(width: int, text: str) -> np.ndarray:
    """Header strip with ``text`` centered horizontally."""
    banner = np.full((HEADER_PX, width, 3), HEADER_BG, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, (width - tw) // 2)
    y = (HEADER_PX + th) // 2
    cv2.putText(banner, text, (x, y), font, scale, HEADER_FG, thickness, cv2.LINE_AA)
    return banner


def side_by_side(
    original: np.ndarray,
    reconstruction: np.ndarray,
    titles: tuple[str, str] = ("Original", "Reconstruction"),
) -> np.ndarray:
    """Stack ``original`` and ``reconstruction`` horizontally with a divider
    and a title banner over each panel."""
    left, right = _match_heights(original, reconstruction)
    divider = np.full((left.shape[0], DIVIDER_PX, 3), DIVIDER_COLOR, dtype=np.uint8)
    body = cv2.hconcat([left, divider, right])

    left_banner = _banner(left.shape[1], titles[0])
    divider_banner = np.full((HEADER_PX, DIVIDER_PX, 3), HEADER_BG, dtype=np.uint8)
    right_banner = _banner(right.shape[1], titles[1])
    header = cv2.hconcat([left_banner, divider_banner, right_banner])

    return cv2.vconcat([header, body])


def process_image(
    image_path: Path,
    output_dir: Path,
    reconstructor: SchematicReconstructor,
    align: int | None,
) -> None:
    schematic, n_polys = run_inference(
        image_path, reconstructor=reconstructor, align=align
    )
    original = _load_image_bgr(image_path)

    with_labels = render_schematic(schematic, labels=True, reconstructor=reconstructor)
    no_labels = render_schematic(schematic, labels=False, reconstructor=reconstructor)

    labelled_out = output_dir / f"{image_path.stem}_with_labels.png"
    unlabelled_out = output_dir / f"{image_path.stem}_no_labels.png"

    cv2.imwrite(
        str(labelled_out),
        side_by_side(original, with_labels, ("Original", "Reconstruction (labels)")),
    )
    cv2.imwrite(
        str(unlabelled_out),
        side_by_side(original, no_labels, ("Original", "Reconstruction")),
    )

    print(
        f"{image_path.name}: polylines={n_polys}, "
        f"components={len(schematic.components)} "
        f"-> {labelled_out.name}, {unlabelled_out.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side demo renders for a directory of schematic images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("demo_diagrams"),
        help="Directory of input images (default: demo_diagrams).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/demo"),
        help="Directory where the paired PNGs are written (default: output/demo).",
    )
    parser.add_argument(
        "--align",
        type=int,
        default=None,
        help="Alignment tolerance in px; passes through to align_components.",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")

    images = sorted(
        p
        for p in args.input.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not images:
        raise SystemExit(f"No .png/.jpg/.jpeg images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    reconstructor = SchematicReconstructor()

    for image_path in images:
        process_image(image_path, args.output, reconstructor, args.align)


if __name__ == "__main__":
    main()
