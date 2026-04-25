# Schematic Parser

This project takes an image of a hand-drawn or scanned electrical schematic and tries to reconstruct a cleaner digital version of it. The pipeline detects schematic symbols, reads nearby handwritten labels, detects wires, links wires to components, and exports both an XML representation and rendered output images.

## Requirements

- Python 3.13
- gcc (required to compile the skeleton tracing library)
- Other python packages are in requirements.txt

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile the wire tracing library:

```bash
cd swig && ./compile.sh && cd ..
```

## Usage

Run the pipeline on a single image:

```bash
python parser.py --image demo_diagrams/C73_D2_P1.jpg
```

Run the batch demo on all images in `demo_diagrams/`:

```bash
python demo.py
```

**Outputs from `parser.py`:**

```
output/<image_stem>.png       # reconstructed schematic
output/<image_stem>.xml       # component and wire description
output/yolo_output.png        # YOLO detection overlay
output/wire_mask.png          # segmented wire mask
output/polylines.png          # traced wire polylines
```

**Outputs from `demo.py`:**

```
output/demo/<image_stem>_with_labels.png
output/demo/<image_stem>_no_labels.png
```

no## Models

The following model files are required and are included via Git LFS:

| File                            | Purpose                          |
| ------------------------------- | -------------------------------- |
| `models/yolo.pt`                | Component and text-box detection |
| `models/unet_best.pth`          | Wire pixel segmentation          |
| `models/trocr-schematic-final/` | Text recognition                 |

## Pipeline

1. **YOLO detection** (`model_inference/yolo_detection.py`) — detects component bounding boxes and text regions
2. **OCR** (`model_inference/text_ocr.py`) — crops text regions, preprocesses, and runs TrOCR
3. **Text classification** (`model_inference/semantic_parser.py`) — classifies OCR output as `reference`, `value`, `pin_label`, or `power_net`, filtering out noise
4. **Wire segmentation** (`model_inference/wire_detect.py`) — runs U-Net to produce a binary wire mask and traces polylines
5. **Reconstruction** (`schematics/schematic_reconstructor.py`) — links text to components, connects wires to components, aligns components, and renders the final output

## Repository Structure

```
parser.py                   Single-image pipeline entry point
demo.py                     Batch demo runner
model_inference/            Model inference code
    yolo_detection.py
    text_ocr.py
    semantic_parser.py
    wire_detect.py
schematics/                 Schematic representation and rendering
    schematic.py            Dataclasses and XML writer
    schematic_reconstructor.py
    routing.py              Orthogonal wire routing helpers
assets/
    Symbols.svg             Source symbol sheet
    symbols/                Exported PNG symbols used for rendering
models/                     Trained model weights
swig/                       C skeleton tracing library and SWIG wrapper
training_notebooks/         YOLO and segmentation training notebooks
demo_diagrams/              Example inputs for the batch demo
tests/
    images/                 Test inputs
    outputs/                Example reconstruction outputs
scripts/                    Helper scirpts (currently just the SVG extraction)
```

## XML Output Format

```xml
<?xml version='1.0' encoding='utf-8'?>
<schematic width="1536" height="1152" image_path="demo_diagrams/example.jpg">
  <component id="0" class="resistor" yolo_conf="0.87"
             xmin="330" ymin="392" xmax="455" ymax="450" />
  <component id="1" class="text" yolo_conf="0.83"
             xmin="59" ymin="145" xmax="123" ymax="203"
             text="IN" text_type="pin_label" />
  <line id="2" status="connected" from_id="0" to_id="3">
    <point x="300" y="413" />
    <point x="250" y="413" />
  </line>
</schematic>
```

Wire statuses: `connected` (both ends match a component), `dangling` (one end matches), `orphan` (no match).

## Credits

- [skeleton-tracing](https://github.com/LingDong-/skeleton-tracing) — C/SWIG library for converting binary wire masks into polylines
- [Training Dataset](https://www.kaggle.com/datasets/johannesbayer/cghd1152)
