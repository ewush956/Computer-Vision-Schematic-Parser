# Computer Vision Schematic Parser

Computer vision pipeline for parsing hand-drawn electrical schematics into a structured XML representation. The current workflow uses YOLO-based object detection to identify schematic components from images, then converts detections into XML for downstream processing.

## Project Scope

This project focuses on hand-drawn or scanned electrical schematics rather than facility site diagrams. The goal is to detect electrical symbols from schematic images and export those detections in a machine-readable XML format.

Current emphasis:

- Computer vision for electrical symbol detection
- YOLO training and inference workflow
- XML generation from detected components
- Support for hand-drawn schematic imagery

## Current Pipeline

The workflow in `yolo_pipeline.ipynb` currently covers:

1. Loading and inspecting the electrical schematic dataset
2. Converting annotations into YOLO training format
3. Training a YOLO model for component detection
4. Running inference on schematic images
5. Exporting detections to XML
6. Exporting the trained model to ONNX for downstream inference use

## Expected Input

- Electrical schematic images
- Hand-drawn, scanned, or otherwise non-natural-image diagram content
- Primarily single-page images

## Current Output

The notebook currently produces an XML structure with image metadata and detected components:

```xml
<?xml version="1.0" ?>
<schematic image="example.jpg" width="1296" height="972">
  <component id="0" class="transformer" confidence="0.9263">
    <bounding_box xmin="418" ymin="311" xmax="684" ymax="462"/>
  </component>
  <component id="1" class="resistor" confidence="0.9012">
    <bounding_box xmin="409" ymin="480" xmax="472" ymax="579"/>
  </component>
</schematic>
```

## Objectives

- Detect electrical components in hand-drawn schematics
- Build a repeatable YOLO-based training and inference pipeline
- Generate structured XML from model detections
- Prepare the pipeline for downstream schematic digitization tasks

## XML Reconstruction Module

`schematic_reconstructor.py` provides the data models and interface for consuming the XML output and reconstructing a visual representation of the schematic.

### Data Models

- `BoundingBox` — absolute pixel coordinates of a detected component (xmin, ymin, xmax, ymax)
- `Component` — a single detection with id, class name, confidence, bounding box, and an optional linked text label
- `Line` — a directed connection between two components, identified by their ids
- `Schematic` — the full parsed document: image metadata, list of components, and list of lines

### SchematicReconstructor

Consumes the XML produced by the pipeline and handles the full reconstruction workflow:

1. Load and parse the XML into the data models
2. Filter detections by confidence and class
3. Link nearby `"text"` detections to their closest component
4. Infer connections between components and build `Line` objects
5. Render components and connections onto a canvas
6. Annotate with labels and export the result

## Limitations

- Output is currently component-level XML, not a full netlist or circuit graph
- Detection quality depends on annotation quality and class balance
- Input assumptions are still biased toward clean, single-page images
- Text understanding and connection reasoning are not yet complete

## TODO

- Clean up the dataset and verify label consistency for electrical component classes
- Improve detection quality on hand-drawn and noisy schematic images
- Add post-processing for wires, junctions, and connectivity inference
- Expand XML schema beyond bounding boxes to support richer schematic structure
- Package inference into a reusable script or API outside the notebook

## Deliverables

- Trained YOLO model weights
- Notebook-based training and inference pipeline
- XML export for detected schematic components
- ONNX export for deployment experiments
