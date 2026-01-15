# Computer Vision Schematic Parser
Proof-of-concept system that converts facility site schematics into structured directed graphs using computer vision. Detects equipment symbols and colored arrows, infers flow direction and relationship type, and outputs a JSON representation via Python API. Designed for facility site diagrams and limited equipment classes.

Facility site schematics encode equipment and flow relationships using symbols, arrows, and color-coded connections. This project proposes a proof-of-concept system that converts a schematic image into a directed graph describing equipment nodes and typed relationships.

## Objectives

- Detect equipment symbols in facility schematics.
- Detect arrows and infer arrow direction.
- Classify arrow colors into relationship types.
- Convert detections into a structured directed graph.
- Expose results through an API interface.

## Learning Outcomes

- Apply object detection models to non-natural image domains. Currently looking at YOLO, but need to investigate any licensing issues.
- Implement feature extraction using classical computer vision.
- Design logic that maps visual features into graphs.
- Build and deploy an ML inference API.
- Analyze system limitations in real-world engineering diagrams.

## Scope and Limitations

- Input images limited to PNG, JPG, or PDF (maybe).
- Schematics must be clean and high-contrast.
- Equipment types restricted to a predefined set.
- Relationship types restricted to a limited color palette.
- No handwritten diagrams or noisy scans.
- No multi-page schematics.

## System Behavior

1. Receive a schematic image.
2. Detect bounding boxes for equipment symbols.
3. Detect arrows and arrowheads.
4. Determine arrow color.
5. Infer directed relationships using spatial reasoning.
6. Construct a directed graph representation.
7. Return a JSON encoding of the graph.

## Output Format (subject to change)

```json
{
  "nodes": [
    {"id": 1, "type": "oil_well", "bbox": [x1,y1,x2,y2]},
    {"id": 2, "type": "separator", "bbox": [x1,y1,x2,y2]}
  ],
  "edges": [
    {"from": 1, "to": 2, "relation": "oil_flow"}
  ]
}
```

## Development Plan

### Phase 1
- Collect sample facility schematics.
- Define initial equipment and relationship classes.

### Phase 2
- Label training data using annotation tools.
- Train an object detection model for equipment and arrows.

### Phase 3
- Implement arrow direction detection logic.
- Implement color-based relationship classification.

### Phase 4
- Design graph generation logic.
- Define JSON schema for API responses.

### Phase 5
- Build a REST API interface in Python.
- Package the system for local deployment.

## Evaluation

- Precision and recall for equipment detection.
- Accuracy of arrow direction inference.
- Accuracy of relationship classification.
- Manual inspection of generated graphs.

## Deliverables

- Trained object detection model.
- Python-based inference pipeline.
- REST API implementation.
- Example schematic dataset.
- Final technical report.


