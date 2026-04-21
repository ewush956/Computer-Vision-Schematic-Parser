from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Absolute pixel coordinates of one detected component."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0


@dataclass
class Label:
    id: int
    raw_text: str
    confidence: float
    bounding_box: BoundingBox
    semantic_type: str | None  # output of SchematicTextClassifier.classify()


@dataclass
class Component:
    """A single detected electrical component."""

    id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    label: Label | None = None  # populated by link_text_to_components()


@dataclass
class Line:
    """A traced wire, with optional matched component endpoints."""

    id: int
    from_id: int | None
    to_id: int | None
    status: str
    polyline: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Schematic:
    id: int
    image_name: str
    width: int
    height: int
    components: list[Component] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
