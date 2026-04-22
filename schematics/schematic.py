from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from xml.etree import ElementTree as ET

from ultralytics.engine.results import Results

Point = tuple[int, int]
Polyline = list[Point]


def _float(val: str | None) -> float | None:
    """Parse an optional XML float attribute."""
    return float(val) if val is not None else None


def _int(val: str | None) -> int | None:
    """Parse an optional XML integer attribute."""
    return int(val) if val is not None else None


@dataclass
class Component:
    """Detected schematic component or text box in absolute pixel coordinates."""

    id: int
    class_name: str
    yolo_conf: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    text: str | None = None
    text_type: str | None = None

    @property
    def is_text(self) -> bool:
        """
        Return True when this detection is a text annotation box.
        """
        return self.class_name == "text"

    @property
    def width(self) -> int:
        """
        Bounding-box width in pixels.
        """
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """
        Bounding-box height in pixels.
        """
        return self.ymax - self.ymin

    @property
    def center_x(self) -> float:
        """
        Horizontal center of the bounding box.
        """
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        """
        Vertical center of the bounding box.
        """
        return (self.ymin + self.ymax) / 2.0


@dataclass
class Line:
    """
    Traced wire polyline with optional matched component endpoints.
    """

    id: int
    polyline: Polyline
    start_component_id: int | None = None
    end_component_id: int | None = None
    status: str = "orphan" # Can be orphan(if not connected to any at all), connected or dangling( if only conencted to one)


@dataclass
class Schematic:
    """
    A schematic is a representation of the hand drawn circuit .
    """

    width: int
    height: int
    image_path: str
    components: list[Component] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    _id_counter: Iterator[int] = field(default_factory=count, init=False, repr=False)




    def _new_id(self) -> int:
        return next(self._id_counter)

    def add_component(
        self,
        class_name: str,
        yolo_conf: float,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        text: str | None = None,
        ocr_conf: float | None = None,
        text_type: str | None = None,
    ) -> Component:
        component = Component(
            id=self._new_id(),
            class_name=class_name,
            yolo_conf=yolo_conf,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            text=text,
            text_type=text_type,
        )
        self.components.append(component)
        return component

    def add_line(
        self,
        polyline: Polyline,
        status: str = "orphan",
        start_component_id: int | None = None,
        end_component_id: int | None = None,
    ) -> Line:
        line = Line(
            id=self._new_id(),
            polyline=[(int(x), int(y)) for x, y in polyline],
            start_component_id=start_component_id,
            end_component_id=end_component_id,
            status=status,
        )
        self.lines.append(line)
        return line

    def add_polylines(
        self,
        polyline: Polyline,
        status: str,
        start_component: int | None = None,
        end_component: int | None = None,
    ) -> Line:
        """Compatibility wrapper for callers that still use the old method name."""
        return self.add_line(
            polyline=polyline,
            status=status,
            start_component_id=start_component,
            end_component_id=end_component,
        )


class SchematicParser:
    """ 
    Utility class for converting schematics from various representations
    """

    @staticmethod
    def save_to_xml(schematic: Schematic, output_path: str | Path) -> None:
        """W
        Writes a Schematic object to the pipeline XML format
        """
        root = ET.Element(
            "schematic",
            {
                "width": str(schematic.width),
                "height": str(schematic.height),
                "image_path": schematic.image_path,
            },
        )

        for comp in schematic.components:
            attribs = {
                "id": str(comp.id),
                "class": comp.class_name,
                "yolo_conf": f"{comp.yolo_conf:.4f}",
                "xmin": str(comp.xmin),
                "ymin": str(comp.ymin),
                "xmax": str(comp.xmax),
                "ymax": str(comp.ymax),
            }
            if comp.text is not None:
                attribs["text"] = comp.text
            if comp.text_type is not None:
                attribs["text_type"] = comp.text_type

            ET.SubElement(root, "component", attribs)

        for line in schematic.lines:
            attribs = {
                "id": str(line.id),
                "status": line.status,
            }
            if line.start_component_id is not None:
                attribs["from_id"] = str(line.start_component_id)
            if line.end_component_id is not None:
                attribs["to_id"] = str(line.end_component_id)

            line_elem = ET.SubElement(root, "line", attribs)
            for x, y in line.polyline:
                ET.SubElement(line_elem, "point", x=str(x), y=str(y))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def from_yolo_to_schematic(result: Results) -> Schematic:
        """
        Convert one YOLO result into a Schematic object
        """
        h, w = result.orig_shape
        schematic = Schematic(width=w, height=h, image_path=str(result.path))
        if result.boxes is None:
            return schematic
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            schematic.add_component(
                class_name=result.names[int(box.cls[0])],
                yolo_conf=float(box.conf[0]),
                xmin=int(round(x1)),
                ymin=int(round(y1)),
                xmax=int(round(x2)),
                ymax=int(round(y2)),
            )

        return schematic
