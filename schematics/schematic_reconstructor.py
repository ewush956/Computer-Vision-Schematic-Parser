"""
Typical usage
-------------
    reconstructor = SchematicReconstructor(confidence_threshold=0.35)
    schematic     = reconstructor.load_xml("schematic_001.xml")
    filtered      = reconstructor.filter_by_confidence(schematic)
    linked        = reconstructor.link_text_to_components(filtered)
    connected     = reconstructor.connect_components(linked)
    non_text      = reconstructor.filter_by_class(connected, ["text"], exclude=True)
    canvas        = reconstructor.render_canvas(non_text)
    reconstructor.draw_components(canvas, non_text)
    reconstructor.draw_lines(canvas, non_text)
    reconstructor.annotate_labels(canvas, non_text)
    reconstructor.export_image(canvas, "output_001.png")

"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from schematics.schematic import BoundingBox, Component, Label, Line, Schematic
import numpy as np
import cv2


class SchematicReconstructor:
    def __init__(
        self,
        confidence_threshold: float = 0.30,
        text_link_distance: float = 50.0,
        connection_distance: float = 30.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.text_link_distance = text_link_distance
        self.connection_distance = connection_distance

    def reconstruct(
        self,
        xml: ET.ElementTree | ET.Element,
        polyLines,
        image_path: str | None = None,
    ):
        schematic = self.load_xml(xml)
        schematic = self.connect_components(schematic, polyLines)
        return schematic

    # DEPANSHU: Parse XML into data model — entry point for all bounding box coordinate data
    def load_xml(self, xml: ET.ElementTree) -> Schematic:
        """
        Parse a pipeline-generated XML file and return a Schematic object.

        Reads the <schematic> root for image name and canvas dimensions,
        then iterates <component> children to populate Component + BoundingBox
        instances.

        Parameters
        ----------
        xml_path : path to the .xml file produced by convert_detections_to_xml()
        """
        root = xml.getroot() if hasattr(xml, "getroot") else xml
        width = int(root.get("width", "0"))
        height = int(root.get("height", "0"))

        components = []
        for component in root.findall("component"):
            comp_id = int(component.get("id", "0"))
            box_data = component.find("bounding_box")
            class_name = component.get("class", "unknown")
            confidence = float(component.get("confidence", "0.0"))
            box = BoundingBox(
                xmin=int(box_data.get("xmin", "0")),
                ymin=int(box_data.get("ymin", "0")),
                xmax=int(box_data.get("xmax", "0")),
                ymax=int(box_data.get("ymax", "0")),
            )
            components.append(
                Component(
                    id=comp_id,
                    class_name=class_name,
                    confidence=confidence,
                    bounding_box=box,
                )
            )
        return Schematic(
            id=0,
            image_name=root.get("image_path", "unknown"),
            width=width,
            height=height,
            components=components,
            labels=[],
            lines=[],
        )

    # ------------------------------------------------------------------
    # 2. Filtering
    # ------------------------------------------------------------------

    # DEPANSHU: Discard low-confidence bounding box detections
    def filter_by_confidence(
        self, schematic: Schematic, threshold: float | None = None
    ) -> Schematic:
        """
        Return a new Schematic containing only components whose confidence
        meets or exceeds threshold. The original Schematic is not mutated.

        Parameters
        ----------
        threshold : defaults to self.confidence_threshold if not provided
        """
        threshold = threshold if threshold is not None else self.confidence_threshold
        kept = [c for c in schematic.components if c.confidence >= threshold]
        return Schematic(
            id=schematic.id,
            image_name=schematic.image_name,
            width=schematic.width,
            height=schematic.height,
            components=kept,
            labels=list(schematic.labels),
            lines=list(schematic.lines),
        )

    # AMTOJ: Isolate or remove component types — used to strip "text" detections after linking
    def filter_by_class(
        self,
        schematic: Schematic,
        class_names: list[str],
        exclude: bool = False,
    ) -> Schematic:
        """
        Return a new Schematic whose components match (or, when exclude=True,
        do not match) any name in class_names.

        Parameters
        ----------
        schematic   : source Schematic
        class_names : list of class names to keep or drop
        exclude     : if True, drop the named classes instead of keeping them
        """
        names = set(class_names)
        if exclude:
            kept = [c for c in schematic.components if c.class_name not in names]
        else:
            kept = [c for c in schematic.components if c.class_name in names]
        return Schematic(
            id=schematic.id,
            image_name=schematic.image_name,
            width=schematic.width,
            height=schematic.height,
            components=kept,
            labels=list(schematic.labels),
            lines=list(schematic.lines),
        )

    # ------------------------------------------------------------------
    # 3. Component alignment
    # ------------------------------------------------------------------

    def align_components(
        self, schematic: Schematic, tolerance: int | None = None
    ) -> Schematic:
        """
        Cluster component centers along each axis independently and snap
        every member of a cluster to the cluster mean. Two components whose
        centers differ by less than ``tolerance`` on one axis are forced to
        share that axis — so Manhattan routing between them collapses to a
        single straight segment instead of a small dog-leg.

        Clustering is done by sorting the axis values and cutting a new
        cluster whenever the gap between consecutive values exceeds
        ``tolerance``.

        Parameters
        ----------
        tolerance : maximum center-to-center gap (px) that still counts as
                    "aligned". Defaults to 8 px, which keeps closely-spaced
                    components from collapsing onto each other while still
                    straightening wire rows/columns.
        """
        if tolerance is None:
            tolerance = 8
        if tolerance <= 0 or not schematic.components:
            return schematic

        boxes = [c.bounding_box for c in schematic.components]

        def spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
            return a1 > b0 and b1 > a0

        def cluster_means_guarded(
            values: list[float], other_spans: list[tuple[int, int]], tol: int
        ) -> list[float]:
            """Cluster by axis value, but never merge two components whose
            bboxes already overlap on the OTHER axis — collapsing their
            value here would stack them on top of each other.
            """
            order = sorted(range(len(values)), key=lambda i: values[i])
            result = [0.0] * len(values)
            i = 0
            while i < len(order):
                cluster = [order[i]]
                j = i + 1
                while j < len(order) and values[order[j]] - values[order[j - 1]] <= tol:
                    cand = order[j]
                    c0, c1 = other_spans[cand]
                    if any(
                        spans_overlap(c0, c1, other_spans[m][0], other_spans[m][1])
                        for m in cluster
                    ):
                        break
                    cluster.append(cand)
                    j += 1
                mean = sum(values[k] for k in cluster) / len(cluster)
                for k in cluster:
                    result[k] = mean
                i = j
            return result

        centers_x = [(b.xmin + b.xmax) / 2.0 for b in boxes]
        centers_y = [(b.ymin + b.ymax) / 2.0 for b in boxes]
        x_spans = [(b.xmin, b.xmax) for b in boxes]
        y_spans = [(b.ymin, b.ymax) for b in boxes]

        # When clustering x-centers, forbid merges that already overlap in y
        # (same row → collapsing x would stack them). And vice versa.
        new_x = cluster_means_guarded(centers_x, y_spans, tolerance)
        new_y = cluster_means_guarded(centers_y, x_spans, tolerance)

        new_components = []
        for c, nx, ny in zip(schematic.components, new_x, new_y):
            box = c.bounding_box
            # Integer cluster centers, matched against the int center the
            # snap routine will compute: (ymin+ymax)//2. Using the same
            # formula here guarantees every cluster member ends with the
            # exact same integer center — no 1-px jitter between aligned
            # wires.
            target_cx = int(round(nx))
            target_cy = int(round(ny))
            old_cx = (box.xmin + box.xmax) // 2
            old_cy = (box.ymin + box.ymax) // 2
            dx = target_cx - old_cx
            dy = target_cy - old_cy
            new_box = BoundingBox(
                xmin=box.xmin + dx,
                ymin=box.ymin + dy,
                xmax=box.xmax + dx,
                ymax=box.ymax + dy,
            )
            new_components.append(
                Component(
                    id=c.id,
                    class_name=c.class_name,
                    confidence=c.confidence,
                    bounding_box=new_box,
                    label=c.label,
                )
            )
        return Schematic(
            id=schematic.id,
            image_name=schematic.image_name,
            width=schematic.width,
            height=schematic.height,
            components=new_components,
            labels=list(schematic.labels),
            lines=list(schematic.lines),
        )

    # ------------------------------------------------------------------
    # 4. Text & component linking
    # ------------------------------------------------------------------

    # AMTOJ: Spatially associate each "text" detection with its nearest non-text component
    # Implement Euclidean (pacman) distance
    def link_text_to_components(
        self,
        schematic: Schematic,
        # 50 is just a guess, set this to whatever you think is reasonable
        max_distance_px: float = 50.0,
    ) -> Schematic:
        """
        For every component with class_name == "text", find the nearest
        non-text component by center distance. If within max_distance_px,
        write that text value into the target component's linked_text field.

        Returns a new Schematic; the original is not mutated.

        Parameters
        ----------
        schematic       : Schematic still containing "text" components
        max_distance_px : maximum center-to-center distance to form a link
        """
        text_components = [
            c for c in schematic.components if c.class_name.lower() == "text"
        ]
        non_text = [c for c in schematic.components if c.class_name.lower() != "text"]

        # Copy non-text components so we don't mutate the input Schematic.
        new_components: dict[int, Component] = {
            c.id: Component(
                id=c.id,
                class_name=c.class_name,
                confidence=c.confidence,
                bounding_box=c.bounding_box,
                label=c.label,
            )
            for c in non_text
        }

        new_labels: list[Label] = list(schematic.labels)

        for text_comp in text_components:
            tx, ty = self.get_center(text_comp)

            best_target = None
            best_dist = float("inf")
            for target in non_text:
                cx, cy = self.get_center(target)
                dist = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_target = target

            if best_target is None or best_dist > max_distance_px:
                continue

            label = Label(
                id=text_comp.id,
                raw_text=text_comp.class_name,
                confidence=text_comp.confidence,
                bounding_box=text_comp.bounding_box,
                semantic_type=None,
            )
            new_labels.append(label)
            new_components[best_target.id].label = label

        # Preserve original component order (keep "text" components too).
        rebuilt = []
        for c in schematic.components:
            if c.id in new_components:
                rebuilt.append(new_components[c.id])
            else:
                rebuilt.append(c)

        return Schematic(
            id=schematic.id,
            image_name=schematic.image_name,
            width=schematic.width,
            height=schematic.height,
            components=rebuilt,
            labels=new_labels,
            lines=list(schematic.lines),
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_center(self, component: Component) -> tuple[float, float]:
        """Return (cx, cy) center of the component's bounding box."""
        box = component.bounding_box
        return (box.center_x, box.center_y)

    def get_bounds(self, component: Component) -> tuple[int, int, int, int]:
        """Return (xmin, ymin, xmax, ymax) for the component's bounding box."""
        box = component.bounding_box
        return (box.xmin, box.ymin, box.xmax, box.ymax)

    # DEPANSHU: Determine which components are electrically connected based on proximity
    # Feel free to do this however you want, this is just a thought on how you could go about it.
    def connect_components(self, schematic, polyLines, strict_margin=30):
        schematic.lines = []

        for line_id, polyline in enumerate(polyLines):
            if len(polyline) < 2:
                continue

            start = polyline[0]
            end = polyline[-1]

            start_match = self.nearest_component_box(
                start,
                schematic.components,
                strict_margin,
            )

            end_match = self.nearest_component_box(
                end,
                schematic.components,
                strict_margin,
            )

            if start_match and end_match and start_match.id != end_match.id:
                status = "connected"
                from_id = start_match.id
                to_id = end_match.id

            elif start_match or end_match:
                status = "dangling"
                from_id = start_match.id if start_match else None
                to_id = end_match.id if end_match else None

            else:
                status = "orphan"
                from_id = None
                to_id = None

            schematic.lines.append(
                Line(
                    id=line_id,
                    from_id=from_id,
                    to_id=to_id,
                    status=status,
                    polyline=[(int(x), int(y)) for x, y in polyline],
                )
            )

        return schematic

    def point_to_box_distance(self, point, box):
        x, y = point

        dx = max(box.xmin - x, 0, x - box.xmax)
        dy = max(box.ymin - y, 0, y - box.ymax)

        return (dx * dx + dy * dy) ** 0.5

    def nearest_component_box(self, point, components, strict_margin=30):
        best_component = None
        best_distance = float("inf")

        for component in components:
            if component.class_name.lower() == "text":
                continue

            distance = self.point_to_box_distance(point, component.bounding_box)

            if distance < best_distance:
                best_component = component
                best_distance = distance

        if best_component is not None and best_distance <= strict_margin:
            return best_component

        return None

    # ------------------------------------------------------------------
    # 5. Canvas & rendering
    # ------------------------------------------------------------------

    # EVAN: Allocate the GUI drawing surface
    def render_canvas(self, schematic: Schematic, scale: float = 1.0):
        """
        Create and return a blank canvas sized to schematic.width × schematic.height
        (optionally scaled). Canvas type is implementation choice — NumPy
        array (OpenCV), PIL Image, or matplotlib Figure all work.

        Parameters
        ----------
        schematic : provides the canvas dimensions
        scale     : resize factor; 1.0 = original pixel dimensions
        """
        width = max(1, int(schematic.width * scale))
        height = max(1, int(schematic.height * scale))
        return np.full((height, width, 3), 255, dtype=np.uint8)

    # EVAN: Draw each detected component as a coloured bounding box on the canvas
    def draw_components(self, canvas, schematic: Schematic) -> None:
        """
        Draw a coloured bounding box for every component in schematic onto
        canvas. Colour is assigned per class_name internally. Uses get_bounds()
        to retrieve coordinates.

        Parameters
        ----------
        canvas    : the canvas returned by render_canvas()
        schematic : filtered Schematic whose component boxes will be drawn
        """
        symbol_dir = Path("assets/symbols")

        # Precompute each component's preferred orientation from the wires
        # that terminate on it. Uses schematic.lines (populated by
        # connect_components) so we don't need to know about drawing order.
        orientations = self._component_orientations(schematic)

        for component in schematic.components:
            class_name = component.class_name.lower()
            if class_name == "text":
                continue

            xmin, ymin, xmax, ymax = self.get_bounds(component)
            w = max(1, xmax - xmin)
            h = max(1, ymax - ymin)

            # Clip to canvas bounds.
            canvas_h, canvas_w = canvas.shape[:2]
            x0, y0 = max(0, xmin), max(0, ymin)
            x1, y1 = min(canvas_w, xmax), min(canvas_h, ymax)
            if x1 <= x0 or y1 <= y0:
                continue

            # Junctions are wire-branch dots, not full symbols.
            if class_name == "junction":
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2
                radius = max(2, min(w, h) // 4)
                cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), thickness=-1)
                continue

            orientation = orientations.get(component.id, "horizontal")
            primary = symbol_dir / f"{component.class_name}_{orientation}.png"
            fallback = symbol_dir / (
                f"{component.class_name}_"
                f"{'vertical' if orientation == 'horizontal' else 'horizontal'}.png"
            )
            symbol_path = primary if primary.exists() else fallback
            symbol = None
            if symbol_path.exists():
                symbol = cv2.imread(str(symbol_path), cv2.IMREAD_UNCHANGED)

            if symbol is not None:
                # Composite RGBA onto white -> BGR
                if symbol.ndim == 3 and symbol.shape[2] == 4:
                    bgr = symbol[..., :3].astype(np.float32)
                    alpha = symbol[..., 3:4].astype(np.float32) / 255.0
                    white = np.full_like(bgr, 255.0)
                    bgr = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
                elif symbol.ndim == 2:
                    bgr = cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = symbol[..., :3]

                # Letterbox: preserve aspect ratio, center inside the bounding box.
                sym_h, sym_w = bgr.shape[:2]
                scale = min(w / sym_w, h / sym_h)
                new_w = max(1, int(round(sym_w * scale)))
                new_h = max(1, int(round(sym_h * scale)))
                resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Placement in image coordinates, centered inside [xmin..xmax, ymin..ymax].
                px0 = xmin + (w - new_w) // 2
                py0 = ymin + (h - new_h) // 2
                px1 = px0 + new_w
                py1 = py0 + new_h

                # Clip to canvas bounds, tracking source offsets.
                dx0 = max(px0, 0)
                dy0 = max(py0, 0)
                dx1 = min(px1, canvas_w)
                dy1 = min(py1, canvas_h)
                if dx1 <= dx0 or dy1 <= dy0:
                    continue
                sx0 = dx0 - px0
                sy0 = dy0 - py0
                sx1 = sx0 + (dx1 - dx0)
                sy1 = sy0 + (dy1 - dy0)
                canvas[dy0:dy1, dx0:dx1] = resized[sy0:sy1, sx0:sx1]
            else:
                color = self._class_color(component.class_name)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

    # EVAN: Draw arrows between connected components on the canvas
    def draw_lines(self, canvas, schematic: Schematic) -> None:
        """
        For every Line in schematic.lines, draw an arrow from the center of
        the from_id component to the center of the to_id component.
        Uses get_center() to retrieve coordinates.

        Parameters
        ----------
        canvas    : canvas already passed through draw_components()
        schematic : Schematic whose lines list will be rendered
        """
        components_by_id = {c.id: c for c in schematic.components}

        for line in schematic.lines:
            if line.status not in ("connected", "dangling"):
                continue
            if len(line.polyline) < 2:
                continue

            start = line.polyline[0]
            end = line.polyline[-1]

            # Snap each endpoint onto the matched component's bounding box edge
            # so the wire visibly terminates at the component. Unmatched ends of
            # dangling wires are left at the traced endpoint.
            from_comp = components_by_id.get(line.from_id)
            to_comp = components_by_id.get(line.to_id)
            if from_comp is not None:
                start = self._snap_to_box_edge(start, from_comp.bounding_box)
            if to_comp is not None:
                end = self._snap_to_box_edge(end, to_comp.bounding_box)

            # Manhattan Z-route between the (snapped) endpoints.
            x0, y0 = start
            x1, y1 = end
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            if dx >= dy:
                mid_x = self._snap((x0 + x1) / 2, 8)
                route = [(x0, y0), (mid_x, y0), (mid_x, y1), (x1, y1)]
            else:
                mid_y = self._snap((y0 + y1) / 2, 8)
                route = [(x0, y0), (x0, mid_y), (x1, mid_y), (x1, y1)]

            points = np.array(route, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [points],
                isClosed=False,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_8,
            )

    # AMTOJ: Resolve label text for linked components
    # EVAN: Render the label overlay onto the canvas
    # We can probably leave this until later and just do this one together using the power of friendship </3.
    def annotate_labels(
        self,
        canvas,
        schematic: Schematic,
        show_confidence: bool = False,
    ) -> None:
        """
        Overlay a text label on each bounding box drawn by draw_components().
        If a component has a linked_text value, that is shown in place of the
        raw class name. Appends the confidence score when show_confidence=True.

        My though originally was showing the confidence as a percentage, but that's probably low priority.

        Parameters
        ----------
        canvas          : canvas already passed through draw_components()
        schematic       : the same Schematic used in draw_components()
        show_confidence : whether to append "(0.87)" to each label
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.32
        thickness = 1
        pad = 1
        canvas_h, canvas_w = canvas.shape[:2]

        for component in schematic.components:
            if component.class_name.lower() in ("text", "junction"):
                continue

            # Prefer a linked label, but skip the placeholder "text" raw_text
            # that leaks through when no OCR is present.
            if (
                component.label is not None
                and component.label.raw_text.lower() != "text"
            ):
                text = component.label.raw_text
            else:
                text = component.class_name

            if show_confidence:
                text = f"{text} ({component.confidence:.2f})"

            xmin, ymin, xmax, ymax = self.get_bounds(component)
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            box_h = th + pad * 2

            # Prefer above-the-box; fall back to inside-the-top when there's no room.
            bg_x0 = xmin
            bg_x1 = min(canvas_w, bg_x0 + tw + pad * 2)
            if ymin - box_h >= 0:
                bg_y0 = ymin - box_h
            else:
                bg_y0 = ymin
            bg_y1 = bg_y0 + box_h

            cv2.rectangle(
                canvas,
                (bg_x0, bg_y0),
                (bg_x1, bg_y1),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(
                canvas,
                text,
                (bg_x0 + pad, bg_y1 - pad),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------

    # EVAN: Component count data for the GUI legend / stats panel
    # This will be nice to have for the presentation. I'm thinking of building a little info dashboard to show how well the model did.
    def summarise(self, schematic: Schematic) -> dict[str, int]:
        """
        Return a dict mapping each class_name to its instance count.
        Example: {"resistor": 12, "capacitor": 7, "ground": 4}

        Used to populate the legend panel in the GUI.
        """
        counts: dict[str, int] = {}
        for component in schematic.components:
            if component.class_name.lower() in ("text", "junction"):
                continue
            counts[component.class_name] = counts.get(component.class_name, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # 7. Export
    # ------------------------------------------------------------------

    # EVAN: Save the rendered GUI canvas to disk
    def export_image(
        self,
        canvas,
        output_path: str | Path,
        quality: int = 95,
    ) -> None:
        """
        Save the annotated canvas to disk as an image file.
        Format is inferred from the file extension (.png, .jpg, etc.).

        Parameters
        ----------
        canvas      : the annotated canvas from annotate_labels()
        output_path : destination file path
        quality     : JPEG quality (1–100); ignored for lossless formats
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ext = output_path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

        cv2.imwrite(str(output_path), canvas, params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _component_orientations(self, schematic: Schematic) -> dict[int, str]:
        """Vote per-component on whether wires enter horizontally or vertically.

        For each wire endpoint that terminates at a component (from_id/to_id),
        compare the endpoint's horizontal-vs-vertical offset from the box
        center. The larger axis wins a vote for that orientation. Components
        with no incident wires default to 'horizontal'.
        """
        votes: dict[int, tuple[int, int]] = {}  # id -> (horiz, vert)
        components_by_id = {c.id: c for c in schematic.components}

        def cast_vote(comp_id: int, point: tuple[int, int]) -> None:
            comp = components_by_id.get(comp_id)
            if comp is None:
                return
            box = comp.bounding_box
            cx = (box.xmin + box.xmax) / 2.0
            cy = (box.ymin + box.ymax) / 2.0
            h, v = votes.get(comp_id, (0, 0))
            if abs(point[0] - cx) >= abs(point[1] - cy):
                h += 1
            else:
                v += 1
            votes[comp_id] = (h, v)

        for line in schematic.lines:
            if not line.polyline:
                continue
            if line.from_id is not None:
                cast_vote(line.from_id, line.polyline[0])
            if line.to_id is not None:
                cast_vote(line.to_id, line.polyline[-1])

        result: dict[int, str] = {}
        for comp_id, (h, v) in votes.items():
            result[comp_id] = "horizontal" if h >= v else "vertical"
        return result

    def _snap(self, v: float, grid: int) -> int:
        return int(round(v / grid) * grid)

    def snap_components_to_grid(self, schematic: Schematic, grid: int = 8) -> Schematic:
        new_components = []

        for c in schematic.components:
            b = c.bounding_box

            cx = (b.xmin + b.xmax) // 2
            cy = (b.ymin + b.ymax) // 2

            snapped_cx = self._snap(cx, grid)
            snapped_cy = self._snap(cy, grid)

            dx = snapped_cx - cx
            dy = snapped_cy - cy

            new_box = BoundingBox(
                xmin=b.xmin + dx,
                ymin=b.ymin + dy,
                xmax=b.xmax + dx,
                ymax=b.ymax + dy,
            )

            new_components.append(
                Component(
                    id=c.id,
                    class_name=c.class_name,
                    confidence=c.confidence,
                    bounding_box=new_box,
                    label=c.label,
                )
            )

        return Schematic(
            id=schematic.id,
            image_name=schematic.image_name,
            width=schematic.width,
            height=schematic.height,
            components=new_components,
            labels=list(schematic.labels),
            lines=list(schematic.lines),
        )

    def _snap_to_box_edge(
        self, point: tuple[int, int], box: BoundingBox, grid: int = 8
    ) -> tuple[int, int]:
        """Snap ``point`` to the midpoint of whichever box edge it approaches.

        Pick the edge whose overshoot from ``point`` is largest (left/right/
        top/bottom), then return that edge's midpoint. Unlike projecting the
        point onto the edge, this collapses any per-wire jitter in the free
        coordinate: two components that share a y-coordinate both receive
        wires at their exact center_y, so Manhattan routing produces a single
        straight segment instead of a small step.

        Tradeoff: two wires entering the same side of the same component land
        on the same pixel. In practice multi-pin components (transistors, op-
        amps, ICs) have pins on different sides, so this rarely causes visual
        overlap.
        """
        x, y = point
        cx_box = (box.xmin + box.xmax) // 2
        cy_box = (box.ymin + box.ymax) // 2

        overs = (
            (box.xmin - x, "left"),
            (x - box.xmax, "right"),
            (box.ymin - y, "top"),
            (y - box.ymax, "bottom"),
        )
        max_over, side = max(overs, key=lambda o: o[0])

        if max_over <= 0:
            dists = (
                (x - box.xmin, "left"),
                (box.xmax - x, "right"),
                (y - box.ymin, "top"),
                (box.ymax - y, "bottom"),
            )
            _, side = min(dists, key=lambda d: d[0])

        if side == "left":
            return (self._snap(box.xmin, grid), self._snap(cy_box, grid))
        if side == "right":
            return (self._snap(box.xmax, grid), self._snap(cy_box, grid))
        if side == "top":
            return (self._snap(cx_box, grid), self._snap(box.ymin, grid))
        return (self._snap(cx_box, grid), self._snap(box.ymax, grid))

    def _class_color(self, class_name: str) -> tuple[int, int, int]:
        """Deterministic BGR color keyed off the class name."""
        h = hash(class_name) & 0xFFFFFF
        b = 60 + (h & 0xFF) % 180
        g = 60 + ((h >> 8) & 0xFF) % 180
        r = 60 + ((h >> 16) & 0xFF) % 180
        return (int(b), int(g), int(r))


def visualize_schematic(schematic, output_path="output/schematic_graph.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = np.full((schematic.height, schematic.width, 3), 255, dtype=np.uint8)
    components_by_id = {component.id: component for component in schematic.components}

    status_colors = {
        "connected": (0, 180, 0),
        "dangling": (0, 0, 255),
        "orphan": (160, 160, 160),
        "repaired": (255, 180, 0),
    }

    for line in schematic.lines:
        if line.status != "connected":
            continue
        color = status_colors.get(line.status, (0, 0, 0))

        if len(line.polyline) >= 2:
            points = np.array(line.polyline, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [points],
                isClosed=False,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            continue

        if line.from_id is None or line.to_id is None:
            continue

        from_component = components_by_id[line.from_id]
        to_component = components_by_id[line.to_id]

        p1 = (
            int(from_component.bounding_box.center_x),
            int(from_component.bounding_box.center_y),
        )
        p2 = (
            int(to_component.bounding_box.center_x),
            int(to_component.bounding_box.center_y),
        )

        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    for component in schematic.components:
        if component.class_name.lower() == "text":
            continue

        box = component.bounding_box
        cv2.rectangle(
            canvas,
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            (0, 180, 255),
            2,
        )

        cv2.putText(
            canvas,
            component.class_name,
            (box.xmin, max(15, box.ymin - 5)),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


if __name__ == "__main__":
    # Full reconstruction pipeline on tests/images/testcase1.png.
    import argparse

    from detectors.wire_detect import detect_wires
    from detectors.yolo_detection import detect_and_export_to_xml

    WIRE_MODEL = "models/unet_best.pth"
    YOLO_MODEL = "models/yolo.pt"

    arg_parser = argparse.ArgumentParser(description="Reconstruct a schematic image.")
    arg_parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input schematic image. If omitted, all .png/.jpg in tests/images/ are processed.",
    )
    arg_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (single-image mode only; default: tests/outputs/<stem>_reconstructed.png).",
    )
    labels_group = arg_parser.add_mutually_exclusive_group()
    labels_group.add_argument(
        "--labels",
        dest="labels",
        action="store_true",
        help="Draw component labels (default).",
    )
    labels_group.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        help="Skip drawing labels.",
    )
    arg_parser.set_defaults(labels=True)
    arg_parser.add_argument(
        "--align",
        type=int,
        default=None,
        help="Alignment tolerance (px) for clustering nearby component centers "
        "onto a shared axis. 0 disables alignment. Default: adaptive to image width.",
    )
    args = arg_parser.parse_args()

    if args.image is not None:
        images = [args.image]
    else:
        images_dir = Path("tests/images")
        images = sorted(
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not images:
            raise SystemExit(f"No .png/.jpg images found in {images_dir}")

    reconstructor = SchematicReconstructor()
    suffix = "_reconstructed" if args.labels else "_nolabels"

    for image_path in images:
        if args.output is not None and len(images) == 1:
            output_path = args.output
        else:
            output_path = Path(f"tests/outputs/{image_path.stem}{suffix}.png")

        component_xml = detect_and_export_to_xml(
            model_path=YOLO_MODEL,
            image_path=str(image_path),
            output_xml_path=f"output/{image_path.stem}.xml",
        )

        _cleaned, _erased, _skeleton, polylines = detect_wires(
            image_path, component_xml.getroot(), WIRE_MODEL
        )

        schematic = reconstructor.reconstruct(component_xml, polylines, str(image_path))
        schematic = reconstructor.filter_by_confidence(schematic)
        schematic = reconstructor.link_text_to_components(schematic)
        # Keep junctions — they're rendered as connection dots; only drop text.
        schematic = reconstructor.filter_by_class(schematic, ["text"], exclude=True)
        schematic = reconstructor.align_components(schematic, tolerance=args.align)

        canvas = reconstructor.render_canvas(schematic)
        reconstructor.draw_components(canvas, schematic)
        reconstructor.draw_lines(canvas, schematic)
        if args.labels:
            reconstructor.annotate_labels(canvas, schematic)
        reconstructor.export_image(canvas, output_path)

        print(
            f"Wrote {output_path}  (labels={'on' if args.labels else 'off'}, "
            f"polylines={len(polylines)})"
        )
        print(f"  Counts: {reconstructor.summarise(schematic)}")
