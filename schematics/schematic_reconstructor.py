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

from copy import deepcopy
import xml.etree.ElementTree as ET
from pathlib import Path

from schematics.schematic import Schematic
from schematics import routing
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Cached TTF font handle — Hershey stroke fonts used by cv2.putText are
# ASCII-only, so Ω/μ/° etc. render as "?". Pillow + TrueType gives us full
# Unicode coverage for the OCR'd labels.
_UNICODE_FONT_CANDIDATES = (
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:\\Windows\\Fonts\\arial.ttf",
)


def _unicode_font_path() -> str | None:
    for candidate in _UNICODE_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


_UNICODE_FONT_PATH = _unicode_font_path()


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

    # ------------------------------------------------------------------
    # Filtering
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
        kept = [c for c in schematic.components if c.yolo_conf >= threshold]
        return Schematic(
            width=schematic.width,
            height=schematic.height,
            image_path=schematic.image_path,
            components=kept,
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
            width=schematic.width,
            height=schematic.height,
            image_path=schematic.image_path,
            components=kept,
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

        Text components are excluded from clustering: their centers are
        often off-grid relative to the real symbols, and shifting them would
        move OCR'd labels away from where the writer placed them.

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

        comps = [c for c in schematic.components if not c.is_text]
        if not comps:
            return schematic

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

        centers_x = [(c.xmin + c.xmax) / 2.0 for c in comps]
        centers_y = [(c.ymin + c.ymax) / 2.0 for c in comps]
        x_spans = [(c.xmin, c.xmax) for c in comps]
        y_spans = [(c.ymin, c.ymax) for c in comps]

        # When clustering x-centers, forbid merges that already overlap in y
        # (same row → collapsing x would stack them). And vice versa.
        new_x = cluster_means_guarded(centers_x, y_spans, tolerance)
        new_y = cluster_means_guarded(centers_y, x_spans, tolerance)

        # Mutate in place: the Schematic dataclass doesn't expose a way to
        # rebuild with preserved IDs without poking at internal state, and
        # Component is a plain dataclass so attribute assignment is fine.
        for c, nx, ny in zip(comps, new_x, new_y):
            # Integer cluster centers, matched against the int center the
            # snap routine computes: (xmin+xmax)//2. Using the same formula
            # here guarantees every cluster member ends with the exact same
            # integer center — no 1-px jitter between aligned wires.
            target_cx = int(round(nx))
            target_cy = int(round(ny))
            old_cx = (c.xmin + c.xmax) // 2
            old_cy = (c.ymin + c.ymax) // 2
            dx = target_cx - old_cx
            dy = target_cy - old_cy
            c.xmin += dx
            c.xmax += dx
            c.ymin += dy
            c.ymax += dy
        return schematic

    # AMTOJ: Spatially associate each "text" detection with its nearest non-text component
    # Implement Euclidean (pacman) distance
    def link_text_to_components(
        self,
        schematic: Schematic,
        max_distance_px: float = 150.0,
    ) -> Schematic:
        """
        For every component with class_name == "text", find the nearest
        non-text component by center distance. If within max_distance_px,
        copy the OCR string into that target component's ``text`` field so
        the legend / banner label can display the value. The text component
        itself is *kept* in the schematic — ``annotate_labels`` renders the
        handwritten OCR at the original bbox in-place (matching the size and
        position of the writer's hand-lettered label).

        Parameters
        ----------
        schematic       : Schematic still containing "text" components
        max_distance_px : maximum center-to-center distance to form a link
        """
        text_components = [c for c in schematic.components if c.is_text]
        non_text = [c for c in schematic.components if not c.is_text]

        for text_comp in text_components:
            if text_comp.text is None:
                continue
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
                print(
                    f"TEXT (unlinked): '{text_comp.text}' "
                    f"nearest={best_target.class_name if best_target else None} "
                    f"dist={best_dist:.1f}"
                )
                continue

            # Mirror OCR text onto the nearest non-text component so legend
            # summaries can reference it; the text component keeps its own
            # copy for in-place rendering.
            if best_target.text is None:
                best_target.text = text_comp.text
                best_target.text_type = text_comp.text_type
            print(
                f"TEXT: '{text_comp.text}' -> {best_target.class_name} "
                f"dist={best_dist:.1f}"
            )
        return schematic

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_center(self, component) -> tuple[float, float]:
        """Return (cx, cy) center of the component's bounding box."""
        return (component.center_x, component.center_y)

    def get_bounds(self, component) -> tuple[int, int, int, int]:
        """Return (xmin, ymin, xmax, ymax) for the component's bounding box."""
        return (component.xmin, component.ymin, component.xmax, component.ymax)

    # DEPANSHU: Determine which components are electrically connected based on proximity
    # Feel free to do this however you want, this is just a thought on how you could go about it.
    def connect_components(self, schematic: Schematic, polyLines, strict_margin=30):
        schematic_with_lines = deepcopy(schematic)
        for polyline in polyLines:
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
                exclude_component_id=start_match.id if start_match else None,
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

            schematic_with_lines.add_polylines(
                polyline, status, start_component=from_id, end_component=to_id
            )

        return schematic_with_lines

    def point_to_box_distance(self, point, component):
        x, y = point

        dx = max(component.xmin - x, 0, x - component.xmax)
        dy = max(component.ymin - y, 0, y - component.ymax)

        return (dx * dx + dy * dy) ** 0.5

    def nearest_component_box(
        self,
        point,
        components,
        strict_margin=30,
        exclude_component_id: int | None = None,
    ):
        best_component = None
        best_distance = float("inf")

        for component in components:

            if (
                component.class_name.lower() == "text"
                or component.id == exclude_component_id
            ):
                continue

            distance = self.point_to_box_distance(point, component)

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
        """Render every Line as an orthogonal A*-routed polyline.

        Routing happens on an 8-px grid (see :mod:`schematics.routing`):

        1. Choose the port pair on the two components that minimises
           Manhattan distance.
        2. A* search avoids every other component's bounding box.
        3. Collinear waypoints are collapsed so the final polyline is only
           endpoints and bends.

        Dangling wires (only one matched end) are routed from the matched
        component's nearest port to the traced endpoint.
        """
        components_by_id = {c.id: c for c in schematic.components}
        # Text bboxes shouldn't deflect wires — they're just annotations.
        obstacles = [c for c in schematic.components if not c.is_text]

        for line in schematic.lines:
            if line.status not in ("connected", "dangling"):
                continue
            if len(line.polyline) < 2:
                continue

            from_comp = components_by_id.get(line.start_component_id)
            to_comp = components_by_id.get(line.end_component_id)

            route = self._route_line(line, from_comp, to_comp, obstacles)
            if route is None or len(route) < 2:
                continue

            points = np.array(route, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [points],
                isClosed=False,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    def _route_line(
        self, line, from_comp, to_comp, obstacles
    ) -> list[tuple[int, int]] | None:
        """Return an orthogonal grid route for ``line``, or None if unroutable."""
        start_dir = None
        end_dir = None
        exclude_ids: set[int] = set()

        if from_comp is not None and to_comp is not None:
            (start, from_side), (end, to_side) = routing.select_port_pair(
                from_comp, to_comp
            )
            start_dir = routing.port_outward(from_side)
            # end_dir is the step *into* the end port, i.e. opposite the
            # port's outward direction.
            ox, oy = routing.port_outward(to_side)
            end_dir = (-ox, -oy)
            exclude_ids = {from_comp.id, to_comp.id}
        elif from_comp is not None:
            start, from_side = routing.nearest_port(from_comp, line.polyline[-1])
            end = routing.snap_point(line.polyline[-1])
            start_dir = routing.port_outward(from_side)
            exclude_ids = {from_comp.id}
        elif to_comp is not None:
            start = routing.snap_point(line.polyline[0])
            end, to_side = routing.nearest_port(to_comp, line.polyline[0])
            ox, oy = routing.port_outward(to_side)
            end_dir = (-ox, -oy)
            exclude_ids = {to_comp.id}
        else:
            return None

        return routing.route_line(
            start,
            end,
            obstacles,
            exclude_ids=exclude_ids,
            start_dir=start_dir,
            end_dir=end_dir,
        )

    # AMTOJ: Resolve label text for linked components
    # EVAN: Render the label overlay onto the canvas
    def annotate_labels(
        self,
        canvas,
        schematic: Schematic,
        show_confidence: bool = False,
    ) -> None:
        """
        Two kinds of labels are drawn:

        * **Text components** — the OCR'd string is rendered at the original
          text bounding box with a font scale that fits the box, so a
          handwritten "24V" re-appears at the same place and size as the
          original label. This is the user-facing schematic annotation.
        * **Non-text components** — a small banner above the box shows the
          class name (or the linked OCR value when present). This is a
          lightweight legend for debugging detections.

        Junctions are skipped entirely.

        Parameters
        ----------
        canvas          : canvas already passed through draw_components()
        schematic       : the same Schematic used in draw_components()
        show_confidence : whether to append "(0.87)" to each banner label
        """
        canvas_h, canvas_w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ---- First pass: render OCR text at each text component's bbox ----
        # Use Pillow so Unicode glyphs (Ω, μ, °, ...) render correctly. We
        # blit the text onto a PIL image wrapping the same numpy buffer.
        text_items: list[tuple[str, int, int, int, int]] = []
        for component in schematic.components:
            if not component.is_text:
                continue
            if component.text is None:
                continue
            text = component.text.strip()
            if not text or text.lower() == "text":
                continue
            xmin, ymin, xmax, ymax = self.get_bounds(component)
            text_items.append((text, xmin, ymin, xmax, ymax))

        if text_items:
            # cv2 canvas is BGR; PIL expects RGB. Round-trip through a
            # numpy view so the final BGR write overwrites canvas in place.
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_img)
            for text, xmin, ymin, xmax, ymax in text_items:
                box_w = max(1, xmax - xmin)
                box_h = max(1, ymax - ymin)
                pil_font, (tw, th), offset = self._fit_pil_font(
                    text, box_w, box_h
                )
                ox, oy = offset
                # Center glyphs inside the bbox, correcting for the glyph
                # bearing (PIL's draw origin is top-left of the glyph's ink,
                # minus its left/top bearings — subtracting the bbox offset
                # puts the visible ink where we expect).
                tx = xmin + (box_w - tw) // 2 - ox
                ty = ymin + (box_h - th) // 2 - oy
                draw.text((tx, ty), text, font=pil_font, fill=(0, 0, 0))
            # Copy the result back into the cv2 canvas (BGR, in-place).
            canvas[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

        # ---- Second pass: banner labels for non-text components ----
        used_regions: list[tuple[int, int, int, int]] = []
        banner_scale = 0.32
        banner_thickness = 1
        pad = 1

        for component in schematic.components:
            if component.is_text:
                continue
            if component.class_name.lower() == "junction":
                continue

            if component.text is not None and component.text.lower() != "text":
                text = component.text
            else:
                text = component.class_name

            if show_confidence:
                text = f"{text} ({component.yolo_conf:.2f})"

            xmin, ymin, xmax, ymax = self.get_bounds(component)
            (tw, th), _ = cv2.getTextSize(
                text, font, banner_scale, banner_thickness
            )
            box_h = th + pad * 2

            cx = (xmin + xmax) // 2
            bg_x0 = max(0, cx - (tw + pad * 2) // 2)
            bg_x1 = min(canvas_w, bg_x0 + tw + pad * 2)
            if ymin - box_h >= 0:
                bg_y0 = ymin - box_h
            else:
                bg_y0 = ymin
            bg_y1 = bg_y0 + box_h
            for ux0, uy0, ux1, uy1 in used_regions:
                overlap = not (
                    bg_x1 < ux0 or bg_x0 > ux1 or bg_y1 < uy0 or bg_y0 > uy1
                )
                if overlap:
                    bg_y0 -= box_h + 2
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
                banner_scale,
                (255, 255, 255),
                banner_thickness,
                cv2.LINE_AA,
            )
            used_regions.append((bg_x0, bg_y0, bg_x1, bg_y1))

    def _fit_pil_font(
        self,
        text: str,
        max_w: int,
        max_h: int,
        max_px: int = 200,
    ) -> tuple[ImageFont.ImageFont, tuple[int, int], tuple[int, int]]:
        """Pick the largest TrueType font size that fits ``text`` in
        ``max_w × max_h`` px. Falls back to PIL's default bitmap font if no
        TTF is available.

        Returns ``(font, (width, height), (offset_x, offset_y))`` where the
        offset is the inked bbox's (left, top) so callers can correct for
        glyph bearing when centering.
        """
        target_w = max(1, int(max_w * 0.95))
        target_h = max(1, int(max_h * 0.85))

        if _UNICODE_FONT_PATH is None:
            font = ImageFont.load_default()
            left, top, right, bottom = font.getbbox(text)
            return font, (right - left, bottom - top), (left, top)

        # Binary search the font size that fits the bbox.
        lo, hi = 6, max_px
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            font = ImageFont.truetype(_UNICODE_FONT_PATH, mid)
            left, top, right, bottom = font.getbbox(text)
            tw, th = right - left, bottom - top
            if tw <= target_w and th <= target_h:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        font = ImageFont.truetype(_UNICODE_FONT_PATH, best)
        left, top, right, bottom = font.getbbox(text)
        return font, (right - left, bottom - top), (left, top)

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
            cx = comp.center_x
            cy = comp.center_y
            h, v = votes.get(comp_id, (0, 0))
            if abs(point[0] - cx) >= abs(point[1] - cy):
                h += 1
            else:
                v += 1
            votes[comp_id] = (h, v)

        for line in schematic.lines:
            if not line.polyline:
                continue
            if line.start_component_id is not None:
                cast_vote(line.start_component_id, line.polyline[0])
            if line.end_component_id is not None:
                cast_vote(line.end_component_id, line.polyline[-1])

        result: dict[int, str] = {}
        for comp_id, (h, v) in votes.items():
            result[comp_id] = "horizontal" if h >= v else "vertical"
        return result

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
    color = (0, 180, 0)

    for line in schematic.lines:
        if line.status != "connected":
            continue

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

        from_component = components_by_id[line.start_component_id]
        to_component = components_by_id[line.end_component_id]

        p1 = (
            int(from_component.center_x),
            int(from_component.center_y),
        )
        p2 = (
            int(to_component.center_x),
            int(to_component.center_y),
        )

        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    for component in schematic.components:
        cv2.rectangle(
            canvas,
            (component.xmin, component.ymin),
            (component.xmax, component.ymax),
            (0, 180, 255),
            2,
        )

        cv2.putText(
            canvas,
            component.class_name,
            (component.xmin, max(15, component.ymin - 5)),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


# ---------------------------------------------------------------------------
# High-level pipeline helpers
# ---------------------------------------------------------------------------

WIRE_MODEL_PATH = "models/unet_best.pth"
YOLO_MODEL_PATH = "models/yolo.pt"


def run_inference(
    image_path: str | Path,
    reconstructor: "SchematicReconstructor | None" = None,
    align: int | None = None,
    yolo_model_path: str = YOLO_MODEL_PATH,
    wire_model_path: str = WIRE_MODEL_PATH,
) -> tuple[Schematic, int]:
    """Run YOLO + wire detection + filtering + alignment on ``image_path``.

    Returns the finalised ``Schematic`` plus the raw polyline count (useful
    for logging). The image's detected component XML is also written to
    ``output/<stem>.xml`` as a side-effect, matching the existing pipeline.
    """
    from model_inference.wire_detect import detect_wires
    from model_inference.yolo_detection import detect_components
    from schematics.schematic import SchematicParser
    from model_inference.text_ocr import process_schematic_with_yolo

    reconstructor = reconstructor or SchematicReconstructor()
    image_path = Path(image_path)

    yolo_result = detect_components(
        model_path=yolo_model_path,
        image_path=str(image_path),
    )
    schematic = SchematicParser.from_yolo_to_schematic(yolo_result)
    schematic = process_schematic_with_yolo(
        schematic, model_dir=Path("models/trocr-schematic-final")
    )
    SchematicParser.save_to_xml(schematic, f"output/{image_path.stem}.xml")

    _cleaned, _erased, _skeleton, polylines = detect_wires(
        image_path, schematic, wire_model_path
    )

    schematic = reconstructor.filter_by_confidence(schematic)
    schematic = reconstructor.link_text_to_components(schematic)
    schematic = reconstructor.connect_components(schematic, polylines)
    # Text components are retained so annotate_labels can render the OCR'd
    # handwriting at its original bbox. draw_components/draw_lines both skip
    # text, and connect_components already excludes it when matching wire
    # endpoints to components.
    schematic = reconstructor.align_components(schematic, tolerance=align)
    return schematic, len(polylines)


def render_schematic(
    schematic: Schematic,
    labels: bool = True,
    reconstructor: "SchematicReconstructor | None" = None,
) -> np.ndarray:
    """Render ``schematic`` to a fresh BGR canvas and return it."""
    reconstructor = reconstructor or SchematicReconstructor()
    canvas = reconstructor.render_canvas(schematic)
    reconstructor.draw_components(canvas, schematic)
    reconstructor.draw_lines(canvas, schematic)
    if labels:
        reconstructor.annotate_labels(canvas, schematic)
    return canvas


if __name__ == "__main__":
    import argparse

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

        schematic, n_polys = run_inference(
            image_path,
            reconstructor=reconstructor,
            align=args.align,
        )
        canvas = render_schematic(
            schematic, labels=args.labels, reconstructor=reconstructor
        )
        reconstructor.export_image(canvas, output_path)

        print(
            f"Wrote {output_path}  (labels={'on' if args.labels else 'off'}, "
            f"polylines={n_polys})"
        )
        print(f"  Counts: {reconstructor.summarise(schematic)}")
