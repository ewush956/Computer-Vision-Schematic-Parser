# Reconstructor — Status & Next Steps

Snapshot of what `SchematicReconstructor` (in
[`schematic_recontructor.py`](../schematic_recontructor.py)) can do today
and what still needs to be written before the end-to-end pipeline runs.

## 1. What is implemented

### 1.1 Data model — `schematic.py`
Complete. Dataclasses for `BoundingBox`, `Label`, `Component`, `Line`,
`Schematic`, with derived `width` / `height` / `center_x` / `center_y`
properties on the bounding box.

### 1.2 Symbol renderer — `schematic_symbols.py`
Complete for the 17 classes that appear in the two test XMLs. See
[`docs/symbol_coverage.md`](symbol_coverage.md) for the full coverage
report and the list of 43 classes that still need glyphs.

Public entry point: `draw_symbol(canvas, class_name, center, orientation,
size_scale, line_thickness)` → returns terminal points.

### 1.3 Reconstructor methods (EVAN — done)
| Method               | Status | Notes |
| -------------------- | :----: | ----- |
| `render_canvas`      | ✅ done | White NumPy BGR canvas, honours `scale`, remembers scale internally |
| `draw_components`    | ✅ done | Currently draws coloured *bounding rectangles*, not glyphs (see §3.1) |
| `draw_lines`         | ✅ done | Arrowed lines between component centers; skips missing IDs |
| `annotate_labels`    | ✅ done | Labels above or below each box, with confidence |
| `summarise`          | ✅ done | `Counter` over `class_name` |
| `export_image`       | ✅ done | Infers format from extension; PNG/JPEG/WEBP quality handled |
| `_color_for_class`   | ✅ done | Palette-cycled per-class colours |

## 2. What is stubbed (blocks the pipeline)

All of the following have the signature + docstring but the body is `...`.
Nothing in the `Typical usage` block of the module will actually run until
these are filled in.

### 2.1 Blocking — must be done before Evan's rendering code can run

| Method          | Owner    | Why it blocks rendering |
| --------------- | -------- | ----------------------- |
| `load_xml`      | DEPANSHU | Entry point — without it we have no `Schematic` to render. |
| `get_bounds`    | DEPANSHU | `draw_components` and `annotate_labels` both call it on every component. |
| `get_center`    | DEPANSHU | `draw_lines` calls it on every endpoint. |

> The two accessor methods are one-line wrappers over `component.bounding_box`.
> They are blocking only because the draw methods call them, not because
> the logic is hard.

### 2.2 Blocking — must be done before the default pipeline produces useful output

| Method                    | Owner    | Why |
| ------------------------- | -------- | --- |
| `filter_by_confidence`    | DEPANSHU | `Typical usage` calls it before linking/connecting. Without it the noisy low-confidence detections survive into rendering. |
| `filter_by_class`         | AMTOJ    | Needed to strip `"text"` components after linking so they don't appear as rectangles in the final image. |
| `link_text_to_components` | AMTOJ    | Populates `Component.label`; `annotate_labels` already reads this field and falls back to `class_name` when absent — working but uninformative. |
| `connect_components`      | DEPANSHU | Populates `Schematic.lines`. Without it, `draw_lines` is a no-op (loops over an empty list). |

## 3. What is not yet scoped

### 3.1 Glyph rendering instead of bounding-box rendering
Today `draw_components` draws a coloured **rectangle** per component.
The symbol renderers in `schematic_symbols.py` exist but are **not wired
into the reconstructor yet**. A future method (e.g. `draw_symbols`) should:

1. For each non-text component, derive orientation from `bounding_box`
   aspect ratio (`schematic_symbols.infer_orientation`).
2. Pick `size_scale` from the bounding-box size vs. the canonical size in
   `schematic_symbols.CANONICAL_SIZES`.
3. Call `schematic_symbols.draw_symbol(canvas, class_name, center, ...)`.
4. Use the returned terminal points to drive wire routing in
   `draw_lines` (replacing the current center-to-center arrows).

This is an enhancement, not a blocker — the current rectangle renderer
is enough to validate the rest of the pipeline.

### 3.2 Wire routing
`draw_lines` currently draws straight arrows between component centers.
For presentable output we'll want orthogonal (Manhattan-style) routing
that attaches to the nearest returned terminal point instead of the
center. Defer until `connect_components` is landed.

### 3.3 Fallback-symbol quick wins
Two one-line additions to `SYMBOL_REGISTRY` would upgrade prefix fallback
quality for common classes (`capacitor.polarized`, `transistor.bjt`, …).
See §3 of [`symbol_coverage.md`](symbol_coverage.md).

## 4. Critical path — what Evan needs from the team

In dependency order, the smallest set of stubs that unblocks an
end-to-end run on `tests/outputs/testcase1.xml`:

1. **DEPANSHU → `load_xml`** (parses the XML already produced by the
   pipeline into a `Schematic`).
2. **DEPANSHU → `get_bounds`** (one line: return the four bbox ints).
3. **DEPANSHU → `get_center`** (one line: return `(center_x, center_y)`).

After those three land, the full `Typical usage` block runs and produces
an image, though `draw_lines` will be empty and every non-text detection
will show as a rectangle. To get a *useful* image:

4. **DEPANSHU → `filter_by_confidence`** (drop noisy detections).
5. **AMTOJ → `filter_by_class`** (strip `"text"` after linking).
6. **AMTOJ → `link_text_to_components`** (populate labels).
7. **DEPANSHU → `connect_components`** (populate lines).

Once 1–7 exist, everything in `schematic_recontructor.py` is live and the
only remaining work is the glyph-rendering wiring in §3.1.

## 5. Test coverage to add after stubs land

`tests/test_reconstructor.py` already exists. After each stub is
implemented, add a focused test that:

- feeds a hand-written minimal `Schematic` (or a fixture XML) through the
  new function,
- asserts the returned `Schematic` has the expected `components` /
  `labels` / `lines` lists, and
- confirms the original `Schematic` argument is **not** mutated
  (every method is documented as non-mutating).
