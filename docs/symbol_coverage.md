# Schematic Symbol Renderer — Coverage Report

Status of procedural renderers in `schematic_symbols.py` relative to the
62-class label set of `data/best.pt` (YOLO schematic detector).

- Total model classes: **62** (IDs 0–61)
- Excluded from rendering by design: `__background__` (non-class) and `text`
  (rendered separately by the annotator, never via `draw_symbol`)
- Candidate symbol classes: **60**
- Currently implemented: **17**
- Currently unimplemented: **43**

## 1. Implemented (17)

These are the non-text classes that appear in `tests/outputs/testcase1.xml`
and `tests/outputs/testcase2.xml`. All render black-on-white with terminal
points returned for downstream wire routing.

| Class ID | Class name               | Renderer                  |
| -------: | ------------------------ | ------------------------- |
|        2 | `junction`               | `draw_junction`           |
|        3 | `crossover`              | `draw_crossover`          |
|        4 | `terminal`               | `draw_terminal`           |
|        5 | `gnd`                    | `draw_gnd`                |
|        6 | `vss`                    | `draw_vss`                |
|        9 | `voltage.battery`        | `draw_voltage_battery`    |
|       10 | `resistor`               | `draw_resistor`           |
|       11 | `resistor.adjustable`    | `draw_resistor_adjustable`|
|       13 | `capacitor.unpolarized`  | `draw_capacitor_unpolarized` |
|       16 | `inductor`               | `draw_inductor`           |
|       19 | `transformer`            | `draw_transformer`        |
|       20 | `diode`                  | `draw_diode`              |
|       23 | `diode.zener`            | `draw_diode_zener`        |
|       29 | `transistor.fet`         | `draw_transistor_fet`     |
|       31 | `operational_amplifier`  | `draw_operational_amplifier` |
|       34 | `integrated_circuit`     | `draw_integrated_circuit` |
|       46 | `switch`                 | `draw_switch`             |

Coverage of `testcase1.xml`: **100%** of non-text detections.
Coverage of `testcase2.xml`: **100%** of non-text detections.

## 2. Unimplemented (43)

Grouped by family. "Falls back via prefix" means the dispatcher's
`split(".", 1)[0]` rule already picks up a reasonable visual (e.g.
`resistor.photo` → draws as plain `resistor`). Classes marked "fallback
rectangle" currently render as a labeled box.

### 2.1 Voltage sources (2)

| Class ID | Class name     | Suggested symbol                       | Current behavior  |
| -------: | -------------- | -------------------------------------- | ----------------- |
|        7 | `voltage.dc`   | Circle with `+` / `−` terminals        | Fallback rectangle |
|        8 | `voltage.ac`   | Circle with sine-wave glyph inside     | Fallback rectangle |

### 2.2 Capacitors (2)

| Class ID | Class name              | Suggested symbol                             | Current behavior  |
| -------: | ----------------------- | -------------------------------------------- | ----------------- |
|       14 | `capacitor.polarized`   | One straight plate + one curved plate, `+`   | Fallback rectangle (prefix `capacitor` not registered) |
|       15 | `capacitor.adjustable`  | Two plates with diagonal arrow through them  | Fallback rectangle |

### 2.3 Resistors (1)

| Class ID | Class name        | Suggested symbol                           | Current behavior     |
| -------: | ----------------- | ------------------------------------------ | -------------------- |
|       12 | `resistor.photo`  | Zigzag resistor inside a circle with two arrows pointing in | Falls back via prefix to `resistor` (acceptable) |

### 2.4 Inductors (2)

| Class ID | Class name          | Suggested symbol                          | Current behavior |
| -------: | ------------------- | ----------------------------------------- | ---------------- |
|       17 | `inductor.ferrite`  | Coil with two parallel lines above it      | Falls back via prefix to `inductor` (acceptable) |
|       18 | `inductor.coupled`  | Two inductor coils with parallel core bars | Falls back via prefix to `inductor` (poor — looks nothing like coupled inductor) |

### 2.5 Diodes (2)

| Class ID | Class name              | Suggested symbol                              | Current behavior |
| -------: | ----------------------- | --------------------------------------------- | ---------------- |
|       21 | `diode.light_emitting`  | Diode + two outward arrows (emission)          | Falls back via prefix to `diode` |
|       22 | `diode.thyrector`       | Two zener-style bars back-to-back on one triangle | Falls back via prefix to `diode` |

### 2.6 Thyristor family (4)

| Class ID | Class name     | Suggested symbol                                        | Current behavior |
| -------: | -------------- | ------------------------------------------------------- | ---------------- |
|       24 | `diac`         | Two triangles point-to-point, no gate                   | Fallback rectangle |
|       25 | `triac`        | Two triangles point-to-point with gate lead             | Fallback rectangle |
|       26 | `thyristor`    | Diode triangle + bar with gate lead off the triangle    | Fallback rectangle |
|       27 | `varistor`     | Zigzag resistor with `U` stroked across it (VDR)        | Fallback rectangle |

### 2.7 Transistors (2)

| Class ID | Class name          | Suggested symbol                                      | Current behavior |
| -------: | ------------------- | ----------------------------------------------------- | ---------------- |
|       28 | `transistor.bjt`    | Circle (or bare) with base, collector, emitter + arrow on emitter | Fallback rectangle (prefix `transistor` not registered) |
|       30 | `transistor.photo`  | BJT with two inward arrows on base lead                | Fallback rectangle |

### 2.8 Op-amp variants (1)

| Class ID | Class name                              | Suggested symbol                     | Current behavior |
| -------: | --------------------------------------- | ------------------------------------ | ---------------- |
|       32 | `operational_amplifier.schmitt_trigger` | Triangle with hysteresis-loop glyph  | Falls back via prefix to `operational_amplifier` (decent) |

### 2.9 Optocoupler / coupler (1)

| Class ID | Class name     | Suggested symbol                                        | Current behavior |
| -------: | -------------- | ------------------------------------------------------- | ---------------- |
|       33 | `optocoupler`  | LED on left + photodiode or phototransistor on right inside dashed box | Fallback rectangle |

### 2.10 IC variants (2)

| Class ID | Class name                                 | Suggested symbol                              | Current behavior |
| -------: | ------------------------------------------ | --------------------------------------------- | ---------------- |
|       35 | `integrated_circuit.ne555`                 | DIP rectangle labeled `NE555` (8 pins)         | Falls back via prefix to `integrated_circuit` (acceptable — text label missing) |
|       36 | `integrated_circuit.voltage_regulator`     | DIP rectangle labeled `VREG` (3 pins)          | Falls back via prefix to `integrated_circuit` (acceptable — pin count wrong) |

### 2.11 Logic gates (6)

All use the standard distinctive-shape ANSI convention: buffer/NOT triangles,
AND D-shape, OR shield shape; invert versions add a bubble; XOR/XNOR add a
second arc behind OR/NOR.

| Class ID | Class name | Suggested symbol          | Current behavior   |
| -------: | ---------- | ------------------------- | ------------------ |
|       37 | `xor`      | OR shape + extra arc      | Fallback rectangle |
|       38 | `and`      | D-shape                   | Fallback rectangle |
|       39 | `or`       | Curved-back shield        | Fallback rectangle |
|       40 | `not`      | Triangle + output bubble  | Fallback rectangle |
|       41 | `nand`     | AND + output bubble       | Fallback rectangle |
|       42 | `nor`      | OR + output bubble        | Fallback rectangle |

### 2.12 Probes (3)

| Class ID | Class name        | Suggested symbol                                   | Current behavior |
| -------: | ----------------- | -------------------------------------------------- | ---------------- |
|       43 | `probe`           | Arrowhead terminator                                | Fallback rectangle |
|       44 | `probe.current`   | Arrow + `I` label                                   | Falls back via prefix to `probe` → fallback rectangle |
|       45 | `probe.voltage`   | Arrow + `V` label                                   | Falls back via prefix to `probe` → fallback rectangle |

### 2.13 Electromechanical / discrete (9)

| Class ID | Class name    | Suggested symbol                                                 | Current behavior |
| -------: | ------------- | ---------------------------------------------------------------- | ---------------- |
|       47 | `relay`       | Coil (inductor) + adjacent switch contact                        | Fallback rectangle |
|       48 | `socket`      | Open-circle pair or angle-bracket pair                            | Fallback rectangle |
|       49 | `fuse`        | Rectangle with horizontal line through it (or sine line)          | Fallback rectangle |
|       50 | `speaker`     | Rectangle + triangle (cone) combo                                 | Fallback rectangle |
|       51 | `motor`       | Circle with `M` inside                                            | Fallback rectangle |
|       52 | `lamp`        | Circle with `X` inside                                            | Fallback rectangle |
|       53 | `microphone`  | Circle with vertical bar (diaphragm)                              | Fallback rectangle |
|       54 | `antenna`     | Triangle with one vertex on the base lead                         | Fallback rectangle |
|       55 | `crystal`     | Two parallel plates with a small rectangle between                | Fallback rectangle |

### 2.14 Meta / filler categories (6)

These labels are used in the training dataset for annotations that are
not standardized electrical symbols. Rendering them as the labeled
rectangle fallback is probably correct long-term — they are best treated
as bounding-box overlays rather than authored symbols.

| Class ID | Class name     | Recommended handling        |
| -------: | -------------- | --------------------------- |
|       56 | `mechanical`   | Keep fallback rectangle     |
|       57 | `magnetic`     | Keep fallback rectangle     |
|       58 | `optical`      | Keep fallback rectangle     |
|       59 | `block`        | Keep fallback rectangle     |
|       60 | `explanatory`  | Keep fallback rectangle     |
|       61 | `unknown`      | Keep fallback rectangle     |

## 3. Quick wins

Two small changes to `SYMBOL_REGISTRY` would materially improve fallback
quality without writing new renderers:

1. Register `"capacitor"` → `draw_capacitor_unpolarized`.
   This upgrades the prefix fallback for `capacitor.polarized` and
   `capacitor.adjustable` from a rectangle to a plausible capacitor glyph.
2. Register `"transistor"` → `draw_transistor_fet`.
   Upgrades `transistor.bjt` and `transistor.photo` from a rectangle to a
   plausible transistor glyph (wrong family, but visually legible).

These are one-line additions and are safe because the exact-match lookup
always wins over the prefix fallback.

## 4. Priority order for future work

If we want to close the gap efficiently:

1. **High schematic frequency + distinct visual**
   `voltage.dc`, `voltage.ac`, `capacitor.polarized`, `transistor.bjt`,
   `fuse`, `lamp`, `motor`, `speaker`.
2. **Logic gates (batch of 6)**
   `and`, `or`, `not`, `nand`, `nor`, `xor` — share geometry, worth
   authoring as one module.
3. **Probes and specialty diodes**
   `probe`, `diode.light_emitting`, `diode.zener` (already done as ref).
4. **Thyristor family + optocoupler + relay**
   Lower frequency but visually unique; no good fallback exists today.
5. **Remaining metaclass rendering**
   Leave as labeled rectangle.
