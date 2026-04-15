import re


class SchematicTextClassifier:
    def __init__(self):
        self.power_nets = {
            "VDD",
            "VCC",
            "VSS",
            "GND",
            "AGND",
            "DGND",
            "PGND",
            "+5V",
            "-5V",
            "+3.3V",
            "+3V3",
            "3V3",
            "+1.8V",
            "+12V",
            "-12V",
            "VBAT",
            "VMCU",
            "AVCC",
            "AVDD",
            "DVDD",
            "IOVDD",
            "PWR",
            "VIN",
            "VOUT",
            "VREF",
            "VB",
            "VP",
            "VM",
        }

        self.pin_labels = {
            # Generic I/O
            "IN",
            "OUT",
            "IN+",
            "IN-",
            "INP",
            "INN",
            # Clocks and control
            "CLK",
            "CLKIN",
            "CLKOUT",
            "SCL",
            "SDA",
            "EN",
            "ENB",
            "OE",
            "OEB",
            "RST",
            "RESET",
            "SHDN",
            "CS",
            "CSB",
            "CE",
            "SS",
            # Serial interfaces
            "RX",
            "TX",
            "MOSI",
            "MISO",
            "SCK",
            # BJT / FET / diode terminals
            "B",
            "C",
            "E",  # BJT: Base, Collector, Emitter
            "G",
            "D",
            "S",  # MOSFET: Gate, Drain, Source
            "A",
            "K",  # Diode: Anode, Cathode
            # Op-amp / comparator
            "V+",
            "V-",
            "OUT",
            # Other common labels
            "NC",
            "TP",
            "Q",
            "I",
            "V",
            "W",
            # Interrupt / direction
            "INT",
            "IRQ",
            "DIR",
            "SEL",
            "MODE",
        }

        self.regex_reference = re.compile(
            r"^(LED|VR|RV|CR|DS|SW|TP|FB|TR|IC|[RCLQUDFJSXYTMKPF])\d{1,4}[A-Z]?$",
            re.IGNORECASE,
        )
        self.regex_value = re.compile(
            r"^[+-]?("
            # Code notation: 4R7, 2k2, 1n5  (letter between digits acts as decimal point)
            r"\d+[RrKkMmUuNnPp]\d+"
            r"|"
            # Standard: digits + SI multiplier + optional unit
            # e.g. 10k, 4.7uF, 100nH, 330R, 1kΩ, 0.1u, 10MHz, 3.3V, 100mA
            r"\d+(\.\d+)?(?:k|K|M|m|u|U|µ|n|N|p|P|f|F|R|r|Ω)"
            r"(?:F|H|z|V|A|Ω|Hz|hZ)?"  # optional explicit unit after multiplier
            r"|"
            # Frequency with unit written out: 10MHz, 4.096kHz, 32768Hz
            r"\d+(\.\d+)?(?:MHz|kHz|Hz)"
            r"|"
            # Current: 100mA, 50uA, 1A
            r"\d+(\.\d+)?(?:mA|uA|µA|A)"
            r"|"
            # Bare voltage (no + / - prefix above): 5V, 3.3V, 12V
            r"\d+(\.\d+)?[vV]"
            r")$"
        )

    _OCR_FIXES = str.maketrans(
        {
            "O": "0",  # capital-O → zero  (in numeric contexts this is checked below)
            "l": "1",  # lowercase-L → one
            "I": "1",  # capital-I → one   (in numeric contexts)
            "S": "5",  # S → 5             (e.g. "S6k" should be "56k" ... heuristic)
            "B": "8",  # B → 8             (in numeric contexts)
            "µ": "u",  # unicode micro → ASCII u  (simplifies regex)
            "Ω": "R",  # ohm symbol → R   (common schematic shorthand)
            "ω": "R",
        }
    )

    def _normalise(self, text: str) -> str:
        """
        Light normalisation pass before classification.
        Only applies OCR fixes when the character sits between digits,
        to avoid mangling genuine letter-only labels like 'B' (base pin).
        """
        # Strip whitespace and collapse any internal spaces
        text = re.sub(r"\s+", "", text.strip())

        # Replace unicode micro / ohm globally (safe, unit-only characters)
        text = text.replace("µ", "u").replace("Ω", "R").replace("ω", "R")

        return text

    def classify(self, raw_text: str) -> str | None:
        """
        Takes a raw OCR string and returns its semantic type, or None if noise.

        Return values
        -------------
        "power_net"   – named supply rail (GND, VCC, +3.3V …)
        "pin_label"   – functional signal name (EN, CLK, MOSI …)
        "reference"   – component reference designator (R1, U3A, C12 …)
        "value"       – component value (10k, 4.7uF, 3.3V, 10MHz …)
        None          – unrecognised / noise
        """
        clean = self._normalise(raw_text)
        upper = clean.upper()

        # Power nets — match case-insensitively against the canonical set
        if upper in {n.upper() for n in self.power_nets}:
            return "power_net"

        # Pin labels — exact uppercase match
        if upper in self.pin_labels:
            return "pin_label"

        # Reference designator
        if self.regex_reference.match(clean):
            return "reference"

        # Component value
        if self.regex_value.match(clean):
            return "value"

        return None

