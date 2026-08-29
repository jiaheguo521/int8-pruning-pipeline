"""Parse `edgetpu_compiler` stdout into per-input-model memory stats.

Rescued from the retired `search_edgetpu_pair.py` (the abandoned two-model
co-residency search). Only this parser survived: the search driver around it is
in `attic/`, but `parse_compiler_stdout` is a live dependency of the Re-ID
report, so it lives here as shared infrastructure instead.

The compiler prints, per input model, an "Input model:" line followed by the
on-chip and off-chip byte counts. Both are reported in whatever unit the
compiler picked (B/KiB/MiB/GiB), so everything is normalised to MiB.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# One "Input model:" block per input, and within each block the on-chip / off-chip used lines.
_INPUT_RE = re.compile(r"^Input model:\s*(?P<path>.+)$")
_ON_CHIP_RE = re.compile(
    r"^On-chip memory used for caching model parameters:\s*(?P<val>[\d.]+)(?P<unit>B|KiB|MiB|GiB)\s*$"
)
_OFF_CHIP_RE = re.compile(
    r"^Off-chip memory used for streaming uncached model parameters:\s*(?P<val>[\d.]+)(?P<unit>B|KiB|MiB|GiB)\s*$"
)

_UNIT_TO_BYTES = {"B": 1.0, "KiB": 1024.0, "MiB": 1024.0**2, "GiB": 1024.0**3}


def _to_mib(val: float, unit: str) -> float:
    return val * _UNIT_TO_BYTES[unit] / _UNIT_TO_BYTES["MiB"]


@dataclass
class ModelMem:
    """Per-input-model memory stats parsed from edgetpu_compiler stdout."""

    input_path: str
    on_chip_mib: float
    off_chip_mib: float


def parse_compiler_stdout(stdout: str) -> list[ModelMem]:
    """Walk the compiler output and yield one ModelMem per `Input model:` block.

    The compiler prints the three lines in a stable order (Input -> On-chip used
    -> ... -> Off-chip used), so we track the current input and assign the
    matching memory values to it.
    """
    results: list[ModelMem] = []
    current: dict | None = None
    for line in stdout.splitlines():
        m = _INPUT_RE.match(line)
        if m:
            if current is not None:
                results.append(ModelMem(**current))
            current = {
                "input_path": m.group("path").strip(),
                "on_chip_mib": float("nan"),
                "off_chip_mib": float("nan"),
            }
            continue
        if current is None:
            continue
        m = _ON_CHIP_RE.match(line)
        if m:
            current["on_chip_mib"] = _to_mib(float(m.group("val")), m.group("unit"))
            continue
        m = _OFF_CHIP_RE.match(line)
        if m:
            current["off_chip_mib"] = _to_mib(float(m.group("val")), m.group("unit"))
            continue
    if current is not None:
        results.append(ModelMem(**current))
    return results
