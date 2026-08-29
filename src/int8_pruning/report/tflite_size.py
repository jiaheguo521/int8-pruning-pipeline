"""Split a `.tflite` file's bytes into constant data vs graph packaging.

Why this exists: the raw file size of a `.tflite` is not a proxy for how much
model it contains, and on one family it is off by 3x. Measured 2026-08-18 on
`efficientdet_lite1_..._int8.tflite` (13,000,704 B):

    constant data (all non-empty Buffers)   4,510,008 B   34.7%
    tensor NAME strings                     7,080,421 B   54.5%
    remaining flatbuffer/graph structure    1,410,275 B   10.8%

onnx2tf lowers the BiFPN's 32 depthwise 3x3 convolutions (2,816 channels) into
SPLIT -> 2,816 single-channel CONV_2D -> CONCAT, and MLIR names every split
output with the fused location list of all 88 siblings -- up to 2,454 chars
each. `edgetpu_compiler` folds all of it away (11,916 tensors -> 680, names
down to 0.5%), so the deployed artifact is unaffected; only the intermediate
file, and therefore any "size" column computed from it, is wrong.

The failure mode this guards against: those name bytes are a FIXED floor,
because the BiFPN sits in `ignored_layers` and is never pruned. Reporting raw
file size makes a 43.9% parameter reduction look like 13.8%. Constant-data
bytes track parameters to within ~2 points on every family measured, so that
is the number to report when the compiled artifact is not available.

Optional dependency: the flatbuffer schema, read from `ai_edge_litert` (the
same generated module TensorFlow ships, and the one the rest of the convert
path already uses -- see int8_pruning.convert.flatbuffer). Without it `measure()`
returns a record with `const_bytes=None` rather than raising, in the same
spirit as `inspect_tflite`'s interpreter fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TfliteSize:
    """Byte budget of one `.tflite`. `None` fields mean the schema was absent."""

    file_bytes: int
    const_bytes: int | None = None
    name_bytes: int | None = None
    num_tensors: int | None = None
    num_operators: int | None = None
    num_subgraphs: int | None = None

    @property
    def const_mib(self) -> float | None:
        if self.const_bytes is None:
            return None
        return self.const_bytes / (1024.0 ** 2)

    @property
    def packaging_frac(self) -> float | None:
        """Fraction of the file that is NOT constant data. >0.5 is pathological."""
        if self.const_bytes is None or not self.file_bytes:
            return None
        return 1.0 - self.const_bytes / self.file_bytes

    def as_dict(self) -> dict:
        d = asdict(self)
        d["const_mib"] = None if self.const_mib is None else round(self.const_mib, 4)
        d["packaging_frac"] = (None if self.packaging_frac is None
                               else round(self.packaging_frac, 4))
        return d


def measure(tflite_path) -> TfliteSize:
    """Byte budget of `tflite_path`, degrading gracefully without the schema."""
    path = Path(tflite_path)
    rec = TfliteSize(file_bytes=path.stat().st_size)
    try:
        from ai_edge_litert import schema_py_generated as schema
    except ImportError:  # older env: fall back to the copy inside tensorflow
        try:
            from tensorflow.lite.python import schema_py_generated as schema
        except (ImportError, AttributeError):
            return rec

    buf = bytearray(path.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(buf, 0))
    rec.num_subgraphs = len(model.subgraphs)
    rec.const_bytes = sum(len(b.data) for b in model.buffers
                          if b.data is not None and len(b.data))
    rec.name_bytes = sum(len(t.name) for sg in model.subgraphs
                         for t in sg.tensors if t.name)
    rec.num_tensors = sum(len(sg.tensors) for sg in model.subgraphs)
    rec.num_operators = sum(len(sg.operators) for sg in model.subgraphs)
    return rec
