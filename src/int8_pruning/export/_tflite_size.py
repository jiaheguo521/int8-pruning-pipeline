"""A `.tflite` file's bytes split into constant data and graph packaging.

Raw size can be dominated by tensor-name strings (onnx2tf names every split output with
the locations of all its siblings), a floor that edgetpu_compiler folds away; constant
bytes track the parameters. The schema comes from `ai_edge_litert` or tensorflow; without
either the counts are None.
"""

from pathlib import Path


def tflite_size(tflite_path):
    """Byte budget of `tflite_path`, as the matching fields of `ExportInt8Result`."""
    path = Path(tflite_path)
    rec = {"file_bytes": path.stat().st_size, "const_bytes": None, "name_bytes": None,
           "num_tensors": None, "num_operators": None, "num_subgraphs": None,
           "const_mib": None, "packaging_frac": None}
    # Imported here: the schema is optional, and without it the record stays at the file size.
    try:
        from ai_edge_litert import schema_py_generated as schema
    except ImportError:
        try:
            from tensorflow.lite.python import schema_py_generated as schema
        except ImportError:
            return rec

    raw = bytearray(path.read_bytes())
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))
    const = sum(len(b.data) for b in model.buffers if b.data is not None and len(b.data))
    rec.update(
        const_bytes=const,
        name_bytes=sum(len(t.name) for sg in model.subgraphs for t in sg.tensors if t.name),
        num_tensors=sum(len(sg.tensors) for sg in model.subgraphs),
        num_operators=sum(len(sg.operators) for sg in model.subgraphs),
        num_subgraphs=len(model.subgraphs),
        const_mib=round(const / 2**20, 4),
        packaging_frac=(round(1.0 - const / rec["file_bytes"], 4) if rec["file_bytes"]
                        else None))
    return rec
