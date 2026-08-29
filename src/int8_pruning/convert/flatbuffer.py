"""Two .tflite rewrites that sit between ai-edge-quantizer and edgetpu_compiler 16.0.

ORDER MATTERS, and getting it wrong fails silently:

    litert-torch -> ai-edge-quantizer -> inline_buffers -> etpu_fixups -> edgetpu_compiler

Run ``etpu_fixups`` first and the model loses its weights. ai-edge-quantizer writes the
big constants as external buffers -- (offset, size) pairs past the end of the flatbuffer
-- and the ModelT round-trip only sees what is inside the flatbuffer, so it
re-serializes without them. The output still loads, still compiles, and is simply empty:
mobilenetv2 came out 0.58 MiB instead of 3.5 MiB, with 7.25 KiB on-chip and 4 of 70
operators mapped. ``inline_buffers`` must bring the constants inside first.

Neither rewrite is model knowledge; every one is a verifiable property of the graph and
refuses to fire unless its precondition holds.

Both were previously standalone scripts under results/protocol_audit/export_litert/,
where three separate investigations of classification, reid and line_seg independently
arrived at the same rewrites. They moved here when litert-torch became the pipeline's
only export path.

The `etpu` in ``etpu_fixups`` is "Edge TPU", not the old package name: it says which
compiler the rewrites target. It is also a key in every ``<stem>_int8.size.json``
sidecar (via ``export_litert.quantize_int8``), so renaming it would fork that
schema across build generations for nothing.
"""
import sys
from pathlib import Path

import numpy as np
import flatbuffers

# Same generated schema either way; ai_edge_litert first so this step needs no tensorflow.
try:
    from ai_edge_litert import schema_py_generated as schema
except ImportError:  # older env: fall back to the copy inside tensorflow
    from tensorflow.lite.python import schema_py_generated as schema


def _load(src: Path):
    raw = bytearray(Path(src).read_bytes())
    return raw, schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw, 0))


def _save(model, dst: Path) -> Path:
    builder = flatbuffers.Builder(1024 * 1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    dst = Path(dst)
    dst.write_bytes(bytes(builder.Output()))
    return dst


def inline_buffers(src: Path, dst: Path, verbose: bool = True) -> int:
    """Re-serialize with every external buffer moved back inline. Returns the count.

    ai-edge-quantizer emits buffers whose (offset, size) point past the end of the
    flatbuffer. edgetpu_compiler 16.0 ships an older TFLite parser that ignores those
    fields, so every weight reads as "not constant at compile-time" and only 3 of 345
    ops map. The bytes are all present, so inlining them is lossless.
    """
    raw, model = _load(src)
    moved = 0
    for b in model.buffers:
        off, size = getattr(b, "offset", 0), getattr(b, "size", 0)
        if off:
            b.data = list(raw[off: off + size])
            b.offset, b.size = 0, 0
            moved += 1
    out = _save(model, dst)
    if verbose:
        print(f"[inline] moved {moved} external buffers -> {out.name} "
              f"{out.stat().st_size / 2**20:.2f} MiB")
    return moved


def etpu_fixups(src: Path, dst: Path, verbose: bool = True) -> dict:
    """Three graph rewrites edgetpu_compiler 16.0 needs. Returns the fire counts.

    1. PAD/PADV2 ``paddings`` INT64 -> INT32.
       litert-torch emits this operand as INT64. edgetpu_compiler 16.0's bundled TFLite
       interpreter reads the buffer as INT32, so [[0,0],[1,1],[1,1],[0,0]] is consumed
       as [[0,0],[0,0],[0,0],[1,0]] -- it pads the CHANNEL axis. The next CONV_2D then
       sees 65 channels against a 64-channel filter:
         ERROR: :349 input->dims->data[3] != filter->dims->data[3] (65 != 64)
       Value-preserving re-encode.

    2. PADV2 -> PAD when the pad constant equals the input's zero point.
       The Edge TPU maps PAD and rejects PADV2 ("Operation not supported"). When the
       constant is the zero point the two ops compute the same thing, because TFLite's
       quantized PAD pads with the zero point. Fires only on that equality.

    3. Drop an identity RESHAPE -> GATHER_ND -> RESHAPE triple.
       to_channel_last_io lowers the global-pool -> flatten transition to a GATHER_ND
       whose indices are arange(N) -- a reshape written as a gather. The Edge TPU has no
       GATHER_ND, so it cuts the subgraph and strands the classifier on the CPU. Fires
       only when the indices are exactly arange and the tensors on either side of the
       triple have identical shape and quantization.

    Without this, resnet50 does not compile at all and mobilenetv2 / squeezenet1_1 /
    efficientnet_lite0 strand their final FULLY_CONNECTED on the CPU. line_seg's w96 and
    base128 do NOT need it (verified: they map 16/16 without), link_r34 does.
    """
    _, model = _load(src)
    BOP = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    NP = {schema.TensorType.INT32: np.int32, schema.TensorType.INT64: np.int64,
          schema.TensorType.INT8: np.int8, schema.TensorType.FLOAT32: np.float32}
    say = print if verbose else (lambda *a, **k: None)

    def name(op):
        oc = model.operatorCodes[op.opcodeIndex]
        return BOP.get(max(oc.builtinCode, oc.deprecatedBuiltinCode), "?")

    def const(sg, idx):
        t = sg.tensors[idx]
        b = model.buffers[t.buffer]
        data = getattr(b, "data", None)
        if data is None or len(data) == 0 or t.type not in NP:
            return None
        return np.frombuffer(bytes(data), dtype=NP[t.type])

    def qparams(sg, idx):
        q = sg.tensors[idx].quantization
        if q is None or q.scale is None or len(q.scale) == 0:
            return None
        return (list(q.scale), list(q.zeroPoint))

    def opcode_for(builtin):
        for i, oc in enumerate(model.operatorCodes):
            if max(oc.builtinCode, oc.deprecatedBuiltinCode) == builtin:
                return i
        oc = schema.OperatorCodeT()
        oc.builtinCode = builtin
        oc.deprecatedBuiltinCode = min(builtin, 127)
        oc.version = 1
        model.operatorCodes.append(oc)
        return len(model.operatorCodes) - 1

    n_dtype = n_padv2 = n_gather = 0

    for sg in model.subgraphs:
        # ---- 1. paddings INT64 -> INT32 -------------------------------------
        for op in sg.operators:
            if name(op) not in ("PAD", "PADV2", "MIRROR_PAD"):
                continue
            t = sg.tensors[op.inputs[1]]
            if t.type != schema.TensorType.INT64:
                continue
            v = const(sg, op.inputs[1])
            assert v is not None and v.min() >= -2**31 and v.max() < 2**31
            b = model.buffers[t.buffer]
            b.data = list(v.astype(np.int32).tobytes())
            b.offset, b.size = 0, 0
            t.type = schema.TensorType.INT32
            n_dtype += 1
            say(f"  [1] {name(op)} paddings t{op.inputs[1]}: INT64 -> INT32 "
                f"{v.reshape(-1, 2).tolist()}")

        # ---- 2. PADV2 -> PAD where the constant is the zero point -----------
        for op in sg.operators:
            if name(op) != "PADV2" or len(op.inputs) < 3:
                continue
            cv = const(sg, op.inputs[2])
            qp = qparams(sg, op.inputs[0])
            if cv is None or qp is None or len(cv) != 1:
                say("  [2] PADV2 left alone (non-constant pad value or float tensor)")
                continue
            zp = qp[1][0]
            if int(cv[0]) != int(zp):
                say(f"  [2] PADV2 left alone: pad value {int(cv[0])} != zero point {zp}")
                continue
            op.opcodeIndex = opcode_for(schema.BuiltinOperator.PAD)
            op.inputs = op.inputs[:2]
            op.builtinOptionsType = schema.BuiltinOptions.PadOptions
            op.builtinOptions = schema.PadOptionsT()
            n_padv2 += 1
            say(f"  [2] PADV2 -> PAD (pad value {int(cv[0])} == zero point {zp}, real 0.0)")

        # ---- 3. RESHAPE -> GATHER_ND(arange) -> RESHAPE  ---------------------
        producer, consumers = {}, {}
        for op in sg.operators:
            for o in op.outputs:
                producer[o] = op
            for i in op.inputs:
                consumers.setdefault(i, []).append(op)

        doomed = []
        for g in sg.operators:
            if name(g) != "GATHER_ND":
                continue
            params, idx_t, out = g.inputs[0], g.inputs[1], g.outputs[0]
            ind = const(sg, idx_t)
            n = int(np.prod([d for d in sg.tensors[params].shape]))
            if ind is None or list(sg.tensors[idx_t].shape)[-1] != 1 \
                    or len(ind) != n or not np.array_equal(ind, np.arange(n)):
                say("  [3] GATHER_ND left alone: indices are not arange")
                continue
            pre, post = producer.get(params), consumers.get(out, [])
            if pre is None or name(pre) != "RESHAPE" or len(post) != 1 \
                    or name(post[0]) != "RESHAPE":
                say("  [3] GATHER_ND left alone: not wrapped in RESHAPE/RESHAPE")
                continue
            a, b = pre.inputs[0], post[0].outputs[0]
            same_numel = (int(np.prod(list(sg.tensors[a].shape)))
                          == int(np.prod(list(sg.tensors[b].shape))))
            if not same_numel or qparams(sg, a) != qparams(sg, b):
                say("  [3] GATHER_ND left alone: triple is not element/quant preserving")
                continue
            if list(sg.tensors[a].shape) == list(sg.tensors[b].shape):
                # Identity: the triple computes nothing, so drop all three and read the input directly. (cls)
                for op in sg.operators:              # rewire consumers of b to a
                    op.inputs = [a if i == b else i for i in op.inputs]
                sg.outputs = [a if o == b else o for o in sg.outputs]
                doomed += [id(pre), id(g), id(post[0])]
                say(f"  [3] dropped identity RESHAPE->GATHER_ND(arange {n})->RESHAPE "
                    f"(t{a} {list(sg.tensors[a].shape)} passes straight through)")
            else:
                # Same elements in the same order under a different shape -- a pure reshape written as a
                # gather. Keep the trailing RESHAPE, which already carries b's target shape, and feed it
                # from a. (Re-ID: [1,2048] -> [1,1,1,2048])
                post[0].inputs = [a] + list(post[0].inputs[1:])
                doomed += [id(pre), id(g)]
                say(f"  [3] RESHAPE->GATHER_ND(arange {n})->RESHAPE collapsed to one "
                    f"RESHAPE (t{a} {list(sg.tensors[a].shape)} -> "
                    f"t{b} {list(sg.tensors[b].shape)})")
            n_gather += 1
        if doomed:
            sg.operators = [op for op in sg.operators if id(op) not in doomed]

    out = _save(model, dst)
    counts = {"int64_paddings": n_dtype, "padv2_to_pad": n_padv2,
              "identity_gather": n_gather}
    if verbose:
        print(f"[fixups] " + " ".join(f"{k}={v}" for k, v in counts.items())
              + f" -> {out.name} {out.stat().st_size / 2**20:.2f} MiB")
    return counts


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("inline", "fixups"):
        sys.exit("usage: python -m int8_pruning.convert.flatbuffer {inline|fixups} in.tflite out.tflite")
    fn = inline_buffers if sys.argv[1] == "inline" else etpu_fixups
    fn(Path(sys.argv[2]), Path(sys.argv[3]))
