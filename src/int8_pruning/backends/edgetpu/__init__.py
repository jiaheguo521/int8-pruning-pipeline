"""Code that needs the Edge TPU toolchain or a Coral device to run.

That is the whole membership rule, and it is what makes the backend optional:
everything outside this directory produces int8 TFLite on a plain machine, with
no edgetpu_compiler installed and no accelerator plugged in.

Nothing else in `int8_pruning` imports from here. Check it:

    grep -rn backends src/int8_pruning --include='*.py' | grep -v '^src/int8_pruning/backends/'

Note what is deliberately NOT here. `convert/flatbuffer.py` rewrites a .tflite for
edgetpu_compiler 16.0's older parser, but it needs neither the compiler nor a
device, and `quantize_int8` calls it unconditionally -- every committed
measurement is on its output. The same is true of the four model-side rewrites in
`convert/export_litert.py`. Motivation is not the rule; runtime requirement is.
"""
