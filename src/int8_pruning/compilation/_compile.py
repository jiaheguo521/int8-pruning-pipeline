"""A wrapper over the `edgetpu_compiler` CLI, which writes `<stem>_edgetpu.tflite` and
`<stem>_edgetpu.log` for each input."""
import shutil
import subprocess
from pathlib import Path

__all__ = ["compile_edgetpu"]


def compile_edgetpu(tflite, out_dir=None, *, show_ops=False, extra_args=(),
                    compiler="edgetpu_compiler"):
    """Compile `tflite` for the Edge TPU into `out_dir`.

    Parameters
    ----------
    tflite : str or pathlib.Path, or list of them
        A list is compiled together, so the models share the on-chip parameter cache.
    out_dir : str or pathlib.Path, optional
        Default beside the (first) input.
    show_ops : bool, default False
        Also write each input's operator mapping to ``<stem>.show_ops.txt``.
    extra_args : sequence of str, default ()
        Passed to the compiler, e.g. ``("-m", "13")`` to pin the runtime version.
    compiler : str, default "edgetpu_compiler"

    Returns
    -------
    pathlib.Path or list of pathlib.Path
        The ``_edgetpu.tflite`` path(s), matching the input.

    Raises
    ------
    FileNotFoundError
        If `compiler` is not on PATH.
    subprocess.CalledProcessError
        If the compiler fails.
    """
    paths = [Path(p) for p in ([tflite] if isinstance(tflite, (str, Path)) else tflite)]
    if shutil.which(compiler) is None:
        raise FileNotFoundError(f"{compiler} not found; install it from Coral's apt "
                                f"repository: https://coral.ai/docs/edgetpu/compiler/")
    out = Path(out_dir) if out_dir is not None else paths[0].parent
    out.mkdir(parents=True, exist_ok=True)
    if show_ops:
        # Once per input, so each mapping lands in its own file. --show_operations compiles
        # too, into the working directory unless -o says otherwise.
        for p in paths:
            ops = subprocess.run(
                [compiler, "--show_operations", "-o", str(out), *extra_args, str(p)],
                check=True, capture_output=True, text=True).stdout
            (out / f"{p.stem}.show_ops.txt").write_text(ops)
    subprocess.run([compiler, "-o", str(out), *extra_args, *map(str, paths)], check=True)
    compiled = [out / f"{p.stem}_edgetpu.tflite" for p in paths]
    return compiled[0] if isinstance(tflite, (str, Path)) else compiled
