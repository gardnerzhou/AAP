"""Training, validation, and testing routines for AAP.

The implementation is stored in compressed form in `_function_impl.py.xz`
to keep the public repository compact. It is decompressed in memory at import
time; no files are written to disk.
"""
from pathlib import Path
import lzma

_impl_path = Path(__file__).with_name("_function_impl.py.xz")
_source = lzma.decompress(_impl_path.read_bytes()).decode("utf-8")
exec(compile(_source, str(Path(__file__)), "exec"), globals(), globals())
