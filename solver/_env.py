"""Process-level environment setup. Import this BEFORE any torch import.

Windows ships multiple OpenMP runtimes (torch + numpy/MKL); without this flag
the process aborts with "OMP: Error #15" on import.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
