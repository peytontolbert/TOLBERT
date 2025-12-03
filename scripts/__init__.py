"""
Local scripts package for TOLBERT utilities (dataset builders, training, eval).

This file exists so that tests and external tools can reliably import modules
like `scripts.build_wos_spans` without accidentally resolving to any unrelated
third-party `scripts` package on the Python path.
"""


